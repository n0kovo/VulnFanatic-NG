"""Loading, validation and matching for VulnFanatic-NG rule files.

Phase 1 rules describe dangerous *functions* to find calls to. Phase 2 rules
describe security-sensitive *functions* to audit, matched by name keyword/regex
and/or referenced string constants. Both files share an envelope with a shared
``system_prompt`` and ``output_schema``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Pattern, Sequence

BUILTIN_DIR = os.path.join(os.path.dirname(__file__), "rules")
BUILTIN_PHASE1 = os.path.join(BUILTIN_DIR, "phase1_rules.json")
BUILTIN_PHASE2 = os.path.join(BUILTIN_DIR, "phase2_rules.json")
BUILTIN_PHASE3 = os.path.join(BUILTIN_DIR, "phase3_rules.json")

VALID_SEVERITIES = {"critical", "high", "medium", "low", "info"}


class RuleError(Exception):
    """Raised when a rule file is malformed."""


def _compile_all(patterns: Sequence[str]) -> List[Pattern]:
    compiled = []
    for p in patterns or []:
        try:
            compiled.append(re.compile(p))
        except re.error as exc:
            raise RuleError(f"Invalid regex {p!r}: {exc}") from exc
    return compiled


@dataclass
class Phase1Rule:
    id: str
    name: str
    severity: str
    cwe: str
    prompt: str
    functions: List[str] = field(default_factory=list)
    name_regex: List[Pattern] = field(default_factory=list)
    # When true, a call site whose arguments are all compile-time constants is
    # skipped before any LLM call (it cannot be attacker-controlled). Only safe
    # for overflow-class rules where a constant source/size is genuinely benign.
    prefilter_const_safe: bool = False
    # Declarative config for the no-LLM "Scan Offline" mode (see offline.py).
    # None/empty means the rule is not evaluated offline.
    offline: dict = field(default_factory=dict)

    def __post_init__(self):
        self._function_set = {f for f in self.functions}

    def matches_name(self, *names: Optional[str]) -> bool:
        """True if any of the supplied names (raw, demangled, short) matches."""
        for name in names:
            if not name:
                continue
            if name in self._function_set:
                return True
            for rx in self.name_regex:
                if rx.search(name):
                    return True
        return False


@dataclass
class Phase2Rule:
    id: str
    name: str
    severity: str
    cwe: str
    prompt: str
    name_keywords: List[str] = field(default_factory=list)
    name_regex: List[Pattern] = field(default_factory=list)
    string_keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._name_keywords_lc = [k.lower() for k in self.name_keywords]
        self._string_keywords_lc = [k.lower() for k in self.string_keywords]

    def matches(self, names: Iterable[Optional[str]], strings: Iterable[str]) -> bool:
        name_blob = " ".join(n.lower() for n in names if n)
        for kw in self._name_keywords_lc:
            if kw in name_blob:
                return True
        for n in names:
            if not n:
                continue
            for rx in self.name_regex:
                if rx.search(n):
                    return True
        if self._string_keywords_lc:
            for s in strings:
                sl = s.lower()
                for kw in self._string_keywords_lc:
                    if kw in sl:
                        return True
        return False


@dataclass
class RuleSet:
    version: int
    phase: int
    system_prompt: str
    output_schema: str
    rules: list
    tainted_sources: List[str] = field(default_factory=list)

    @property
    def tainted_source_set(self):
        return set(self.tainted_sources)


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise RuleError(f"{where}: missing required key {key!r}")
    return d[key]


def _validate_severity(sev: str, where: str) -> str:
    if sev not in VALID_SEVERITIES:
        raise RuleError(
            f"{where}: invalid severity {sev!r} (expected one of {sorted(VALID_SEVERITIES)})"
        )
    return sev


def _load_json(path: str) -> dict:
    if not os.path.isfile(path):
        raise RuleError(f"Rule file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise RuleError(f"Failed to read rule file {path}: {exc}") from exc


def _parse_phase1(data: dict, where: str) -> RuleSet:
    rules: List[Phase1Rule] = []
    raw_rules = _require(data, "rules", where)
    seen_ids = set()
    for idx, r in enumerate(raw_rules):
        rwhere = f"{where} rule[{idx}]"
        rid = _require(r, "id", rwhere)
        if rid in seen_ids:
            raise RuleError(f"{rwhere}: duplicate rule id {rid!r}")
        seen_ids.add(rid)
        functions = r.get("functions", []) or []
        name_regex = r.get("name_regex", []) or []
        if not functions and not name_regex:
            raise RuleError(f"{rwhere}: needs at least one of 'functions' or 'name_regex'")
        rules.append(
            Phase1Rule(
                id=rid,
                name=_require(r, "name", rwhere),
                severity=_validate_severity(r.get("severity", "info"), rwhere),
                cwe=r.get("cwe", ""),
                prompt=_require(r, "prompt", rwhere),
                functions=list(functions),
                name_regex=_compile_all(name_regex),
                prefilter_const_safe=bool(r.get("prefilter_const_safe", False)),
                offline=dict(r.get("offline") or {}),
            )
        )
    return RuleSet(
        version=int(data.get("version", 1)),
        phase=1,
        system_prompt=_require(data, "system_prompt", where),
        output_schema=_require(data, "output_schema", where),
        rules=rules,
        tainted_sources=list(data.get("tainted_sources", []) or []),
    )


def _parse_phase2(data: dict, where: str, phase: int = 2) -> RuleSet:
    rules: List[Phase2Rule] = []
    raw_rules = _require(data, "rules", where)
    seen_ids = set()
    for idx, r in enumerate(raw_rules):
        rwhere = f"{where} rule[{idx}]"
        rid = _require(r, "id", rwhere)
        if rid in seen_ids:
            raise RuleError(f"{rwhere}: duplicate rule id {rid!r}")
        seen_ids.add(rid)
        name_keywords = r.get("name_keywords", []) or []
        name_regex = r.get("name_regex", []) or []
        string_keywords = r.get("string_keywords", []) or []
        if not name_keywords and not name_regex and not string_keywords:
            raise RuleError(
                f"{rwhere}: needs at least one of 'name_keywords', 'name_regex', "
                "or 'string_keywords'"
            )
        rules.append(
            Phase2Rule(
                id=rid,
                name=_require(r, "name", rwhere),
                severity=_validate_severity(r.get("severity", "info"), rwhere),
                cwe=r.get("cwe", ""),
                prompt=_require(r, "prompt", rwhere),
                name_keywords=list(name_keywords),
                name_regex=_compile_all(name_regex),
                string_keywords=list(string_keywords),
            )
        )
    return RuleSet(
        version=int(data.get("version", 1)),
        phase=phase,
        system_prompt=_require(data, "system_prompt", where),
        output_schema=_require(data, "output_schema", where),
        rules=rules,
    )


def load_phase1_rules(path: str = "") -> RuleSet:
    target = path.strip() or BUILTIN_PHASE1
    return _parse_phase1(_load_json(target), f"phase1({os.path.basename(target)})")


def load_phase2_rules(path: str = "") -> RuleSet:
    target = path.strip() or BUILTIN_PHASE2
    return _parse_phase2(_load_json(target), f"phase2({os.path.basename(target)})")


def load_phase3_rules(path: str = "") -> RuleSet:
    target = path.strip() or BUILTIN_PHASE3
    return _parse_phase2(_load_json(target), f"phase3({os.path.basename(target)})", phase=3)
