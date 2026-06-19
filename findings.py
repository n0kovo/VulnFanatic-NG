"""Finding data model, BNDB persistence, and export helpers."""

from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Optional

METADATA_KEY = "vulnfanatic_ng.findings"   # legacy: a single scan's findings
SCANS_KEY = "vulnfanatic_ng.scans"         # current: list of named scans

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
# "unknown" is assigned programmatically when the model returned no parseable verdict
# (an "Unscored" lead); it ranks below low so such leads sort to the bottom.
CONFIDENCE_UNKNOWN = "unknown"
_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2, "unknown": 3}

# Triage workflow: every finding starts Untriaged; the analyst marks it.
TRIAGE_UNTRIAGED = "untriaged"
TRIAGE_REAL = "real_issue"
TRIAGE_FALSE = "false_positive"
# Statuses assigned automatically (not by the analyst) during a DEBUG online scan so
# the full set of analyzed candidates is visible. Normally these candidates are
# dropped silently; in debug mode they are kept and shown with one of:
#   REJECTED — the LLM returned a verdict of "not an issue"
#   SKIPPED  — eliminated before the LLM by a provably-safe rule (constant format/args)
#   ERROR    — the site could not be analyzed (context build failed, or the LLM
#              response was unparseable / the connection failed)
# The analyst can still re-triage any of them (Real / False).
TRIAGE_REJECTED = "rejected"
TRIAGE_SKIPPED = "skipped"
TRIAGE_ERROR = "error"
# Statuses that are NOT genuine candidate issues (debug-only bookkeeping rows).
TRIAGE_NON_ISSUE = (TRIAGE_REJECTED, TRIAGE_SKIPPED, TRIAGE_ERROR)
TRIAGE_STATUSES = (TRIAGE_REAL, TRIAGE_UNTRIAGED, TRIAGE_FALSE,
                   TRIAGE_REJECTED, TRIAGE_SKIPPED, TRIAGE_ERROR)
# The debug-only rows sort below everything else (lowest signal).
_TRIAGE_ORDER = {
    TRIAGE_REAL: 0, TRIAGE_UNTRIAGED: 1, TRIAGE_FALSE: 2,
    TRIAGE_REJECTED: 3, TRIAGE_SKIPPED: 4, TRIAGE_ERROR: 5,
}
TRIAGE_LABELS = {
    TRIAGE_UNTRIAGED: "Untriaged",
    TRIAGE_REAL: "Real Issue",
    TRIAGE_FALSE: "False Positive",
    TRIAGE_REJECTED: "REJECTED",
    TRIAGE_SKIPPED: "SKIPPED",
    TRIAGE_ERROR: "ERROR",
}


def _now_iso() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()


@dataclass
class Finding:
    phase: int
    rule_id: str
    severity: str
    confidence: str
    title: str
    function_name: str
    function_addr: int
    address: int
    cwe: str = ""
    explanation: str = ""
    tainted_input: str = ""
    recommendation: str = ""
    representation: str = ""
    call_paths: str = ""
    context_excerpt: str = ""
    scratchpad: str = ""
    validation_notes: str = ""
    prompt_file: str = ""
    model: str = ""
    triage_status: str = TRIAGE_UNTRIAGED
    triage_reason: str = ""
    created_at: str = field(default_factory=_now_iso)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"p{self.phase}-{self.rule_id}-{self.address:#x}"

    # Triage helpers ---------------------------------------------------------
    @property
    def false_positive(self) -> bool:
        return self.triage_status == TRIAGE_FALSE

    @property
    def triage_rank(self) -> int:
        return _TRIAGE_ORDER.get(self.triage_status, 1)

    @property
    def triage_label(self) -> str:
        return TRIAGE_LABELS.get(self.triage_status, "Untriaged")

    # Sorting helpers --------------------------------------------------------
    @property
    def severity_rank(self) -> int:
        return _SEVERITY_ORDER.get(self.severity, 99)

    @property
    def confidence_rank(self) -> int:
        return _CONFIDENCE_ORDER.get(self.confidence, 99)

    @property
    def sort_key(self):
        # Triage status leads (Real Issue first, then Untriaged, False Positive
        # last); then confidence (the headline metric).
        return (self.triage_rank, self.confidence_rank,
                self.severity_rank, self.function_name, self.address)

    # Serialization ----------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        d = dict(d)
        # Migrate the legacy boolean false_positive flag to a triage status.
        if "triage_status" not in d and "false_positive" in d:
            d["triage_status"] = TRIAGE_FALSE if d.get("false_positive") else TRIAGE_UNTRIAGED
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# Persistence ----------------------------------------------------------------

def save(bv, findings: List[Finding]) -> bool:
    """Persist findings into the BinaryView metadata (saved into the BNDB).

    Stored as non-auto metadata so it is written into the .bndb when the database
    is saved and restored on reopen.
    """
    if bv is None:
        return False
    payload = json.dumps([f.to_dict() for f in findings])
    try:
        # isAuto=False -> persisted with the database.
        bv.store_metadata(METADATA_KEY, payload, False)
    except TypeError:
        # Older API without the isAuto parameter.
        try:
            bv.store_metadata(METADATA_KEY, payload)
        except Exception:
            return False
    except Exception:
        return False
    _flush_to_disk(bv)
    return True


def _flush_to_disk(bv) -> None:
    """Best-effort: write the metadata through to the .bndb if one already exists.

    Does nothing (and never raises) when the binary has no database yet — in that
    case the findings live in the BinaryView and are written when the user saves
    the database.
    """
    try:
        if getattr(bv.file, "has_database", False):
            bv.save_auto_snapshot()
    except Exception:
        pass


def load(bv) -> List[Finding]:
    if bv is None:
        return []
    try:
        raw = bv.query_metadata(METADATA_KEY)
    except Exception:
        # KeyError when the key is absent; other errors are non-fatal here.
        return []
    if not raw:
        return []
    try:
        items = json.loads(raw)
        return [Finding.from_dict(d) for d in items]
    except Exception:
        return []


def clear(bv) -> None:
    if bv is None:
        return
    try:
        bv.remove_metadata(METADATA_KEY)
    except Exception:
        pass


def sort_findings(findings: List[Finding]) -> List[Finding]:
    return sorted(findings, key=lambda f: f.sort_key)


def clip_context(text: str, max_chars: int) -> str:
    """Truncate the per-finding context to ``max_chars`` characters; ``max_chars``
    of 0 (or negative) means keep the full context."""
    if max_chars and max_chars > 0:
        return (text or "")[:max_chars]
    return text or ""


# Scans (multiple named scans persisted in the BNDB) -------------------------

@dataclass
class ScanRecord:
    name: str
    mode: str = "online"            # online | offline | test
    rules_phase1_path: str = ""
    rules_phase2_path: str = ""
    created_at: str = field(default_factory=_now_iso)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    findings: List[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "rules_phase1_path": self.rules_phase1_path,
            "rules_phase2_path": self.rules_phase2_path,
            "created_at": self.created_at,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScanRecord":
        return cls(
            name=d.get("name", "scan"),
            mode=d.get("mode", "online"),
            rules_phase1_path=d.get("rules_phase1_path", ""),
            rules_phase2_path=d.get("rules_phase2_path", ""),
            created_at=d.get("created_at", _now_iso()),
            id=d.get("id") or uuid.uuid4().hex[:12],
            findings=[Finding.from_dict(x) for x in d.get("findings", []) or []],
        )


def save_scans(bv, scans: List["ScanRecord"]) -> bool:
    """Persist ALL scans into the BNDB under SCANS_KEY (non-auto metadata)."""
    if bv is None:
        return False
    payload = json.dumps([s.to_dict() for s in scans])
    try:
        bv.store_metadata(SCANS_KEY, payload, False)
    except TypeError:
        try:
            bv.store_metadata(SCANS_KEY, payload)
        except Exception:
            return False
    except Exception:
        return False
    _flush_to_disk(bv)
    return True


def load_scans(bv) -> List["ScanRecord"]:
    """Load all saved scans. Falls back to wrapping a legacy single-findings
    blob as one scan so results from before multi-scan support still appear."""
    if bv is None:
        return []
    raw = None
    try:
        raw = bv.query_metadata(SCANS_KEY)
    except Exception:
        raw = None
    if raw:
        try:
            return [ScanRecord.from_dict(d) for d in json.loads(raw)]
        except Exception:
            return []
    # Legacy fallback: a single findings list from an older version.
    legacy = load(bv)
    if legacy:
        return [ScanRecord(name="Imported scan", mode="online", findings=legacy)]
    return []


# Export ---------------------------------------------------------------------

def to_json(findings: List[Finding]) -> str:
    return json.dumps([f.to_dict() for f in findings], indent=2)


def _reconstruct_user_prompt(rule, output_schema: str, finding: Finding) -> str:
    """Rebuild (approximately) the user prompt that produced this finding, from
    the stored rule + context excerpt — used to build fine-tuning examples."""
    rule_prompt = ""
    if rule is not None:
        rule_prompt = rule.prompt.replace("{function}", finding.function_name or "the function")
    return (f"{rule_prompt}\n\nOutput format: {output_schema}\n\n"
            f"----- BEGIN BINARY CONTEXT -----\n{finding.context_excerpt}\n"
            "----- END BINARY CONTEXT -----")


def _corrected_verdict_json(finding: Finding) -> str:
    """The human-corrected verdict for a triaged finding, as the assistant target
    of a fine-tuning example. Reason becomes the explanation."""
    verdict = {
        "scratchpad": finding.scratchpad or "",
        "is_vulnerable": finding.triage_status == TRIAGE_REAL,
        "confidence": "high",  # human-confirmed
        "severity": finding.severity,
        "cwe": finding.cwe,
        "title": finding.title,
        "explanation": finding.triage_reason or finding.explanation,
        "tainted_input": finding.tainted_input,
        "recommendation": finding.recommendation,
    }
    return json.dumps(verdict, ensure_ascii=False)


def to_finetuning_jsonl(findings: List[Finding], scan_record=None) -> str:
    """Export ONLY triaged findings (Real Issue / False Positive) as OpenAI-style
    chat fine-tuning JSONL. Each line is {"messages":[system,user,assistant]} where
    the assistant message is the human-corrected verdict — so a False Positive
    teaches is_vulnerable=false (with the analyst's reason) and a Real Issue
    reinforces is_vulnerable=true. Reconstructs the prompts from the rule files used
    by the scan."""
    # Resolve system prompt + rules per phase (using the scan's rule paths).
    phase_info = {}
    try:
        from . import rules as _rules
        loaders = {1: _rules.load_phase1_rules, 2: _rules.load_phase2_rules,
                   3: _rules.load_phase3_rules}
        paths = {
            1: getattr(scan_record, "rules_phase1_path", "") if scan_record else "",
            2: getattr(scan_record, "rules_phase2_path", "") if scan_record else "",
            3: getattr(scan_record, "rules_phase3_path", "") if scan_record else "",
        }
        for ph, loader in loaders.items():
            try:
                rs = loader(paths.get(ph, ""))
                phase_info[ph] = (rs.system_prompt, rs.output_schema,
                                  {r.id: r for r in rs.rules})
            except Exception:
                phase_info[ph] = None
    except Exception:
        phase_info = {}

    generic_system = ("You are a binary security analyst. Decide whether the "
                      "candidate is a genuine security issue and respond with a "
                      "single JSON verdict.")
    lines = []
    for f in findings:
        if f.triage_status not in (TRIAGE_REAL, TRIAGE_FALSE):
            continue
        info = phase_info.get(f.phase)
        if info:
            system_prompt, schema, rule_map = info
            user = _reconstruct_user_prompt(rule_map.get(f.rule_id), schema, f)
        else:
            system_prompt = generic_system
            user = (f"Rule: {f.rule_id}\n\n----- BINARY CONTEXT -----\n"
                    f"{f.context_excerpt}")
        example = {"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": _corrected_verdict_json(f)},
        ]}
        lines.append(json.dumps(example, ensure_ascii=False))
    return ("\n".join(lines) + "\n") if lines else ""


def to_markdown(findings: List[Finding], binary_name: str = "") -> str:
    lines = [f"# VulnFanatic-NG findings{f' — {binary_name}' if binary_name else ''}", ""]
    lines.append(f"Total confirmed issues: {len(findings)}")
    lines.append("")
    for f in sort_findings(findings):
        lines.append(f"## [confidence: {f.confidence.upper()}] {f.title or f.rule_id}")
        meta = [
            f"- **Status:** {f.triage_label}" + (f" — {f.triage_reason}" if f.triage_reason else ""),
            f"- **Rule:** {f.rule_id} (phase {f.phase})",
            f"- **CWE:** {f.cwe or 'n/a'}",
            f"- **Confidence:** {f.confidence}",
            f"- **Severity (model estimate):** {f.severity}",
            f"- **Function:** {f.function_name} @ {f.function_addr:#x}",
            f"- **Location:** {f.address:#x}",
        ]
        if f.representation:
            meta.append(f"- **Context source:** {f.representation}")
        if f.prompt_file:
            meta.append(f"- **Prompt file (test mode):** `{f.prompt_file}`")
        lines.extend(meta)
        lines.append("")
        if f.explanation:
            lines.append(f.explanation)
            lines.append("")
        if f.tainted_input:
            lines.append(f"**Tainted input:** {f.tainted_input}")
            lines.append("")
        if f.recommendation:
            lines.append(f"**Recommendation:** {f.recommendation}")
            lines.append("")
        if f.scratchpad:
            lines.append("**Analysis scratchpad (claimed flow + snippets):**")
            lines.append("")
            lines.append("```")
            lines.append(f.scratchpad)
            lines.append("```")
            lines.append("")
        if f.validation_notes:
            lines.append("**Validation (adversarial second pass):**")
            lines.append("")
            lines.append("```")
            lines.append(f.validation_notes)
            lines.append("```")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
