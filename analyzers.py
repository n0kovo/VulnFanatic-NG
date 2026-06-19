"""Analyzers turn a (system_prompt, user_prompt) pair into a verdict.

Two implementations share one interface so the scan phases don't branch on mode:

* ``LLMAnalyzer`` delegates to the OpenAI-compatible client.
* ``TestModeAnalyzer`` (enabled when the API base URL is ``TEST``) never calls a
  model: it writes the full prompt and context to a file under /tmp and returns a
  synthetic "flagged" verdict that points at that file. This lets you review and
  validate the exact prompts/context VulnFanatic-NG would send.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from typing import Dict, Optional

from . import debuglog as _dbg
from . import llm as _llm

# Verdict key carrying the path of the saved prompt (test mode only).
PROMPT_FILE_KEY = "prompt_file"

# Adversarial second-pass validator. Generic across rules; it re-checks the first
# analyst's claim against the SAME context and tries to disprove it.
VALIDATION_SYSTEM_PROMPT = (
    "You are a principal security reviewer double-checking another analyst's "
    "vulnerability assessment of decompiled (HLIL) code. Your goal is ACCURACY, not "
    "rejection: correct mistakes in EITHER direction — confirm genuine issues, raise "
    "or lower the confidence as warranted, and overturn a verdict only when the "
    "evidence clearly contradicts it. You are given the same binary context the "
    "first analyst saw, their verdict, and their scratchpad. Proceed: (1) Check the "
    "code snippets quoted in the scratchpad against the provided context; if a "
    "snippet is clearly fabricated or materially misquoted, that undermines the "
    "claim — note it. (Do not reject a sound finding over a minor paraphrase.) "
    "(2) Re-trace the FULL execution path from the input source through the "
    "call-path and data-flow functions to the sink, and validate EVERY constraint "
    "on that path (each if/loop/switch condition, length/size/bounds check, "
    "validation, early return) plus buffer sizes/types and integer width/sign. If a "
    "guard that is present in the context — e.g. a length or bounds check such as "
    "'if (len < sizeof(buf))', input validation, or a return-on-error — prevents the "
    "vulnerable value on every path that reaches the sink, the issue is not "
    "exploitable and you MUST set is_vulnerable=false (unless a different, unguarded "
    "path to the sink is shown). (3) Also consider whether the first "
    "analyst MISSED something — a real issue they under-rated, or extra reachable "
    "paths. (4) Decide the FINAL verdict. Report a plausible, security-relevant "
    "issue as is_vulnerable=true and use confidence to reflect how complete the "
    "evidence is (high = chain shown; medium = likely with some inference; low = a "
    "lead worth manual review — still report it). Reserve is_vulnerable=false for "
    "cases that are clearly safe or clearly unsupported. Record your reasoning in "
    "the scratchpad and END it with an explicit 'Confidence justification:' line "
    "stating the final confidence (high/medium/low) and why, naming which links are "
    "proven by snippets versus inferred; the confidence field MUST match it. "
    "Summarize in the validation field what you confirmed, corrected, or added and "
    "why. Output ONLY a single minified JSON object using the schema given in the "
    "user message, with no markdown, no code fences, and no text before or after it."
)


# Appended to the system prompt to control how much written reasoning the model
# emits. Most per-call latency is the reasoning text, so trimming it is the main
# speed lever. "full" appends nothing (the rules' default detailed scratchpad).
_REASONING_DIRECTIVES = {
    "concise": (
        "\n\nRESPONSE BREVITY (this OVERRIDES any instruction above to quote code "
        "verbatim or show extensive work): reason internally, then keep the output "
        "COMPACT to minimize latency. The \"scratchpad\" must be AT MOST 1-3 short "
        "sentences naming the single most decisive line or condition by reference "
        "(function + address) WITHOUT pasting code, ending with a brief 'Confidence "
        "justification:' clause. Keep \"explanation\" to one or two sentences and do "
        "not restate the context."
    ),
    "none": (
        "\n\nRESPONSE BREVITY (this OVERRIDES any instruction above to produce a "
        "scratchpad): reason internally and emit NO written reasoning. Set "
        "\"scratchpad\" to an empty string and keep \"explanation\" to a single short "
        "sentence. Output only the verdict fields. This minimizes latency."
    ),
}


def reasoning_directive(style: str) -> str:
    """Directive text appended to the system prompt for a reasoning style ('' = none)."""
    return _REASONING_DIRECTIVES.get((style or "").strip().lower(), "")


# The output schema (in the USER prompt) lists "scratchpad" as the FIRST field with a
# long spec, so the model generates it before the verdict — the dominant driver of
# response length. Rewrite that field to match the reasoning style.
_SCRATCHPAD_FIELD_RE = re.compile(r'"scratchpad"\s*:\s*"(?:[^"\\]|\\.)*"\s*,?\s*')
_CONCISE_SCRATCHPAD = (
    '"scratchpad": "<at most 1-3 short sentences naming the single most decisive '
    "line/condition by function+address, NO verbatim code; end with 'Confidence "
    'justification: <high|medium|low> because ...\'>", '
)


def reasoning_schema(schema: str, style: str) -> str:
    """Rewrite the output schema for the reasoning style: 'none' drops the scratchpad
    field, 'concise' shrinks it, 'full' (or unknown) leaves it unchanged."""
    style = (style or "").strip().lower()
    if not schema or style not in ("none", "concise"):
        return schema
    repl = "" if style == "none" else _CONCISE_SCRATCHPAD
    return _SCRATCHPAD_FIELD_RE.sub(repl, schema, count=1)


def apply_reasoning(system_prompt: str, style: str) -> str:
    """Return ``system_prompt`` with the reasoning-style directive appended."""
    directive = reasoning_directive(style)
    return system_prompt + directive if directive else system_prompt


def _build_validation_user(original_user: str, first_verdict: Dict, output_schema: str) -> str:
    first = {
        k: first_verdict.get(k)
        for k in ("is_vulnerable", "confidence", "severity", "cwe", "title",
                  "explanation", "tainted_input", "recommendation")
    }
    return (
        original_user
        + "\n\n----- FIRST ANALYST VERDICT (validate and dispute this) -----\n"
        + json.dumps(first)
        + "\n\n----- FIRST ANALYST SCRATCHPAD (their claimed flow + snippets) -----\n"
        + (first_verdict.get("scratchpad") or "(no scratchpad was provided)")
        + "\n\nDouble-check this assessment against the BINARY CONTEXT shown above. "
        "Verify the quoted snippets, re-trace the flow yourself, correct the verdict "
        "or confidence in either direction if warranted, and then output the FINAL "
        "verdict. Output format: " + (output_schema or "")
        + " Also include a \"validation\" field summarizing what you confirmed, "
        "corrected, or added and why."
    )


def _sanitize(name: str, maxlen: int = 48) -> str:
    name = name or "unknown"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return (name or "unknown")[:maxlen]


def test_output_dir(bv=None) -> str:
    """Per-scan directory under /tmp/vulnfanatic_ng (falls back if /tmp is absent)."""
    root = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()
    base = os.path.join(root, "vulnfanatic_ng")
    binname = "binary"
    if bv is not None:
        try:
            raw = bv.file.original_filename or bv.file.filename or "binary"
            binname = _sanitize(os.path.basename(raw))
        except Exception:
            pass
    return os.path.join(base, f"{binname}-{time.strftime('%Y%m%d-%H%M%S')}")


class LLMAnalyzer:
    """Real analyzer: first-pass analysis + optional adversarial validation.

    The first pass produces a verdict whose ``scratchpad`` holds the reasoning and
    the verbatim snippets the model relied on. If it flags an issue and validation
    is enabled, a second pass critically re-checks that claim against the same
    context; its verdict (and confidence) become the final result, while the first
    pass's scratchpad is preserved so the dispute can be reviewed.
    """

    def __init__(self, client, validation_pass: bool = True, reasoning: str = "full",
                 validator_client=None):
        self.client = client
        # Optional SEPARATE client for the validation pass (a second opinion from an
        # independent model). When None, the validation pass reuses ``client``.
        self.validator_client = validator_client
        self.validation_pass = validation_pass
        self.reasoning = reasoning

    def analyze(self, system_prompt: str, user_prompt: str, meta: Optional[Dict] = None) -> Dict:
        _dbg.log(f"  analyzer: first pass (reasoning={self.reasoning}, "
                 f"validation={'on' if self.validation_pass else 'off'})")
        v1 = self.client.analyze(apply_reasoning(system_prompt, self.reasoning), user_prompt)
        if not self.validation_pass or not v1.get("is_vulnerable"):
            v1.setdefault("validation", "")
            return v1

        validator = self.validator_client or self.client
        _dbg.log("  analyzer: running validation (second) pass"
                 + (" on the separate validator model" if self.validator_client else ""))
        schema = (meta or {}).get("output_schema", "")
        val_user = _build_validation_user(user_prompt, v1, schema)
        try:
            v2 = validator.analyze(
                apply_reasoning(VALIDATION_SYSTEM_PROMPT, self.reasoning), val_user)
        except _llm.LLMError as exc:
            # The validation pass failed (unparseable response, or — more likely with a
            # separate validator endpoint — a connection error). Keep the first verdict
            # rather than losing the finding, and record why validation didn't run.
            v1["validation"] = f"(validation pass did not complete: {exc})"
            return v1

        final = dict(v2)
        # Preserve the first analyst's claimed flow + snippets for review.
        final["scratchpad"] = v1.get("scratchpad", "")
        validator_notes = v2.get("validation", "") or ""
        if v2.get("scratchpad"):
            validator_notes = (validator_notes + "\n\n[Validator independent trace]\n"
                               + v2["scratchpad"]).strip()
        final["validation"] = validator_notes
        # Fall back to first-pass metadata where the validator left fields empty.
        for k in ("severity", "cwe", "title", "explanation", "tainted_input", "recommendation"):
            if not final.get(k):
                final[k] = v1.get(k, "")
        return final


class TestModeAnalyzer:
    """Test analyzer: dump prompt+context to a file, flag everything."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self._used = set()

    def _filename(self, meta: Dict) -> str:
        func = _sanitize(meta.get("function_name") or "func")
        addr = meta.get("address", 0) or 0
        name = f"p{meta.get('phase', 0)}_{_sanitize(meta.get('rule_id') or 'rule')}_{addr:#x}_{func}.txt"
        candidate = name
        n = 1
        while candidate in self._used:
            candidate = f"{name[:-4]}_{n}.txt"
            n += 1
        self._used.add(candidate)
        return candidate

    def analyze(self, system_prompt: str, user_prompt: str, meta: Optional[Dict] = None) -> Dict:
        meta = meta or {}
        path = os.path.join(self.out_dir, self._filename(meta))
        header = (
            "# VulnFanatic-NG TEST MODE prompt dump (the LLM was NOT called)\n"
            f"# phase:    {meta.get('phase')}\n"
            f"# rule:     {meta.get('rule_id')} ({meta.get('rule_name')})\n"
            f"# severity: {meta.get('severity')}   cwe: {meta.get('cwe')}\n"
            f"# function: {meta.get('function_name')} @ {meta.get('function_addr', 0):#x}\n"
            f"# location: {meta.get('address', 0):#x}\n\n"
        )
        body = (
            "===== ANALYSIS SYSTEM PROMPT (first pass) =====\n"
            f"{system_prompt}\n\n"
            "===== ANALYSIS USER PROMPT (includes the full generated context) =====\n"
            f"{user_prompt}\n\n"
            "===== VALIDATION SYSTEM PROMPT (second, adversarial pass) =====\n"
            f"{VALIDATION_SYSTEM_PROMPT}\n\n"
            "# NOTE: the validation USER prompt is generated at runtime from the "
            "first pass's response (its verdict + scratchpad) and is not shown here.\n"
        )
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(header + body)
        except Exception as exc:  # keep the scan going even if a write fails
            path = f"(failed to write prompt file: {exc})"

        return {
            "is_vulnerable": True,
            "confidence": "low",
            "severity": meta.get("severity") or "info",
            "cwe": meta.get("cwe", ""),
            "title": f"[TEST] {meta.get('rule_name') or meta.get('rule_id') or 'candidate'}",
            "explanation": (
                "TEST MODE: the LLM was not called. This candidate was flagged "
                "automatically and the full prompt + context were written to the "
                "prompt file for manual review and validation."
            ),
            "tainted_input": "",
            "recommendation": "",
            "scratchpad": "TEST MODE: not analyzed; see the prompt file.",
            "validation": "TEST MODE: validation pass not run.",
            PROMPT_FILE_KEY: path,
        }
