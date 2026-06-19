"""Phase 2: locate security-sensitive functions and audit them with the LLM.

Only runs when the binary appears to have real symbols / variable names, because
the audit relies on meaningful identifiers. The gate is a heuristic (ratio of
non-auto-named functions, plus debug-info presence) and can be overridden with
the ``vulnfanatic.phase2ForceEnable`` setting.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from . import analyzers as _analyzers
from . import context_builder as ctx
from . import debuglog as _dbg
from . import llm as _llm
from .findings import (
    Finding, clip_context, CONFIDENCE_UNKNOWN,
    TRIAGE_ERROR, TRIAGE_NON_ISSUE, TRIAGE_REJECTED, TRIAGE_UNTRIAGED,
)
from .rules import Phase2Rule, RuleSet

try:
    import binaryninja  # noqa: F401
except Exception:  # pragma: no cover
    binaryninja = None  # type: ignore

_AUTO_NAME_PREFIXES = ("sub_", "j_", "nullsub_", "_sub_", "unknown_", "data_")
SYMBOL_RATIO_THRESHOLD = 0.30


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def has_symbols(bv) -> bool:
    """Heuristic: does this binary carry meaningful symbol / variable names?"""
    funcs = _safe(lambda: list(bv.functions)) or []
    if not funcs:
        return False
    named = 0
    for f in funcs:
        name = _safe(lambda f=f: f.name) or ""
        if not name.startswith(_AUTO_NAME_PREFIXES):
            named += 1
    ratio = named / len(funcs)
    if ratio >= SYMBOL_RATIO_THRESHOLD:
        return True
    # Debug info (DWARF/PDB) is a strong positive signal even at low ratios.
    if _safe(lambda: bv.debug_info) is not None and _safe(lambda: bool(bv.debug_info)):
        return True
    return False


def collect_targets(bv, ruleset: RuleSet) -> List[Tuple[Phase2Rule, object, List[str]]]:
    """Return (rule, func, referenced_strings) for each security-sensitive match."""
    string_index = ctx.build_string_index(bv)
    out: List[Tuple[Phase2Rule, object, List[str]]] = []
    seen = set()
    funcs = _safe(lambda: list(bv.functions)) or []
    for func in funcs:
        # Skip thunks / library-call wrappers — linker artifacts, not real code.
        if _safe(lambda f=func: bool(f.is_thunk)):
            continue
        names = ctx.function_names(func)
        fstart = _safe(lambda f=func: f.start)
        strings = string_index.get(fstart, []) if fstart is not None else []
        for rule in ruleset.rules:
            if rule.matches(names, strings):
                key = (fstart, rule.id)
                if key in seen:
                    continue
                seen.add(key)
                out.append((rule, func, strings))
    return out


def _build_user_prompt(rule: Phase2Rule, func_name: str, context_text: str,
                       output_schema: str) -> str:
    rule_prompt = rule.prompt.replace("{function}", func_name or "the function")
    return (
        f"{rule_prompt}\n\n"
        f"Output format: {output_schema}\n\n"
        f"----- BEGIN BINARY CONTEXT -----\n{context_text}\n----- END BINARY CONTEXT -----"
    )


def run_phase2(
    bv,
    ruleset: RuleSet,
    cfg,
    analyzer,
    *,
    on_status: Callable[[str], None] = lambda s: None,
    on_finding: Callable[[Finding], None] = lambda f: None,
    on_progress: Callable[[int, int], None] = lambda c, t: None,
    on_site_error: Callable[[str], None] = lambda s: None,
    is_cancelled: Callable[[], bool] = lambda: False,
    phase_number: int = 2,
) -> List[Finding]:
    """Audit security-sensitive functions with the LLM, emitting findings.

    Shared by Phase 2 (software security-design weaknesses) and Phase 3 (hardware
    attack hardening controls); ``phase_number`` distinguishes them in findings and
    status messages. Both are symbol-gated by the phase-2 symbol settings.
    """
    # On a stripped binary, function/variable names are unreliable but referenced
    # string constants still are — so instead of skipping the phase entirely we run
    # only the rules that match on string evidence (those are symbol-independent).
    string_only = (cfg.phase2_require_symbols and not cfg.phase2_force_enable
                   and not has_symbols(bv))
    if string_only:
        on_status(
            f"Phase {phase_number}: binary appears stripped (no reliable symbols); "
            "running only string-evidence matches. Enable "
            "'vulnfanatic.phase2ForceEnable' to also audit name-based matches."
        )
        _dbg.log(f"phase{phase_number}: stripped binary -> string-evidence rules only")
    else:
        on_status(f"Phase {phase_number}: locating security-sensitive functions...")
    _dbg.log(f"phase{phase_number}: collecting security-sensitive functions")
    targets = collect_targets(bv, ruleset)
    if string_only:
        # Keep only matches that hold on string evidence alone (no name reliance).
        targets = [(rule, func, strings) for (rule, func, strings) in targets
                   if _safe(lambda r=rule, s=strings: r.matches([], s))]
    total = len(targets)
    _dbg.log(f"phase{phase_number}: {total} function match(es)")
    on_status(f"Phase {phase_number}: {total} function match(es); analyzing...")

    schema = _analyzers.reasoning_schema(ruleset.output_schema, cfg.verdict_reasoning)
    # Debug mode: keep every audited function in the results — LLM-rejected ones get
    # the REJECTED status, and functions that errored get the ERROR status, instead of
    # being dropped (see run_phase1).
    debug_all = bool(getattr(cfg, "debug_logging", False))
    findings: List[Finding] = []

    def emit_error_row(rule, func_name, func_start, reason, bundle=None):
        if not debug_all:
            return
        findings.append(Finding(
            phase=phase_number, rule_id=rule.id, severity=rule.severity,
            confidence="low", title=f"{rule.name} ({func_name})",
            function_name=func_name, function_addr=func_start, address=func_start,
            cwe=rule.cwe, explanation=reason,
            representation=(bundle.representation if bundle is not None else ""),
            call_paths=(bundle.call_paths_text if bundle is not None else ""),
            context_excerpt=(clip_context(bundle.text, getattr(cfg, "stored_context_chars", 0))
                             if bundle is not None else ""),
            triage_status=TRIAGE_ERROR, model=cfg.model,
        ))
        on_finding(findings[-1])
        _dbg.log("  -> recorded ERROR row for debug")

    def emit_unscored(rule, func_name, func_start, raw, bundle=None):
        """UNKNOWN-confidence 'Unscored' finding when the model could not return a
        parseable verdict (recall-preserving; keeps the partial output)."""
        text = (raw or "").strip()
        findings.append(Finding(
            phase=phase_number, rule_id=rule.id, severity=rule.severity,
            confidence=CONFIDENCE_UNKNOWN, title=f"Unscored ({func_name}): {rule.name}",
            function_name=func_name, function_addr=func_start, address=func_start,
            cwe=rule.cwe,
            explanation=("The model did not return a parseable JSON verdict (prose, "
                         "truncated reasoning, or empty message), so the result is "
                         "UNKNOWN — review this function manually. Partial model output:"
                         "\n\n" + (text[:4000] if text else "(empty response)")),
            representation=(bundle.representation if bundle is not None else ""),
            call_paths=(bundle.call_paths_text if bundle is not None else ""),
            context_excerpt=(clip_context(bundle.text, getattr(cfg, "stored_context_chars", 0))
                             if bundle is not None else ""),
            scratchpad=text, triage_status=TRIAGE_UNTRIAGED, model=cfg.model,
        ))
        on_finding(findings[-1])
        _dbg.log("  -> reported as UNKNOWN-confidence Unscored lead (unparseable)")

    for idx, (rule, func, strings) in enumerate(targets):
        if is_cancelled():
            on_status(f"Phase {phase_number}: cancelled.")
            break
        on_progress(idx + 1, total)  # 1-based: report the item being worked on

        func_name = _safe(lambda f=func: f.name) or "?"
        func_start = _safe(lambda f=func: f.start) or 0
        matched_apis = ctx.security_apis_in(func, rule.name_keywords)
        _dbg.log(f"phase{phase_number} target {idx + 1}/{total}: rule={rule.id} "
                 f"fn={_dbg.sym(func_name)} @ {_dbg.addr(func_start)}")

        try:
            bundle = ctx.build_function_context(
                bv, func, cfg, referenced_strings=strings, matched_apis=matched_apis
            )
        except Exception as exc:
            on_site_error(f"context build failed for {func_name}@{func_start:#x}: {exc}")
            _dbg.log(f"  -> context build FAILED: {exc}")
            emit_error_row(rule, func_name, func_start,
                           f"Context build failed (not analyzed by the LLM): {exc}")
            continue
        _dbg.log(f"  context built: {_dbg.size(bundle.text)}; sending to analyzer")

        user_prompt = _build_user_prompt(rule, func_name, bundle.text, schema)
        meta = {
            "phase": phase_number,
            "rule_id": rule.id,
            "rule_name": rule.name,
            "severity": rule.severity,
            "cwe": rule.cwe,
            "function_name": func_name,
            "function_addr": func_start,
            "address": func_start,
            "output_schema": schema,
        }

        try:
            verdict = analyzer.analyze(ruleset.system_prompt, user_prompt, meta)
        except _llm.LLMConnectionError as exc:
            on_site_error(f"LLM connection error at {func_start:#x}: {exc}")
            emit_error_row(rule, func_name, func_start,
                           f"LLM connection error (not analyzed): {exc}", bundle)
            continue
        except _llm.LLMResponseError as exc:
            on_site_error(f"LLM response error at {func_start:#x}: {exc}")
            if getattr(cfg, "flag_unparseable", True):
                emit_unscored(rule, func_name, func_start, getattr(exc, "raw", ""), bundle)
            else:
                emit_error_row(rule, func_name, func_start,
                               f"LLM response could not be parsed (not analyzed): {exc}",
                               bundle)
            continue
        except Exception as exc:  # pragma: no cover - defensive
            on_site_error(f"LLM error at {func_start:#x}: {exc}")
            _dbg.log(f"  -> LLM error: {exc}")
            emit_error_row(rule, func_name, func_start,
                           f"LLM error (not analyzed): {exc}", bundle)
            continue

        _dbg.log(f"  verdict: vulnerable={verdict.get('is_vulnerable')} "
                 f"confidence={verdict.get('confidence')} title={_dbg.field(verdict.get('title'))}")
        is_vuln = bool(verdict.get("is_vulnerable"))
        triage = TRIAGE_UNTRIAGED
        if not is_vuln:
            if not debug_all:
                continue
            triage = TRIAGE_REJECTED
            _dbg.log("  -> LLM rejected (not an issue); recorded as REJECTED for debug")
        elif not _llm.meets_min_confidence(verdict.get("confidence", "low"), cfg.min_confidence):
            if not debug_all:
                _dbg.log(f"  -> dropped: confidence below minimum ({cfg.min_confidence})")
                continue
            _dbg.log(f"  -> below min confidence ({cfg.min_confidence}); kept for debug")

        finding = Finding(
            phase=phase_number,
            rule_id=rule.id,
            severity=verdict.get("severity") or rule.severity,
            confidence=verdict.get("confidence", "low"),
            title=verdict.get("title") or rule.name,
            function_name=func_name,
            function_addr=func_start,
            address=func_start,
            cwe=verdict.get("cwe") or rule.cwe,
            explanation=verdict.get("explanation", ""),
            tainted_input=verdict.get("tainted_input", ""),
            recommendation=verdict.get("recommendation", ""),
            representation=bundle.representation,
            call_paths=bundle.call_paths_text,
            context_excerpt=clip_context(bundle.text, getattr(cfg, "stored_context_chars", 0)),
            scratchpad=verdict.get("scratchpad", ""),
            validation_notes=verdict.get("validation", ""),
            prompt_file=verdict.get("prompt_file", ""),
            triage_status=triage,
            model=cfg.model,
        )
        findings.append(finding)
        _dbg.log(f"  -> {'RECORDED (REJECTED)' if triage == TRIAGE_REJECTED else 'REPORTED'} "
                 f"finding: {_dbg.field(finding.title)} "
                 f"[{finding.severity}/{finding.confidence}] {finding.cwe}")
        on_finding(finding)

    on_progress(total, total)
    issues = sum(1 for f in findings if f.triage_status not in TRIAGE_NON_ISSUE)
    debug_extra = len(findings) - issues
    rej_note = f"; {debug_extra} non-issue row(s) shown (debug)" if debug_extra else ""
    on_status(f"Phase {phase_number} complete: {issues} finding(s) "
              f"from {total} function(s){rej_note}.")
    return findings
