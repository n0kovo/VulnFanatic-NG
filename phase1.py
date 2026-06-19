"""Phase 1: find calls to dangerous functions and ask the LLM to judge each one."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from . import analyzers as _analyzers
from . import context_builder as ctx
from . import debuglog as _dbg
from . import llm as _llm
from . import prototypes
from .findings import (
    Finding, clip_context, CONFIDENCE_UNKNOWN,
    TRIAGE_ERROR, TRIAGE_NON_ISSUE, TRIAGE_REJECTED, TRIAGE_SKIPPED, TRIAGE_UNTRIAGED,
)
from .rules import Phase1Rule, RuleSet

try:
    import binaryninja
    from binaryninja import SymbolType
except Exception:  # pragma: no cover
    binaryninja = None  # type: ignore
    SymbolType = None  # type: ignore

# Symbol types that can be a call target.
_TARGET_SYMBOL_TYPE_NAMES = (
    "FunctionSymbol",
    "ImportedFunctionSymbol",
    "ImportAddressSymbol",
    "LibraryFunctionSymbol",
    "ExternalSymbol",
)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _debug_row(phase, rule, matched_name, caller, call_addr, status, reason, cfg,
               bundle=None) -> Finding:
    """A debug-only bookkeeping Finding (SKIPPED/ERROR) for a candidate that never
    produced a real verdict, so a debug scan can show one row per analyzed site."""
    name = _safe(lambda: caller.name) or "?"
    return Finding(
        phase=phase,
        rule_id=rule.id,
        severity=rule.severity,
        confidence="low",
        title=f"{rule.name} ({matched_name})",
        function_name=name,
        function_addr=_safe(lambda: caller.start) or 0,
        address=call_addr,
        cwe=rule.cwe,
        explanation=reason,
        representation=(bundle.representation if bundle is not None else ""),
        call_paths=(bundle.call_paths_text if bundle is not None else ""),
        context_excerpt=(clip_context(bundle.text, getattr(cfg, "stored_context_chars", 0))
                         if bundle is not None else ""),
        triage_status=status,
        model=cfg.model,
    )


def _unscored_finding(phase, rule, matched_name, caller, call_addr, raw, cfg,
                      bundle=None) -> Finding:
    """An UNKNOWN-confidence 'Unscored' Finding for a candidate the model could not
    score (unparseable / truncated / prose reply). Keeps the model's partial output so
    the site is reported for manual review instead of being dropped (recall-preserving).
    Confidence is 'unknown' (distinct from a low-confidence verdict) because the model
    never actually produced a verdict."""
    name = _safe(lambda: caller.name) or "?"
    text = (raw or "").strip()
    explanation = (
        "The model did not return a parseable JSON verdict (it may have replied in "
        "prose, run out of output budget while reasoning, or returned an empty "
        "message), so the result is UNKNOWN — review this site manually. Partial model "
        "output:\n\n" + (text[:4000] if text else "(empty response)")
    )
    return Finding(
        phase=phase,
        rule_id=rule.id,
        severity=rule.severity,
        confidence=CONFIDENCE_UNKNOWN,
        title=f"Unscored ({matched_name}): {rule.name}",
        function_name=name,
        function_addr=_safe(lambda: caller.start) or 0,
        address=call_addr,
        cwe=rule.cwe,
        explanation=explanation,
        representation=(bundle.representation if bundle is not None else ""),
        call_paths=(bundle.call_paths_text if bundle is not None else ""),
        context_excerpt=(clip_context(bundle.text, getattr(cfg, "stored_context_chars", 0))
                         if bundle is not None else ""),
        scratchpad=text,
        triage_status=TRIAGE_UNTRIAGED,
        model=cfg.model,
    )


def _target_symbol_types():
    types = []
    if SymbolType is None:
        return types
    for name in _TARGET_SYMBOL_TYPE_NAMES:
        t = getattr(SymbolType, name, None)
        if t is not None:
            types.append(t)
    return types


def _symbol_names(sym) -> List[str]:
    names = []
    for attr in ("short_name", "name", "full_name", "raw_name"):
        v = _safe(lambda a=attr: getattr(sym, a))
        if v and v not in names:
            names.append(v)
    return names


def _iter_target_symbols(bv):
    seen_keys = set()
    for st in _target_symbol_types():
        syms = _safe(lambda st=st: bv.get_symbols_of_type(st)) or []
        for sym in syms:
            addr = _safe(lambda s=sym: s.address)
            key = (addr, _safe(lambda s=sym: s.name))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            yield sym


def is_wrapper_or_thunk(func, matched_name: str = "") -> bool:
    """True if ``func`` is a thunk / thin forwarding wrapper for a library call —
    a compiler/linker artifact, not real code. These get decompiled to look like
    e.g. 'int printf(char* fmt, ...) { return printf(fmt); }' (the import thunk is
    named after the import and just relays to it), and must not be flagged.
    """
    if func is None:
        return False
    # Binary Ninja marks PLT stubs / import thunks.
    if _safe(lambda: bool(func.is_thunk)):
        return True
    name = (_safe(lambda: func.name) or "").lstrip("_").lower()
    target = (matched_name or "").lstrip("_").lower()
    if not target or not name:
        return False
    # Self-named import thunk: a function named after the very library function it
    # calls (printf -> printf), or a conventional thunk name (j_printf/thunk_printf).
    return name in (target, f"j_{target}", f"thunk_{target}", f"__imp_{target}")


def _match_rule(ruleset: RuleSet, names: List[str]) -> Tuple[Optional[Phase1Rule], str]:
    for rule in ruleset.rules:
        if rule.matches_name(*names):
            # Pick the most representative matched name (first that is in the
            # rule's explicit function list, else the first name).
            for n in names:
                if n in getattr(rule, "_function_set", set()):
                    return rule, n
            return rule, names[0] if names else ""
    return None, ""


# How far to chase a chain of forwarding thunks back to the real callers, and how
# many HLIL instructions to scan per function during the indirect-call sweep.
_MAX_THUNK_HOPS = 4
_MAX_FUNC_INSTRS = 20000


def _walk_refs(bv, target_addr, rule, matched_name, seen, depth=0, visited=None):
    """Yield real (non-thunk) call sites that reach ``target_addr``.

    When a reference lands inside a thunk / self-named forwarding wrapper we do NOT
    flag the relay call itself (it is a linker artifact). Instead we recover the
    REAL callers of that thunk and recurse, so a dangerous call routed through a
    PLT/GOT stub is still analyzed — closing a recall gap where ``get_code_refs``
    on the import symbol resolves only to the stub, not the code that calls it.
    """
    if visited is None:
        visited = set()
    if target_addr in visited or depth > _MAX_THUNK_HOPS:
        return
    visited.add(target_addr)
    refs = _safe(lambda a=target_addr: list(bv.get_code_refs(a))) or []
    for ref in refs:
        caller = _safe(lambda r=ref: r.function)
        call_addr = _safe(lambda r=ref: r.address)
        if caller is None or call_addr is None:
            continue
        if is_wrapper_or_thunk(caller, matched_name):
            cstart = _safe(lambda c=caller: c.start)
            if cstart is not None:
                yield from _walk_refs(bv, cstart, rule, matched_name, seen,
                                      depth + 1, visited)
            continue
        caller_start = _safe(lambda c=caller: c.start)
        key = (caller_start, call_addr, rule.id)
        if key in seen:
            continue
        seen.add(key)
        yield rule, matched_name, caller, call_addr


def resolve_call_target_names(bv, call_expr) -> List[str]:
    """Candidate names of the function a (possibly indirect) HLIL call targets.

    Resolves the call destination through Binary Ninja's analysis — a direct call,
    or a function pointer / vtable slot whose value BN pinned to a concrete address
    — so dangerous calls dispatched indirectly are recognised even though they
    reference no import symbol directly. Returns [] when the target is unresolved.
    """
    out: List[str] = []
    fn = ctx._resolve_callee_function(bv, call_expr)
    if fn is not None:
        out.extend(ctx.function_names(fn))
    if not out:
        dest = _safe(lambda: call_expr.dest)
        cval = ctx.expr_constant_value(dest) if dest is not None else None
        if isinstance(cval, int):
            f2 = _safe(lambda: bv.get_function_at(cval))
            if f2 is not None:
                out.extend(ctx.function_names(f2))
            else:
                sym = _safe(lambda: bv.get_symbol_at(cval))
                if sym is not None:
                    out.extend(_symbol_names(sym))
    return [n for n in out if n]


def collect_indirect_call_sites(bv, ruleset: RuleSet, seen: set):
    """Yield dangerous call sites whose callee BN resolved through a pointer/vtable.

    Complements the symbol-driven discovery by catching calls dispatched INDIRECTLY
    (through a function pointer, vtable slot, or other non-constant destination) that
    Binary Ninja resolved to a dangerous function. Direct calls — whose destination
    is a constant/import address — are intentionally skipped here: they are already
    found comprehensively by the symbol-based pass via ``get_code_refs``, and
    re-resolving them would produce duplicate findings (the two passes can record the
    same call at slightly different addresses). Deduplicated against ``seen``.
    """
    funcs = _safe(lambda: list(bv.functions)) or []
    for func in funcs:
        if _safe(lambda f=func: bool(func.is_thunk)):
            continue
        hlil = _safe(lambda f=func: func.hlil)
        if hlil is None:
            continue
        instrs = _safe(lambda: list(hlil.instructions)) or []
        caller_start = _safe(lambda f=func: func.start)
        for instr in instrs[:_MAX_FUNC_INSTRS]:
            instr_addr = _safe(lambda i=instr: i.address)
            for ce in ctx._iter_call_exprs(instr):
                dest = _safe(lambda c=ce: c.dest)
                # Skip DIRECT calls (constant/import destination) — the symbol pass
                # already covers them. Only resolve genuinely indirect dispatch.
                if dest is None or ctx._op_name(dest) in ctx._CONST_OPS:
                    continue
                names = resolve_call_target_names(bv, ce)
                if not names:
                    continue
                rule, matched_name = _match_rule(ruleset, names)
                if rule is None:
                    continue
                if is_wrapper_or_thunk(func, matched_name):
                    continue
                # Prefer the call expression's own address so it aligns with the
                # symbol pass's get_code_refs address for dedup.
                call_addr = _safe(lambda c=ce: c.address) or instr_addr
                if call_addr is None:
                    continue
                key = (caller_start, call_addr, rule.id)
                if key in seen:
                    continue
                seen.add(key)
                yield rule, matched_name, func, call_addr


def collect_call_sites(bv, ruleset: RuleSet, cfg=None):
    """Yield (rule, matched_name, caller_func, call_addr) for each dangerous call.

    Combines three discovery strategies, all deduplicated on
    (caller.start, call_addr, rule.id): direct calls to named dangerous symbols,
    calls routed through forwarding thunks (recovered via ``_walk_refs``), and —
    unless disabled — indirect calls resolved through value-set analysis.
    """
    seen = set()
    for sym in _iter_target_symbols(bv):
        names = _symbol_names(sym)
        if not names:
            continue
        rule, matched_name = _match_rule(ruleset, names)
        if rule is None:
            continue
        addr = _safe(lambda s=sym: s.address)
        if addr is None:
            continue
        yield from _walk_refs(bv, addr, rule, matched_name, seen)

    scan_indirect = getattr(cfg, "scan_indirect_calls", True) if cfg is not None else True
    if scan_indirect:
        yield from collect_indirect_call_sites(bv, ruleset, seen)


def _build_user_prompt(rule: Phase1Rule, matched_name: str, context_text: str,
                       output_schema: str) -> str:
    rule_prompt = rule.prompt.replace("{function}", matched_name or "the function")
    return (
        f"{rule_prompt}\n\n"
        f"Output format: {output_schema}\n\n"
        f"----- BEGIN BINARY CONTEXT -----\n{context_text}\n----- END BINARY CONTEXT -----"
    )


def run_phase1(
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
) -> List[Finding]:
    """Run the Phase 1 scan, emitting findings incrementally. Returns all findings."""
    on_status("Phase 1: locating dangerous function calls...")
    _dbg.log("phase1: collecting dangerous call sites")
    sites = list(collect_call_sites(bv, ruleset, cfg))
    total = len(sites)
    _dbg.log(f"phase1: {total} dangerous call site(s) found")
    on_status(f"Phase 1: {total} dangerous call site(s) found; analyzing with LLM...")

    tainted = ruleset.tainted_source_set
    # Match the requested verdict-reasoning detail in the output schema (the schema
    # is what makes the model emit a long scratchpad first).
    schema = _analyzers.reasoning_schema(ruleset.output_schema, cfg.verdict_reasoning)
    # Debug mode: keep EVERY analyzed candidate in the results, including the ones
    # the LLM judged not to be an issue (shown with the REJECTED status) and the ones
    # below the confidence threshold — so the full analyzed set is visible for review.
    debug_all = bool(getattr(cfg, "debug_logging", False))
    findings: List[Finding] = []
    skipped_const = 0
    skipped_fmt = 0

    def emit_debug_row(rule, matched_name, caller, call_addr, status, reason, bundle=None):
        """In debug mode, record a SKIPPED/ERROR row so the candidate is visible."""
        if not debug_all:
            return
        f = _debug_row(1, rule, matched_name, caller, call_addr, status, reason, cfg, bundle)
        findings.append(f)
        on_finding(f)
        _dbg.log(f"  -> recorded {status.upper()} row for debug")

    for idx, (rule, matched_name, caller, call_addr) in enumerate(sites):
        if is_cancelled():
            on_status("Phase 1: cancelled.")
            break
        on_progress(idx + 1, total)  # 1-based: report the item being worked on
        _dbg.log(f"phase1 site {idx + 1}/{total}: rule={rule.id} call={matched_name} "
                 f"in {_dbg.sym(_safe(lambda c=caller: c.name))} @ {_dbg.addr(call_addr)}")

        # Cheap pre-filter: for overflow-class rules, a call whose arguments are
        # all compile-time constants cannot be attacker-controlled — skip it
        # without an LLM call. (Saves work and removes a class of false positives.)
        if cfg.skip_constant_arg_calls and getattr(rule, "prefilter_const_safe", False):
            instr0 = ctx.hlil_instr_at(caller, call_addr)
            if instr0 is not None and not ctx.call_arg_vars(instr0):
                skipped_const += 1
                _dbg.log("  -> skipped: all arguments are compile-time constants")
                emit_debug_row(rule, matched_name, caller, call_addr, TRIAGE_SKIPPED,
                               "Skipped before the LLM: all call arguments are "
                               "compile-time constants (cannot be attacker-controlled).")
                continue

        # Provably-safe skip (always on): for format-string rules, a compile-time
        # constant format argument CANNOT be an uncontrolled format string — only the
        # variadic value(s) are attacker-influenced, which is normal/safe. The format
        # arg index depends on the prototype (fortified __*_chk shift it). This matches
        # the offline scanner and is recall-safe (a real bug needs a NON-constant
        # format), so it does not depend on skip_constant_arg_calls. LLMs frequently
        # misread the %s value argument as the format string, so eliminate it here.
        if rule.offline.get("format_arg_lookup"):
            fidx = prototypes.format_arg_index(matched_name)
            if fidx is not None:
                instr0 = ctx.hlil_instr_at(caller, call_addr)
                if instr0 is not None and ctx.call_param_is_constant(instr0, fidx):
                    skipped_fmt += 1
                    _dbg.log(f"  -> skipped: format arg #{fidx} is a constant literal "
                             "(provably not an uncontrolled format string)")
                    emit_debug_row(rule, matched_name, caller, call_addr, TRIAGE_SKIPPED,
                                   f"Skipped before the LLM: the format argument (#{fidx}) "
                                   "is a constant string literal, so it cannot be an "
                                   "uncontrolled format string.")
                    continue

        caller_name = _safe(lambda c=caller: c.name) or "?"
        _dbg.log("  building interprocedural context")
        try:
            bundle = ctx.build_call_site_context(
                bv, caller, matched_name, call_addr, cfg, tainted_source_set=tainted
            )
        except Exception as exc:  # context building should never abort the scan
            on_site_error(f"context build failed for {matched_name}@{call_addr:#x}: {exc}")
            _dbg.log(f"  -> context build FAILED: {exc}")
            emit_debug_row(rule, matched_name, caller, call_addr, TRIAGE_ERROR,
                           f"Context build failed (not analyzed by the LLM): {exc}")
            continue
        _dbg.log(f"  context built: repr={bundle.representation} {_dbg.size(bundle.text)}; "
                 "sending to analyzer")

        user_prompt = _build_user_prompt(rule, matched_name, bundle.text, schema)
        meta = {
            "phase": 1,
            "rule_id": rule.id,
            "rule_name": rule.name,
            "severity": rule.severity,
            "cwe": rule.cwe,
            "function_name": caller_name,
            "function_addr": _safe(lambda c=caller: c.start) or 0,
            "address": call_addr,
            "output_schema": schema,
        }

        try:
            verdict = analyzer.analyze(ruleset.system_prompt, user_prompt, meta)
        except _llm.LLMConnectionError as exc:
            on_site_error(f"LLM connection error at {call_addr:#x}: {exc}")
            _dbg.log(f"  -> LLM connection error: {exc}")
            emit_debug_row(rule, matched_name, caller, call_addr, TRIAGE_ERROR,
                           f"LLM connection error (not analyzed): {exc}", bundle)
            continue
        except _llm.LLMResponseError as exc:
            on_site_error(f"LLM response error at {call_addr:#x}: {exc}")
            _dbg.log(f"  -> LLM response error: {exc}")
            if getattr(cfg, "flag_unparseable", True):
                f = _unscored_finding(1, rule, matched_name, caller, call_addr,
                                      getattr(exc, "raw", ""), cfg, bundle)
                findings.append(f)
                on_finding(f)
                _dbg.log("  -> reported as UNKNOWN-confidence Unscored lead (unparseable)")
            else:
                emit_debug_row(rule, matched_name, caller, call_addr, TRIAGE_ERROR,
                               f"LLM response could not be parsed (not analyzed): {exc}",
                               bundle)
            continue
        except Exception as exc:  # pragma: no cover - defensive
            on_site_error(f"LLM error at {call_addr:#x}: {exc}")
            _dbg.log(f"  -> LLM error: {exc}")
            emit_debug_row(rule, matched_name, caller, call_addr, TRIAGE_ERROR,
                           f"LLM error (not analyzed): {exc}", bundle)
            continue

        _dbg.log(f"  verdict: vulnerable={verdict.get('is_vulnerable')} "
                 f"confidence={verdict.get('confidence')} title={_dbg.field(verdict.get('title'))}")
        is_vuln = bool(verdict.get("is_vulnerable"))
        triage = TRIAGE_UNTRIAGED
        if not is_vuln:
            if not debug_all:
                continue
            triage = TRIAGE_REJECTED  # debug: keep it, flagged as rejected by the LLM
            _dbg.log("  -> LLM rejected (not an issue); recorded as REJECTED for debug")
        elif not _llm.meets_min_confidence(verdict.get("confidence", "low"), cfg.min_confidence):
            if not debug_all:
                _dbg.log(f"  -> dropped: confidence below minimum ({cfg.min_confidence})")
                continue
            _dbg.log(f"  -> below min confidence ({cfg.min_confidence}); kept for debug")

        finding = Finding(
            phase=1,
            rule_id=rule.id,
            severity=verdict.get("severity") or rule.severity,
            confidence=verdict.get("confidence", "low"),
            title=verdict.get("title") or rule.name,
            function_name=caller_name,
            function_addr=_safe(lambda c=caller: c.start) or 0,
            address=call_addr,
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
    skips = []
    if skipped_const:
        skips.append(f"{skipped_const} constant-arg")
    if skipped_fmt:
        skips.append(f"{skipped_fmt} constant-format")
    suffix = f" ({', '.join(skips)} site(s) skipped)" if skips else ""
    issues = sum(1 for f in findings if f.triage_status not in TRIAGE_NON_ISSUE)
    debug_extra = len(findings) - issues
    rej_note = f"; {debug_extra} non-issue row(s) shown (debug)" if debug_extra else ""
    on_status(f"Phase 1 complete: {issues} issue(s) from "
              f"{total} call site(s){rej_note}{suffix}.")
    return findings
