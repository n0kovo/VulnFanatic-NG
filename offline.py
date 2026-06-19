"""Offline (no-LLM) programmatic Phase 1.

Runs the same Phase 1 call-site discovery, then applies cheap, deterministic
heuristics — declared per rule in the ``offline`` block of phase1_rules.json — to
eliminate obviously-safe calls and assign a heuristic confidence to the rest. No
model is contacted.

The heuristics, by design, look across the *whole execution flow* (the calling
function, the functions on the call path, and the other functions those path
functions call) when searching for a mitigating bounds/validation check, mirroring
the interprocedural context the LLM path builds.

Offline rule vocabulary (declarative, evaluated by the predicates below):
  base_confidence: "high"|"medium"|"low"   confidence when nothing mitigates.
  constant_safe_args: [int]                if ALL listed args are compile-time
                                           constants -> eliminate (safe).
  eliminate_if_all_args_constant: bool     if EVERY arg is constant -> eliminate.
  format_arg_lookup: bool                  format-string rules: eliminate when the
                                           (function-specific) format arg is a
                                           constant literal.
  length_guard_vars: [int]                 if a length/validation check on any of
                                           these arg variables is found anywhere on
                                           the flow -> downgrade the confidence.
  skip: bool                               rule is not evaluated offline.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from . import context_builder as ctx
from . import debuglog as _dbg
from . import phase1 as _phase1
from . import prototypes
from .findings import Finding, clip_context
from .rules import RuleSet

# Length / size helper functions whose presence near an argument indicates a
# bounds check on it.
_LEN_FUNCS = (
    "strlen", "strnlen", "wcslen", "wcsnlen", "_tcslen", "lstrlena", "lstrlenw",
    "strlen_s", "strnlen_s", "sizeof",
)

_DOWNGRADE = {"high": "medium", "medium": "low", "low": "low"}
_MAX_FLOW_FUNCS = 40
_MAX_SCAN_INSTRS = 4000
# Comparison tokens that signal a value is actually being range/bounds-checked
# (as opposed to merely mentioned inside a branch body).
_CMP_TOKENS = ("==", "!=", "<=", ">=", "<", ">", "u<", "u>", "u<=", "u>=")
# HLIL conditional/loop operations whose CONDITION can carry a bounds check.
_COND_OPS = (
    "HLIL_IF", "HLIL_WHILE", "HLIL_WHILE_SSA", "HLIL_DO_WHILE",
    "HLIL_DO_WHILE_SSA", "HLIL_FOR", "HLIL_FOR_SSA",
)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _params_of_call(instr) -> list:
    if instr is None:
        return []
    for ce in (list(ctx._iter_call_exprs(instr)) or [instr]):
        params = _safe(lambda ce=ce: list(ce.params))
        if params:
            return params
    return []


def _arg_is_constant(params: list, idx: int) -> bool:
    if idx < 0 or idx >= len(params):
        return False
    p = params[idx]
    # A var-free HLIL expression is a literal constant ...
    if not (_safe(lambda: list(p.vars)) or []):
        return True
    # ... and so is a variable that Binary Ninja's constant propagation pinned to
    # a fixed value (e.g. a size set to 0x40 upstream), which the var-emptiness
    # check alone would miss.
    return ctx.expr_constant_value(p) is not None


def _all_args_constant(params: list) -> bool:
    if not params:
        return False
    return all(_arg_is_constant(params, i) for i in range(len(params)))


def _arg_var_names(params: list, idxs) -> List[str]:
    names: List[str] = []
    for idx in idxs or []:
        if 0 <= idx < len(params):
            for v in (_safe(lambda p=params[idx]: list(p.vars)) or []):
                n = _safe(lambda v=v: v.name)
                if n and n not in names:
                    names.append(n)
    return names


def _format_arg_index(matched_name: str) -> Optional[int]:
    return prototypes.format_arg_index(matched_name)


def _flow_functions(bv, caller, cfg) -> list:
    """Caller + functions on the call path + the OTHER functions those path
    functions call (siblings) — the interprocedural surface a guard could live in."""
    funcs = {}
    cstart = _safe(lambda: caller.start)
    if cstart is not None:
        funcs[cstart] = caller
    paths = _safe(lambda: ctx.call_paths(bv, caller, cfg.call_path_max_depth,
                                         cfg.call_path_max_paths)) or []
    for path in paths:
        for f in path:
            s = _safe(lambda f=f: f.start)
            if s is not None:
                funcs[s] = f
    for f in list(funcs.values()):
        for c in (_safe(lambda f=f: list(f.callees)) or []):
            if not ctx._has_body(c):
                continue
            s = _safe(lambda c=c: c.start)
            if s is not None and s not in funcs:
                funcs[s] = c
            if len(funcs) >= _MAX_FLOW_FUNCS:
                break
        if len(funcs) >= _MAX_FLOW_FUNCS:
            break
    return list(funcs.values())


def _condition_checks_var(instr, var_names: List[str]) -> bool:
    """True if ``instr`` is a branch/loop whose CONDITION compares one of the named
    variables (a genuine range/bounds check), rather than merely using it in a body."""
    if ctx._op_name(instr) not in _COND_OPS:
        return False
    cond = _safe(lambda i=instr: i.condition)
    if cond is None:
        return False
    ctext = _safe(lambda c=cond: str(c), "") or ""
    if not any(vn in ctext for vn in var_names):
        return False
    return any(tok in ctext for tok in _CMP_TOKENS)


def _flow_has_guard(flow_funcs: list, var_names: List[str]) -> bool:
    """True if any flow function bounds one of the named variables.

    A bound counts when the variable is the argument of a length/size helper
    (strlen/sizeof/...) OR is actually COMPARED inside an if/loop condition. The
    earlier version treated any ``if`` whose body merely mentioned the variable as
    a guard, which over-credited unrelated branches; requiring a comparison in the
    condition makes the downgrade signal meaningful. Name-based, so it spans
    function boundaries when symbols exist."""
    if not var_names:
        return False
    for f in flow_funcs:
        hlil = _safe(lambda f=f: f.hlil)
        if hlil is None:
            continue
        instrs = _safe(lambda: list(hlil.instructions)) or []
        for instr in instrs[:_MAX_SCAN_INSTRS]:
            text = _safe(lambda i=instr: str(i), "") or ""
            if any(vn in text for vn in var_names) and \
                    any(lf in text.lower() for lf in _LEN_FUNCS):
                return True
            if _condition_checks_var(instr, var_names):
                return True
    return False


def run_offline_phase1(
    bv,
    ruleset: RuleSet,
    cfg,
    *,
    on_status: Callable[[str], None] = lambda s: None,
    on_finding: Callable[[Finding], None] = lambda f: None,
    on_progress: Callable[[int, int], None] = lambda c, t: None,
    on_site_error: Callable[[str], None] = lambda s: None,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> List[Finding]:
    """Programmatic Phase 1 with no LLM. Emits findings incrementally."""
    on_status("Offline scan: locating dangerous function calls...")
    _dbg.log("offline: collecting dangerous call sites")
    sites = list(_phase1.collect_call_sites(bv, ruleset, cfg))
    total = len(sites)
    _dbg.log(f"offline: {total} call site(s) found; applying programmatic rules")
    on_status(f"Offline scan: {total} call site(s); applying programmatic rules...")

    findings: List[Finding] = []
    eliminated = 0
    skipped = 0    # rule has no offline heuristic (needs the LLM)
    failed = 0     # analysis raised for this site
    cancelled = False

    for idx, (rule, matched_name, caller, call_addr) in enumerate(sites):
        if is_cancelled():
            on_status("Offline scan: cancelled.")
            cancelled = True
            break
        on_progress(idx + 1, total)  # 1-based: report the item being worked on
        _dbg.log(f"offline site {idx + 1}/{total}: rule={rule.id} call={matched_name} "
                 f"in {_dbg.sym(_safe(lambda c=caller: c.name))} @ {_dbg.addr(call_addr)}")

        off = rule.offline or {}
        if not off or off.get("skip"):
            skipped += 1
            _dbg.log("  -> skipped: rule has no offline heuristic (needs the LLM)")
            continue

        try:
            instr = ctx.hlil_instr_at(caller, call_addr)
            params = _params_of_call(instr)
        except Exception as exc:  # never abort the scan
            failed += 1
            on_site_error(f"offline analysis failed at {call_addr:#x}: {exc}")
            _dbg.log(f"  -> analysis FAILED: {exc}")
            continue

        # --- Elimination: obviously not attacker-controlled -----------------
        csa = off.get("constant_safe_args")
        if csa and params and all(_arg_is_constant(params, i) for i in csa):
            eliminated += 1
            _dbg.log(f"  -> eliminated: constant-safe args {csa} are all constant")
            continue
        if off.get("eliminate_if_all_args_constant") and _all_args_constant(params):
            eliminated += 1
            _dbg.log("  -> eliminated: every argument is a compile-time constant")
            continue
        if off.get("format_arg_lookup"):
            fidx = _format_arg_index(matched_name)
            if fidx is not None and _arg_is_constant(params, fidx):
                eliminated += 1
                _dbg.log(f"  -> eliminated: format arg #{fidx} is a constant literal")
                continue

        # --- Confidence: downgrade when a mitigating check is on the flow ---
        confidence = off.get("base_confidence", "medium")
        guard_idxs = off.get("length_guard_vars") or []
        notes = []
        if guard_idxs:
            # 1) Deterministic: did Binary Ninja's value-set analysis prove the
            #    size/length argument is a constant or bounded by a range? That is
            #    a strong "not freely attacker-controlled" signal and needs no flow
            #    scan.
            bounded = []
            for gi in guard_idxs:
                if 0 <= gi < len(params):
                    mx = _safe(lambda gi=gi: ctx.expr_max_value(params[gi]))
                    if mx is not None:
                        bounded.append((gi, mx))
            if bounded:
                confidence = _DOWNGRADE.get(confidence, confidence)
                notes.append(
                    "Binary Ninja resolved a constant/bounded value for the size/"
                    "length argument(s) "
                    + ", ".join(f"#{gi} (<= {mx:#x})" for gi, mx in bounded)
                    + ", so it is constrained rather than freely attacker-controlled"
                )
            else:
                # 2) Otherwise look for a real bounds/validation check on the flow.
                var_names = _arg_var_names(params, guard_idxs)
                if var_names:
                    try:
                        flow = _flow_functions(bv, caller, cfg)
                        if _flow_has_guard(flow, var_names):
                            confidence = _DOWNGRADE.get(confidence, confidence)
                            notes.append(
                                f"a length/bounds check comparing {var_names} was "
                                "found on the execution flow, so confidence is reduced"
                            )
                        else:
                            notes.append(
                                f"no length/bounds check for {var_names} was found "
                                "anywhere on the execution flow"
                            )
                    except Exception as exc:
                        on_site_error(f"guard search failed at {call_addr:#x}: {exc}")

        caller_name = _safe(lambda: caller.name) or "?"
        explanation = (
            f"Programmatic offline heuristic flagged a call to {matched_name} "
            f"({rule.name}). " + (" ".join(notes) if notes else
            "No mitigating check was identified by the offline rules.")
            + " This was NOT analyzed by an LLM; review manually or run a full scan."
        )

        # Build the SAME full interprocedural context an online scan would send,
        # and keep it so triaged offline findings can be exported as fine-tuning
        # data. Only done for flagged findings (eliminated ones are skipped above).
        context_excerpt = ""
        call_paths_text = ""
        representation = "offline-heuristic"
        if getattr(cfg, "offline_build_context", True):
            try:
                bundle = ctx.build_call_site_context(
                    bv, caller, matched_name, call_addr, cfg,
                    tainted_source_set=ruleset.tainted_source_set,
                )
                context_excerpt = clip_context(bundle.text, getattr(cfg, "stored_context_chars", 0))
                call_paths_text = bundle.call_paths_text
                representation = bundle.representation or "offline-heuristic"
            except Exception as exc:
                on_site_error(f"offline context build failed at {call_addr:#x}: {exc}")

        finding = Finding(
            phase=1,
            rule_id=rule.id,
            severity=rule.severity,
            confidence=confidence,
            title=rule.name,
            function_name=caller_name,
            function_addr=_safe(lambda: caller.start) or 0,
            address=call_addr,
            cwe=rule.cwe,
            explanation=explanation,
            representation=representation,
            call_paths=call_paths_text,
            context_excerpt=context_excerpt,
            scratchpad="; ".join(notes),
            model="offline-heuristic",
        )
        findings.append(finding)
        _dbg.log(f"  -> FLAGGED ({confidence}): {rule.name}"
                 + (f"; notes: {'; '.join(notes)}" if notes else ""))
        on_finding(finding)

    on_progress(total, total)
    accounted = len(findings) + eliminated + skipped + failed
    # flagged + eliminated + skipped(no offline rule) + failed [+ unprocessed if cancelled]
    _dbg.log(f"offline {'cancelled' if cancelled else 'complete'}: {len(findings)} flagged, "
             f"{eliminated} eliminated, {skipped} skipped (no offline heuristic), "
             f"{failed} failed = {accounted} of {total} site(s)"
             + (f" ({total - accounted} not processed — cancelled)" if cancelled else ""))
    skip_note = f", {skipped} need the LLM (no offline rule)" if skipped else ""
    fail_note = f", {failed} failed" if failed else ""
    on_status(
        f"Offline scan complete: {len(findings)} flagged, {eliminated} eliminated as "
        f"obviously safe{skip_note}{fail_note} (of {total} call sites). No LLM was used."
    )
    return findings
