"""Background scan orchestration.

``VulnFanaticScan`` is a Binary Ninja ``BackgroundTaskThread`` that runs Phase 1 and
Phase 2 off the UI thread and reports results through Qt signals. Qt delivers
signals emitted from this worker thread to slots on the UI thread via its queued
connection mechanism, so the widget never touches Binary Ninja UI objects from a
background thread.
"""

from __future__ import annotations

from binaryninja import BackgroundTaskThread

# Import binaryninjaui before PySide6 so the correct bindings load.
import binaryninjaui  # noqa: F401
from PySide6.QtCore import QObject, Signal

from . import debuglog as _dbg
from . import offline as _offline
from . import phase1 as _phase1
from . import phase2 as _phase2
from .analyzers import LLMAnalyzer, TestModeAnalyzer, test_output_dir
from .findings import (
    TRIAGE_ERROR, TRIAGE_NON_ISSUE, TRIAGE_REJECTED, TRIAGE_SKIPPED,
)
from .llm import LLMClient
from .rules import RuleError, load_phase1_rules, load_phase2_rules, load_phase3_rules


def _issue_count(findings) -> int:
    """Number of real candidate findings, excluding debug-only bookkeeping rows
    (REJECTED / SKIPPED / ERROR)."""
    return sum(1 for f in findings if f.triage_status not in TRIAGE_NON_ISSUE)


class ScanSignals(QObject):
    status = Signal(str)
    finding = Signal(object)        # Finding
    progress = Signal(int, int, str)  # current, total, phase label
    site_error = Signal(str)
    finished = Signal(object)       # summary dict
    error = Signal(str)             # fatal error -> scan aborted


class VulnFanaticScan(BackgroundTaskThread):
    def __init__(self, bv, cfg, offline: bool = False):
        label = "offline scan" if offline else "scan"
        super().__init__(f"VulnFanatic-NG: starting {label}...", can_cancel=True)
        self.bv = bv
        self.cfg = cfg
        self.offline = offline
        self.signals = ScanSignals()
        self.findings = []
        self._site_errors = 0
        self._test_dir = ""

    # Callback adapters ------------------------------------------------------
    def _on_status(self, text: str):
        self.progress = f"VulnFanatic-NG: {text}"
        self.signals.status.emit(text)

    def _make_on_progress(self, phase_label: str):
        def _cb(cur: int, total: int):
            self.progress = f"VulnFanatic-NG: {phase_label} {cur}/{total}"
            self.signals.progress.emit(cur, total, phase_label)
        return _cb

    def _on_finding(self, finding):
        self.findings.append(finding)
        self.signals.finding.emit(finding)

    def _on_site_error(self, msg: str):
        self._site_errors += 1
        self.signals.site_error.emit(msg)

    def _is_cancelled(self) -> bool:
        return bool(self.cancelled)

    def _bv_name(self):
        try:
            return self.bv.file.original_filename or self.bv.file.filename
        except Exception:
            return None

    # Main ------------------------------------------------------------------
    def run(self):
        try:
            _dbg.configure_from_cfg(self.cfg)
            mode = "offline" if self.offline else ("test" if self.cfg.is_test_mode else "online")
            _dbg.log(
                f"scan start: mode={mode} binary={_dbg.path(self._bv_name())} "
                f"provider={self.cfg.api_provider} model={self.cfg.model or '(none)'} "
                f"reasoning_effort={self.cfg.reasoning_effort} verdict={self.cfg.verdict_reasoning} "
                f"max_ctx={self.cfg.max_context_tokens} max_resp={self.cfg.max_response_tokens} "
                f"phases={'1' if self.cfg.run_phase1 else '-'}"
                f"{'2' if self.cfg.run_phase2 else '-'}{'3' if self.cfg.run_phase3 else '-'}")

            if self.offline:
                self._run_offline()
                return

            err = self.cfg.validation_error()
            if err:
                self.signals.error.emit(err)
                return

            if self.cfg.is_test_mode:
                self._test_dir = test_output_dir(self.bv)
                analyzer = TestModeAnalyzer(self._test_dir)
                self._on_status(
                    f"TEST MODE: the LLM will NOT be called. Flagging every candidate "
                    f"and writing prompts to {self._test_dir}"
                )
            else:
                # Optional separate model for the validation pass (independent second
                # opinion). Only built when validation is on and a validator model is set.
                validator_client = None
                if self.cfg.validation_pass and self.cfg.validator_active:
                    validator_client = LLMClient.validator_from_config(self.cfg)
                    _dbg.log(f"validator: separate model '{self.cfg.validator_model}' "
                             f"({self.cfg.validator_provider}) for the validation pass")
                analyzer = LLMAnalyzer(
                    LLMClient.from_config(self.cfg),
                    validation_pass=self.cfg.validation_pass,
                    reasoning=self.cfg.verdict_reasoning,
                    validator_client=validator_client,
                )

            p1_count = 0
            p2_count = 0
            p3_count = 0

            if self.cfg.run_phase1 and not self._is_cancelled():
                try:
                    ruleset1 = load_phase1_rules(self.cfg.rules_phase1_path)
                except RuleError as exc:
                    self.signals.error.emit(f"Failed to load Phase 1 rules: {exc}")
                    return
                f1 = _phase1.run_phase1(
                    self.bv, ruleset1, self.cfg, analyzer,
                    on_status=self._on_status,
                    on_finding=self._on_finding,
                    on_progress=self._make_on_progress("Phase 1"),
                    on_site_error=self._on_site_error,
                    is_cancelled=self._is_cancelled,
                )
                p1_count = _issue_count(f1)

            if self.cfg.run_phase2 and not self._is_cancelled():
                try:
                    ruleset2 = load_phase2_rules(self.cfg.rules_phase2_path)
                except RuleError as exc:
                    self.signals.error.emit(f"Failed to load Phase 2 rules: {exc}")
                    return
                f2 = _phase2.run_phase2(
                    self.bv, ruleset2, self.cfg, analyzer,
                    on_status=self._on_status,
                    on_finding=self._on_finding,
                    on_progress=self._make_on_progress("Phase 2"),
                    on_site_error=self._on_site_error,
                    is_cancelled=self._is_cancelled,
                )
                p2_count = _issue_count(f2)

            # Phase 3: hardware-attack hardening audit (online only, opt-in).
            if self.cfg.run_phase3 and not self._is_cancelled():
                try:
                    ruleset3 = load_phase3_rules(self.cfg.rules_phase3_path)
                except RuleError as exc:
                    self.signals.error.emit(f"Failed to load Phase 3 rules: {exc}")
                    return
                f3 = _phase2.run_phase2(
                    self.bv, ruleset3, self.cfg, analyzer,
                    on_status=self._on_status,
                    on_finding=self._on_finding,
                    on_progress=self._make_on_progress("Phase 3"),
                    on_site_error=self._on_site_error,
                    is_cancelled=self._is_cancelled,
                    phase_number=3,
                )
                p3_count = _issue_count(f3)

            # Persistence is handled by the sidebar (which records the named scan).
            # Counts exclude debug-only bookkeeping rows (kept in the table, but not
            # "issues"); they are surfaced separately.
            rejected = sum(1 for f in self.findings if f.triage_status == TRIAGE_REJECTED)
            skipped = sum(1 for f in self.findings if f.triage_status == TRIAGE_SKIPPED)
            errored = sum(1 for f in self.findings if f.triage_status == TRIAGE_ERROR)
            total_issues = _issue_count(self.findings)
            summary = {
                "phase1": p1_count,
                "phase2": p2_count,
                "phase3": p3_count,
                "total": total_issues,
                "rejected": rejected,
                "skipped": skipped,
                "errored": errored,
                "site_errors": self._site_errors,
                "cancelled": self._is_cancelled(),
                "test_mode": self.cfg.is_test_mode,
                "test_dir": self._test_dir,
            }
            _dbg.log(f"scan finished: phase1={p1_count} phase2={p2_count} phase3={p3_count} "
                     f"issues={total_issues} rejected={rejected} skipped={skipped} "
                     f"errored={errored} errors={self._site_errors} "
                     f"cancelled={self._is_cancelled()}")
            self.signals.finished.emit(summary)
        except Exception as exc:  # pragma: no cover - defensive
            import traceback
            traceback.print_exc()
            self.signals.error.emit(f"Scan aborted: {exc}")
        finally:
            self.finish()

    def _run_offline(self):
        """Programmatic Phase 1 only, no LLM (model/URL config not required)."""
        try:
            try:
                ruleset1 = load_phase1_rules(self.cfg.rules_phase1_path)
            except RuleError as exc:
                self.signals.error.emit(f"Failed to load Phase 1 rules: {exc}")
                return
            f1 = _offline.run_offline_phase1(
                self.bv, ruleset1, self.cfg,
                on_status=self._on_status,
                on_finding=self._on_finding,
                on_progress=self._make_on_progress("Offline"),
                on_site_error=self._on_site_error,
                is_cancelled=self._is_cancelled,
            )
            self.signals.finished.emit({
                "phase1": len(f1),
                "phase2": 0,
                "total": len(self.findings),
                "site_errors": self._site_errors,
                "cancelled": self._is_cancelled(),
                "offline": True,
            })
        except Exception as exc:  # pragma: no cover - defensive
            import traceback
            traceback.print_exc()
            self.signals.error.emit(f"Offline scan aborted: {exc}")
        # Note: finish() is called by run()'s finally, so not here.
