"""VulnFanatic-NG sidebar widget.

A tabbed panel: tab 0 is always "New Scan" (name + custom rule paths + Start Scan /
Scan Offline). Each completed scan becomes its own closable result tab, and ALL
scans are stored in the BNDB so you can keep an offline scan and a later online
scan (or runs with different rule sets) side by side.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import binaryninja
import binaryninjaui  # noqa: F401  (import before PySide6)
from binaryninjaui import (
    Sidebar,
    SidebarContextSensitivity,
    SidebarWidget,
    SidebarWidgetLocation,
    SidebarWidgetType,
    UIActionHandler,
)
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import findings as _findings
from .. import settings as _settings
from ..findings import ScanRecord
from ..scanner import VulnFanaticScan
from .findings_view import FindingsView


def _log_warn(msg: str):
    try:
        binaryninja.log_warn(f"[VulnFanatic-NG] {msg}")
    except Exception:
        pass


class _ViewState:
    """Per-BinaryView UI state. Keeping each open binary's tabs / running scan /
    progress separate means a scan running in one file never appears in (or blocks
    switching to) another — the widget is self-managed and keyed by view identity."""

    def __init__(self, bv):
        self.bv = bv
        self.views: List[FindingsView] = []        # this binary's result-tab widgets
        self.scan: Optional[VulnFanaticScan] = None
        self.active_view: Optional[FindingsView] = None
        self.scanning = False
        self.site_error_count = 0
        self.progress = None                       # (cur, total, label); None = indeterminate
        self.status = "Open a binary, then Start Scan or Scan Offline."
        self.loaded = False


class VulnFanaticSidebarWidget(SidebarWidget):
    def __init__(self, name, frame, data):
        SidebarWidget.__init__(self, name)
        self.view_frame = frame
        self.bv = None
        self._states: dict = {}      # view identity -> _ViewState

        self.actionHandler = UIActionHandler()
        self.actionHandler.setupActionHandler(self)

        self._build_ui()
        self._show_binary(data, force=True)

    # UI construction -------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("VulnFanatic-NG")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        self.tabs = QTabWidget(self)
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self._on_tab_close)
        self.tabs.addTab(self._build_new_scan_tab(), "New Scan")
        # The New Scan tab (index 0) is permanent — remove its close button.
        bar = self.tabs.tabBar()
        for side in (QTabBar.RightSide, QTabBar.LeftSide):
            bar.setTabButton(0, side, None)
        layout.addWidget(self.tabs, 1)
        self.setLayout(layout)

    def _build_new_scan_tab(self) -> QWidget:
        w = QWidget(self)
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("(auto: timestamp + mode)")
        form.addRow("Scan name:", self.name_edit)

        self.p1_edit = QLineEdit()
        self.p1_edit.setPlaceholderText("(default phase1_rules.json)")
        form.addRow("Phase 1 rules:", self._path_row(self.p1_edit))

        self.p2_edit = QLineEdit()
        self.p2_edit.setPlaceholderText("(default phase2_rules.json)")
        form.addRow("Phase 2 rules:", self._path_row(self.p2_edit))

        self.p3_edit = QLineEdit()
        self.p3_edit.setPlaceholderText("(default phase3_rules.json)")
        form.addRow("Phase 3 rules:", self._path_row(self.p3_edit))
        v.addLayout(form)

        self.phase3_check = QCheckBox(
            "Phase 3: firmware hardware-attack hardening (fault-injection / "
            "side-channel) — online only")
        self.phase3_check.setToolTip(
            "Audits whether the binary implements firmware hardening controls "
            "against fault-injection and side-channel attacks. Requires the LLM "
            "(applies only to Start Scan, not Scan Offline). Off by default.")
        self.phase3_check.setChecked(False)
        v.addWidget(self.phase3_check)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Scan")
        self.start_btn.setToolTip("Full scan using the configured LLM.")
        self.start_btn.clicked.connect(self._on_start)
        self.offline_btn = QPushButton("Scan Offline")
        self.offline_btn.setToolTip(
            "Programmatic Phase 1 only — no LLM. Flags dangerous calls and "
            "eliminates obviously-safe ones with a heuristic confidence.")
        self.offline_btn.clicked.connect(self._on_start_offline)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.offline_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)

        self.export_btn = QPushButton("Export current scan…")
        self.export_btn.clicked.connect(self._on_export)
        v.addWidget(self.export_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        v.addWidget(self.progress)

        self.status_label = QLabel("Open a binary, then Start Scan or Scan Offline.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: gray;")
        v.addWidget(self.status_label)
        v.addStretch(1)
        return w

    def _path_row(self, edit: QLineEdit) -> QWidget:
        row = QWidget(self)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_rules(edit))
        h.addWidget(browse)
        return row

    def _browse_rules(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select rules JSON", edit.text() or "", "JSON (*.json);;All files (*)")
        if path:
            edit.setText(path)

    # Result tabs -----------------------------------------------------------
    def _result_views(self) -> List[FindingsView]:
        """Result-tab widgets currently displayed (i.e. the active binary's)."""
        return [self.tabs.widget(i) for i in range(self.tabs.count())
                if isinstance(self.tabs.widget(i), FindingsView)]

    def _create_view(self, record: ScanRecord) -> FindingsView:
        view = FindingsView(self)
        view.scan_record = record
        view.set_context(self.bv, self.view_frame)
        # Persist to the binary this view belongs to (captured now), not "current".
        view.set_persist_callback(lambda bv=self.bv: self._save_state(self._state(bv)))
        view.set_findings(record.findings)
        return view

    def _make_result_tab(self, record: ScanRecord, select: bool = False) -> FindingsView:
        """Create a result view for the CURRENT binary, track it in that binary's
        state, and show it as a tab."""
        view = self._create_view(record)
        st = self._state()
        if st is not None:
            st.views.append(view)
        idx = self.tabs.addTab(view, self._tab_title(record, len(record.findings)))
        if select:
            self.tabs.setCurrentIndex(idx)
        return view

    @staticmethod
    def _tab_title(record: ScanRecord, count: int, scanning: bool = False) -> str:
        if scanning:
            return f"{record.name}  (scanning…)"
        return f"{record.name} ({count})"

    # Per-binary state ------------------------------------------------------
    def _key(self, bv):
        return id(bv) if bv is not None else None

    def _state(self, bv=None) -> Optional[_ViewState]:
        """Get (creating if needed) the state for ``bv`` (default: the active binary)."""
        bv = self.bv if bv is None else bv
        if bv is None:
            return None
        k = self._key(bv)
        st = self._states.get(k)
        if st is None:
            st = _ViewState(bv)
            self._states[k] = st
        return st

    def _save_state(self, state: Optional[_ViewState]):
        if state is None or state.bv is None:
            return
        records = []
        for view in state.views:
            if view.scan_record is not None:
                view.scan_record.findings = view.findings()
                records.append(view.scan_record)
        try:
            _findings.save_scans(state.bv, records)
        except Exception as exc:
            _log_warn(f"failed to save scans: {exc}")

    def _save_all_scans(self):
        self._save_state(self._state())

    def _on_tab_close(self, index: int):
        if index == 0:
            return  # New Scan tab is permanent
        view = self.tabs.widget(index)
        name = ""
        if isinstance(view, FindingsView) and view.scan_record is not None:
            name = view.scan_record.name
        if not name:
            name = self.tabs.tabText(index)
        if not self._confirm_delete_scan(name):
            return  # not confirmed — keep the tab and its results
        st = self._state()
        if st is not None and isinstance(view, FindingsView) and view in st.views:
            st.views.remove(view)
        self.tabs.removeTab(index)
        # Re-persist the remaining scans, which removes this one from the BNDB.
        self._save_all_scans()
        self._status(f"Deleted scan '{name}' and its results from the database.")
        self._update_states()

    def _confirm_delete_scan(self, name: str) -> bool:
        """Require an explicit checkbox confirmation before permanently deleting a
        scan's results from the BNDB. Returns True only if confirmed."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Delete scan results")
        v = QVBoxLayout(dlg)
        msg = QLabel(
            f"Closing this tab will PERMANENTLY DELETE the results of scan "
            f"“{name}” from the Binary Ninja database (to free space). "
            "This cannot be undone.")
        msg.setWordWrap(True)
        v.addWidget(msg)
        confirm = QCheckBox(
            f"I confirm that I will lose the results from “{name}” forever.")
        v.addWidget(confirm)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        delete_btn = buttons.addButton("Delete results forever",
                                       QDialogButtonBox.DestructiveRole)
        delete_btn.setEnabled(False)
        confirm.toggled.connect(delete_btn.setEnabled)
        delete_btn.clicked.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        return dlg.exec() == QDialog.Accepted and confirm.isChecked()

    # State handling --------------------------------------------------------
    def _update_states(self):
        st = self._state()
        scanning = bool(st and st.scanning)
        has_bv = self.bv is not None
        self.start_btn.setEnabled(has_bv and not scanning)
        self.start_btn.setText("Scanning…" if scanning else "Start Scan")
        self.offline_btn.setEnabled(has_bv and not scanning)
        self.cancel_btn.setEnabled(scanning)
        self.cancel_btn.setVisible(scanning)
        self.export_btn.setEnabled(len(self._result_views()) > 0)

    def _status(self, text: str, state: Optional[_ViewState] = None):
        """Record status on a binary's state; reflect it on the label only if that
        binary is the one currently shown (so a background scan doesn't overwrite it)."""
        st = state if state is not None else self._state()
        if st is not None:
            st.status = text
        if st is None or st.bv is self.bv:
            self.status_label.setText(text)

    # Binary Ninja notifications -------------------------------------------
    def notifyViewChanged(self, view_frame):
        self.view_frame = view_frame
        new_bv = None
        if view_frame is not None:
            try:
                view = view_frame.getCurrentViewInterface()
                new_bv = view.getData() if view is not None else None
            except Exception:
                new_bv = None
        self._show_binary(new_bv)

    def _show_binary(self, bv, force: bool = False):
        """Make ``bv`` the displayed binary: swap in its result tabs and restore its
        scan/progress/status. Per-binary state means a scan running in another open
        binary keeps running there (and stays visible there) without leaking here."""
        if bv is self.bv and not force:
            for view in self._result_views():
                view.set_context(self.bv, self.view_frame)
            self._refresh_from_state(self._state(bv))
            self._update_states()
            return
        # Detach the previously shown binary's result tabs — their FindingsView objects
        # stay alive in that binary's state so we can reattach them on switch-back.
        for i in range(self.tabs.count() - 1, 0, -1):
            self.tabs.removeTab(i)
        self.bv = bv
        st = self._state(bv)
        if st is not None:
            if not st.loaded:
                st.loaded = True
                try:
                    records = _findings.load_scans(bv)
                except Exception:
                    records = []
                st.views = [self._create_view(r) for r in records]
                if records:
                    st.status = f"Loaded {len(records)} saved scan(s) for this binary."
            for view in st.views:
                view.set_context(self.bv, self.view_frame)
                scanning = st.scanning and view is st.active_view
                self.tabs.addTab(view, self._tab_title(
                    view.scan_record, view.count, scanning=scanning))
        self._refresh_from_state(st)
        self._update_states()

    def _refresh_from_state(self, st: Optional[_ViewState]):
        """Restore the shared status label + progress bar from a binary's state."""
        if st is None:
            self.status_label.setText("Open a binary, then Start Scan or Scan Offline.")
            self.progress.setVisible(False)
            return
        self.status_label.setText(st.status)
        if st.scanning:
            if st.progress is None:
                self.progress.setRange(0, 0)
            else:
                cur, total, label = st.progress
                self.progress.setRange(0, total if total > 0 else 0)
                self.progress.setValue(cur)
                self.progress.setFormat(f"{label}: %v/%m")
            self.progress.setVisible(True)
        else:
            self.progress.setVisible(False)

    # Scan lifecycle --------------------------------------------------------
    def _on_start(self):
        self._begin_scan(offline=False)

    def _on_start_offline(self):
        self._begin_scan(offline=True)

    def _begin_scan(self, offline: bool):
        if self.bv is None:
            self._status("Open a binary first.")
            return
        st = self._state()
        if st is None or st.scanning:
            return
        cfg = _settings.get_config(self.bv)
        # Per-scan custom rule paths override the configured defaults.
        p1 = self.p1_edit.text().strip()
        p2 = self.p2_edit.text().strip()
        p3 = self.p3_edit.text().strip()
        if p1:
            cfg.rules_phase1_path = p1
        if p2:
            cfg.rules_phase2_path = p2
        if p3:
            cfg.rules_phase3_path = p3
        # Phase 3 is online-only and controlled by the per-scan checkbox.
        cfg.run_phase3 = (not offline) and self.phase3_check.isChecked()

        if not offline:
            err = cfg.validation_error()
            if err:
                QMessageBox.warning(
                    self, "VulnFanatic-NG",
                    f"{err}\n\nConfigure under Settings (search 'vulnfanatic'): set "
                    "the API base URL and model. Or use 'Scan Offline' (no LLM).",
                )
                self._status(err)
                return

        mode = "offline" if offline else ("test" if cfg.is_test_mode else "online")
        name = self.name_edit.text().strip() or \
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {mode}"
        record = ScanRecord(
            name=name, mode=mode,
            rules_phase1_path=cfg.rules_phase1_path,
            rules_phase2_path=cfg.rules_phase2_path,
        )

        # Live result tab the findings stream into (tracked in this binary's state).
        view = self._make_result_tab(record, select=True)
        st.active_view = view
        idx = self.tabs.indexOf(view)
        self.tabs.setTabText(idx, self._tab_title(record, 0, scanning=True))

        st.site_error_count = 0
        st.scanning = True
        st.progress = None
        self.name_edit.clear()  # next scan auto-names by timestamp
        self.progress.setRange(0, 0)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._update_states()

        scan = VulnFanaticScan(self.bv, cfg, offline=offline)
        st.scan = scan
        sig = scan.signals
        # Bind every signal to THIS binary's state so a background scan never touches
        # another open binary's UI.
        sig.status.connect(lambda m, s=st: self._status(m, s))
        sig.finding.connect(lambda f, s=st: self._on_finding(s, f))
        sig.progress.connect(lambda c, t, l, s=st: self._on_progress(s, c, t, l))
        sig.site_error.connect(lambda m, s=st: self._on_site_error(s, m))
        sig.finished.connect(lambda summ, s=st: self._on_finished(s, summ))
        sig.error.connect(lambda m, s=st: self._on_error(s, m))
        scan.start()

    def _on_cancel(self):
        st = self._state()
        if st is not None and st.scan is not None:
            st.scan.cancel()
            self._status("Cancelling after the current call site…")

    def _on_finding(self, st: _ViewState, finding):
        view = st.active_view
        if view is None:
            return
        view.add_finding(finding)
        if st.bv is self.bv:  # update the tab title only if this binary is shown
            idx = self.tabs.indexOf(view)
            if idx >= 0 and view.scan_record is not None:
                self.tabs.setTabText(
                    idx, self._tab_title(view.scan_record, view.count, scanning=True))

    def _on_progress(self, st: _ViewState, cur: int, total: int, label: str):
        st.progress = (cur, total, label)
        if st.bv is self.bv:
            if total > 0:
                self.progress.setRange(0, total)
                self.progress.setValue(cur)
            self.progress.setFormat(f"{label}: %v/%m")

    def _on_site_error(self, st: _ViewState, msg: str):
        st.site_error_count += 1
        _log_warn(msg)

    def _finalize_active_tab(self, st: _ViewState):
        view = st.active_view
        if view is None:
            return
        if view.scan_record is not None:
            view.scan_record.findings = view.findings()
            if st.bv is self.bv:
                idx = self.tabs.indexOf(view)
                if idx >= 0:
                    self.tabs.setTabText(idx, self._tab_title(view.scan_record, view.count))
        self._save_state(st)
        st.active_view = None

    def _on_finished(self, st: _ViewState, summary: dict):
        st.scanning = False
        st.scan = None
        self._finalize_active_tab(st)

        cancelled = summary.get("cancelled")
        total = summary.get("total", 0)
        errors = summary.get("site_errors", 0)
        offline = summary.get("offline")
        if offline:
            prefix = "Offline scan cancelled" if cancelled else "Offline scan complete"
            msg = f"{prefix}: {total} flagged (heuristic, no LLM)"
        else:
            prefix = "Scan cancelled" if cancelled else "Scan complete"
            verb = "candidate(s)" if summary.get("test_mode") else "issue(s)"
            parts = [f"Phase 1: {summary.get('phase1', 0)}",
                     f"Phase 2: {summary.get('phase2', 0)}"]
            if summary.get("phase3", 0):
                parts.append(f"Phase 3: {summary.get('phase3', 0)}")
            msg = f"{prefix}: {total} {verb} ({', '.join(parts)})"
            dbg = []
            if summary.get("rejected"):
                dbg.append(f"{summary['rejected']} rejected")
            if summary.get("skipped"):
                dbg.append(f"{summary['skipped']} skipped")
            if summary.get("errored"):
                dbg.append(f"{summary['errored']} errored")
            if dbg:
                msg += f"; debug rows: {', '.join(dbg)}"
        if errors:
            msg += f". {errors} analysis error(s) — see the log."
        if summary.get("test_mode") and summary.get("test_dir"):
            msg += f". TEST MODE: prompts saved under {summary['test_dir']}"
        self._status(msg, st)
        if st.bv is self.bv:
            self._update_states()
            self._refresh_from_state(st)

    def _on_error(self, st: _ViewState, msg: str):
        st.scanning = False
        st.scan = None
        # Drop the (empty) live tab if nothing was found before the error.
        view = st.active_view
        if view is not None and view.count == 0:
            if st.bv is self.bv:
                idx = self.tabs.indexOf(view)
                if idx > 0:
                    self.tabs.removeTab(idx)
            if view in st.views:
                st.views.remove(view)
            st.active_view = None
        else:
            self._finalize_active_tab(st)
        self._status(f"Error: {msg}", st)
        if st.bv is self.bv:
            self._update_states()
            self._refresh_from_state(st)
        QMessageBox.critical(self, "VulnFanatic-NG", msg)

    # Export ----------------------------------------------------------------
    def _on_export(self):
        view = self.tabs.currentWidget()
        if not isinstance(view, FindingsView) or view.count == 0:
            self._status("Select a scan tab with findings to export.")
            return
        default = "vulnfanatic_findings.md"
        if view.scan_record is not None:
            safe = "".join(c if c.isalnum() or c in "-_." else "_"
                           for c in view.scan_record.name)
            default = f"{safe or 'scan'}.md"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export scan", default, "Markdown (*.md);;JSON (*.json)")
        if not path:
            return
        items = view.findings()
        bin_name = ""
        try:
            bin_name = self.bv.file.original_filename if self.bv else ""
        except Exception:
            pass
        try:
            if path.lower().endswith(".json"):
                content = _findings.to_json(items)
            else:
                content = _findings.to_markdown(items, bin_name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            self._status(f"Exported {len(items)} finding(s) to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "VulnFanatic-NG", f"Export failed: {exc}")


class VulnFanaticSidebarWidgetType(SidebarWidgetType):
    def __init__(self):
        icon = QImage(56, 56, QImage.Format_RGB32)
        icon.fill(0)
        painter = QPainter()
        painter.begin(icon)
        painter.setFont(QFont("Open Sans", 22, QFont.Bold))
        painter.setPen(QColor(255, 255, 255, 255))
        painter.drawText(QRectF(0, 0, 56, 56), Qt.AlignCenter, "VF")
        painter.end()
        SidebarWidgetType.__init__(self, icon, "VulnFanatic-NG")

    def createWidget(self, frame, data):
        return VulnFanaticSidebarWidget("VulnFanatic-NG", frame, data)

    def defaultLocation(self):
        return SidebarWidgetLocation.RightContent

    def contextSensitivity(self):
        # Single self-managed widget: it keeps per-binary state internally (keyed by
        # view identity) and swaps the displayed tabs/scan/progress on notifyViewChanged.
        # This is robust regardless of whether the build hands out per-view instances.
        return SidebarContextSensitivity.SelfManagedSidebarContext


_registered = False


def register():
    """Register the sidebar widget type with Binary Ninja (idempotent)."""
    global _registered
    if _registered:
        return
    Sidebar.addSidebarWidgetType(VulnFanaticSidebarWidgetType())
    _registered = True
