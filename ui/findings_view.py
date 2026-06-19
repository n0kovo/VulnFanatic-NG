"""Findings table + detail pane with click-to-navigate."""

from __future__ import annotations

import os
from typing import List, Optional

import binaryninjaui  # noqa: F401  (import before PySide6)
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .. import findings as _findings
from ..findings import (
    Finding,
    TRIAGE_ERROR,
    TRIAGE_FALSE,
    TRIAGE_REAL,
    TRIAGE_REJECTED,
    TRIAGE_SKIPPED,
    TRIAGE_UNTRIAGED,
    sort_findings,
)

_COLUMNS = ["Status", "Confidence", "Phase", "CWE", "Function", "Address", "Title",
            "Prompt File"]
_TITLE_COL = 6
_PROMPT_COL = 7

_CONFIDENCE_COLORS = {
    "high": QColor(200, 70, 50),
    "medium": QColor(190, 150, 40),
    "low": QColor(120, 130, 140),
    "unknown": QColor(150, 120, 170),  # muted purple — model produced no verdict
}

_REAL_COLOR = QColor(70, 170, 90)     # green — confirmed real issue
_FP_COLOR = QColor(130, 130, 130)     # grey — false positive
_REJECTED_COLOR = QColor(110, 110, 120)  # dim grey — LLM rejected (debug)
_ERROR_COLOR = QColor(200, 130, 50)   # amber — could not be analyzed (debug)

# Debug-only statuses rendered dim + italic (de-emphasised vs real findings).
_DEBUG_ROW_COLORS = {
    TRIAGE_REJECTED: _REJECTED_COLOR,
    TRIAGE_SKIPPED: _REJECTED_COLOR,
    TRIAGE_ERROR: _ERROR_COLOR,
}

# Right-click menu actions: status -> label.
_MARK_LABELS = {
    TRIAGE_REAL: "Mark as Real Issue",
    TRIAGE_FALSE: "Mark as False Positive",
    TRIAGE_UNTRIAGED: "Mark as Untriaged",
}


def navigate_to(view_frame, bv, addr: int) -> bool:
    """Navigate the active Binary Ninja view to ``addr``. Best-effort."""
    if addr is None:
        return False
    if view_frame is not None:
        for arg in (bv, getattr(bv, "view", None)):
            if arg is None:
                continue
            try:
                if view_frame.navigate(arg, addr):
                    return True
            except Exception:
                pass
    try:
        from binaryninjaui import UIContext

        ctx = UIContext.activeContext()
        if ctx is not None:
            vf = ctx.getCurrentViewFrame()
            if vf is not None:
                for arg in (bv, getattr(bv, "view", None)):
                    if arg is None:
                        continue
                    try:
                        if vf.navigate(arg, addr):
                            return True
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        return bool(bv.navigate(bv.view, addr))
    except Exception:
        return False


class _SortItem(QTableWidgetItem):
    """Table item that sorts by a stored numeric key instead of its text."""

    def __init__(self, text: str, sort_key):
        super().__init__(text)
        self._sort_key = sort_key

    def __lt__(self, other):
        try:
            return self._sort_key < other._sort_key  # type: ignore[attr-defined]
        except Exception:
            return super().__lt__(other)


class FindingsView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._findings: List[Finding] = []
        self._bv = None
        self._view_frame = None
        self._persist_cb = None
        # The ScanRecord this view belongs to (set by the sidebar for result tabs).
        self.scan_record = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: gray;")
        top.addWidget(self.summary_label, 1)
        self.export_btn = QPushButton("Export triaged (fine-tuning)…")
        self.export_btn.setToolTip(
            "Export only Real Issue / False Positive findings as OpenAI chat JSONL "
            "for fine-tuning the model (the human triage becomes the corrected label).")
        self.export_btn.clicked.connect(self._on_export_finetuning)
        top.addWidget(self.export_btn)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Vertical, self)

        self.table = QTableWidget(0, len(_COLUMNS), self)
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(_TITLE_COL, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        self.table.itemActivated.connect(self._on_double_click)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)

        self.detail = QTextBrowser(self)
        self.detail.setOpenExternalLinks(False)
        self.detail.setPlaceholderText("Select a finding to see details.")

        splitter.addWidget(self.table)
        splitter.addWidget(self.detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    # Context ---------------------------------------------------------------
    def set_context(self, bv, view_frame):
        self._bv = bv
        self._view_frame = view_frame

    def set_persist_callback(self, cb):
        """Called (no args) whenever findings change and should be re-saved."""
        self._persist_cb = cb

    # Population ------------------------------------------------------------
    def clear(self):
        self._findings = []
        self.table.setRowCount(0)
        self.detail.clear()
        self._update_summary()

    def set_findings(self, findings: List[Finding]):
        self.clear()
        for f in sort_findings(findings):
            self.add_finding(f, resort=False)
        self.table.sortItems(0, Qt.AscendingOrder)
        self._update_summary()

    def add_finding(self, f: Finding, resort: bool = True):
        self._findings.append(f)
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Triage rank leads every sort key so Real Issues stay at the top and
        # False Positives at the bottom, in whatever column is sorted.
        rank = f.triage_rank
        pf_base = os.path.basename(f.prompt_file) if f.prompt_file else ""

        status_item = _SortItem(f.triage_label, (rank, 0))
        # Stash the finding index on the first column for retrieval.
        status_item.setData(Qt.UserRole, len(self._findings) - 1)
        conf_item = _SortItem(f.confidence.upper(), (rank, f.confidence_rank))
        phase_item = _SortItem(f"P{f.phase}", (rank, f.phase))
        cwe_item = _SortItem(f.cwe or "", (rank, (f.cwe or "").lower()))
        func_item = _SortItem(f.function_name or "", (rank, (f.function_name or "").lower()))
        addr_item = _SortItem(f"{f.address:#x}", (rank, f.address))
        title_item = _SortItem(f.title or "", (rank, (f.title or "").lower()))
        prompt_item = _SortItem(pf_base, (rank, pf_base.lower()))
        if f.prompt_file:
            prompt_item.setToolTip(f.prompt_file)

        items = [status_item, conf_item, phase_item, cwe_item, func_item,
                 addr_item, title_item, prompt_item]

        if f.triage_status == TRIAGE_REAL:
            bold = QFont(self.table.font())
            bold.setBold(True)
            for it in items:
                it.setForeground(QBrush(_REAL_COLOR))
            status_item.setFont(bold)
        elif f.triage_status == TRIAGE_FALSE:
            strike = QFont(self.table.font())
            strike.setStrikeOut(True)
            for it in items:
                it.setForeground(QBrush(_FP_COLOR))
                it.setFont(strike)
        elif f.triage_status in _DEBUG_ROW_COLORS:
            # Debug-only bookkeeping row (REJECTED / SKIPPED / ERROR): dim + italic so
            # it is clearly de-emphasised but distinguishable from a real finding and
            # from a human false positive.
            italic = QFont(self.table.font())
            italic.setItalic(True)
            color = _DEBUG_ROW_COLORS[f.triage_status]
            for it in items:
                it.setForeground(QBrush(color))
                it.setFont(italic)
        else:  # untriaged — colour the confidence cell by confidence
            color = _CONFIDENCE_COLORS.get(f.confidence)
            if color is not None:
                conf_item.setForeground(QBrush(color))

        for col, it in enumerate(items):
            self.table.setItem(row, col, it)

        self.table.setSortingEnabled(sorting)
        if resort and sorting:
            self.table.sortItems(self.table.horizontalHeader().sortIndicatorSection(),
                                 self.table.horizontalHeader().sortIndicatorOrder())
        self.table.resizeColumnsToContents()
        self._update_summary()

    def _update_summary(self):
        total = len(self._findings)
        real = sum(1 for f in self._findings if f.triage_status == TRIAGE_REAL)
        fp = sum(1 for f in self._findings if f.triage_status == TRIAGE_FALSE)
        rej = sum(1 for f in self._findings if f.triage_status == TRIAGE_REJECTED)
        skp = sum(1 for f in self._findings if f.triage_status == TRIAGE_SKIPPED)
        err = sum(1 for f in self._findings if f.triage_status == TRIAGE_ERROR)
        unt = total - real - fp - rej - skp - err
        parts = [f"{real} real", f"{fp} false-positive", f"{unt} untriaged"]
        if rej:
            parts.append(f"{rej} rejected")
        if skp:
            parts.append(f"{skp} skipped")
        if err:
            parts.append(f"{err} error")
        self.summary_label.setText(f"{total} finding(s): " + ", ".join(parts))
        self.export_btn.setEnabled((real + fp) > 0)

    @property
    def count(self) -> int:
        return len(self._findings)

    def findings(self) -> List[Finding]:
        return list(self._findings)

    # Interaction -----------------------------------------------------------
    def _selected_finding(self) -> Optional[Finding]:
        items = self.table.selectedItems()
        if not items:
            return None
        first = self.table.item(items[0].row(), 0)
        if first is None:
            return None
        idx = first.data(Qt.UserRole)
        if idx is None or idx < 0 or idx >= len(self._findings):
            return None
        return self._findings[idx]

    def _on_selection_changed(self):
        f = self._selected_finding()
        if f is not None:
            self.detail.setMarkdown(_finding_markdown(f))

    def _on_double_click(self, _item):
        f = self._selected_finding()
        if f is None:
            return
        navigate_to(self._view_frame, self._bv, f.address)

    def _finding_at_row(self, row: int) -> Optional[Finding]:
        first = self.table.item(row, 0)
        if first is None:
            return None
        idx = first.data(Qt.UserRole)
        if idx is None or idx < 0 or idx >= len(self._findings):
            return None
        return self._findings[idx]

    def _on_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if item is None:
            return
        finding = self._finding_at_row(item.row())
        if finding is None:
            return
        menu = QMenu(self.table)
        # One action per status other than the current one.
        status_actions = []
        for status in (TRIAGE_REAL, TRIAGE_FALSE, TRIAGE_UNTRIAGED):
            if status == finding.triage_status:
                continue
            act = menu.addAction(_MARK_LABELS[status])
            status_actions.append((act, status))
        menu.addSeparator()
        nav_act = menu.addAction("Go to code")
        chosen = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        for act, status in status_actions:
            if chosen is act:
                self._set_triage_status(finding, status)
                return
        if chosen is nav_act:
            navigate_to(self._view_frame, self._bv, finding.address)

    def _set_triage_status(self, finding: Finding, status: str):
        # Prompt for a reason on every status change (used for fine-tuning data).
        reason, ok = QInputDialog.getMultiLineText(
            self, "Triage reason",
            f"Provide reason ({_MARK_LABELS[status].replace('Mark as ', '')}):",
            finding.triage_reason)
        if not ok:
            return  # cancelled — leave the status unchanged
        finding.triage_status = status
        finding.triage_reason = reason.strip()
        self._persist()
        # Repopulate so the row moves to its status group, then reselect it.
        self.set_findings(list(self._findings))
        self._select_finding_by_id(finding.id)

    def _on_export_finetuning(self):
        triaged = [f for f in self._findings
                   if f.triage_status in (TRIAGE_REAL, TRIAGE_FALSE)]
        if not triaged:
            QMessageBox.information(
                self, "Export",
                "No triaged findings to export. Right-click findings and mark them "
                "as Real Issue or False Positive first.")
            return
        default = "vulnfanatic_training.jsonl"
        if self.scan_record is not None:
            safe = "".join(c if c.isalnum() or c in "-_." else "_"
                           for c in self.scan_record.name)
            default = f"{safe or 'scan'}_triaged.jsonl"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export triaged findings for fine-tuning", default,
            "JSON Lines (*.jsonl);;All files (*)")
        if not path:
            return
        try:
            content = _findings.to_finetuning_jsonl(self._findings, self.scan_record)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            QMessageBox.information(
                self, "Export",
                f"Exported {len(triaged)} triaged finding(s) as fine-tuning JSONL to:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export", f"Export failed: {exc}")

    def _persist(self):
        if self._persist_cb is not None:
            self._persist_cb()

    def _select_finding_by_id(self, finding_id: str):
        for row in range(self.table.rowCount()):
            f = self._finding_at_row(row)
            if f is not None and f.id == finding_id:
                self.table.selectRow(row)
                self.detail.setMarkdown(_finding_markdown(f))
                self.table.scrollToItem(self.table.item(row, 0))
                return


def _finding_markdown(f: Finding) -> str:
    parts = []
    if f.triage_status != TRIAGE_UNTRIAGED:
        parts += [f"> **Triage: {f.triage_label}**"
                  + (f" — {f.triage_reason}" if f.triage_reason else "")
                  + "  (right-click the row to change)", ""]
    parts += [
        f"## [confidence: {f.confidence.upper()}] {f.title or f.rule_id}",
        "",
        f"- **Status:** {f.triage_label}",
        f"- **Confidence:** {f.confidence}  (how sure this is a genuine security issue)",
        f"- **Severity (model estimate):** {f.severity}",
        f"- **Rule:** `{f.rule_id}` (phase {f.phase})",
        f"- **CWE:** {f.cwe or 'n/a'}",
        f"- **Function:** `{f.function_name}` @ `{f.function_addr:#x}`",
        f"- **Location:** `{f.address:#x}`  (double-click the row to navigate)",
        f"- **Context source:** {f.representation or 'n/a'}",
        f"- **Model:** {f.model or 'n/a'}",
        "",
    ]
    if f.prompt_file:
        parts += [
            "> **TEST MODE** — the LLM was not called. The full prompt and context "
            "for this candidate were saved for manual review at:",
            "",
            f"> `{f.prompt_file}`",
            "",
        ]
    if f.explanation:
        parts += ["### Explanation", "", f.explanation, ""]
    if f.tainted_input:
        parts += ["### Tainted input", "", f.tainted_input, ""]
    if f.recommendation:
        parts += ["### Recommendation", "", f.recommendation, ""]
    if f.scratchpad:
        parts += ["### Analysis scratchpad (claimed flow + snippets)", "",
                  "```", f.scratchpad, "```", ""]
    if f.validation_notes:
        parts += ["### Validation (adversarial second pass)", "",
                  "```", f.validation_notes, "```", ""]
    if f.call_paths:
        parts += ["### Call paths", "", "```", f.call_paths, "```", ""]
    if f.context_excerpt:
        parts += ["### Context excerpt sent to the model", "", "```", f.context_excerpt, "```", ""]
    return "\n".join(parts)
