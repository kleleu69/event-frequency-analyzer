"""Explicit column mapping avoids silently guessing ambiguous source schemas."""

import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
)

from src.core.construction import load_vocabulary, validate_vocabulary


class ColumnMappingDialog(QDialog):
    def __init__(self, sheet, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map source columns")
        self.resize(760, 440)
        layout = QVBoxLayout(self)
        label = QLabel(f"{sheet.path}\nSheet: {sheet.sheet or 'CSV'}")
        label.setWordWrap(True)
        layout.addWidget(label)
        form = QFormLayout()
        self.date_column = QComboBox()
        self.description_column = QComboBox()
        columns = list(sheet.frame.columns)
        for combo in (self.date_column, self.description_column):
            for column in columns:
                combo.addItem(str(column), column)
        if sheet.date_candidates:
            self.date_column.setCurrentIndex(columns.index(sheet.date_candidates[0]))
        if sheet.text_candidates:
            self.description_column.setCurrentIndex(columns.index(sheet.text_candidates[0]))
        self.day_first = QCheckBox("Interpret ambiguous dates as day/month/year")
        form.addRow("Date column", self.date_column)
        form.addRow("Description column", self.description_column)
        form.addRow("", self.day_first)
        layout.addLayout(form)
        layout.addWidget(QLabel("Preview — verify the suggested columns before continuing."))
        preview = sheet.frame.head(8)
        table = QTableWidget(len(preview), len(columns))
        table.setHorizontalHeaderLabels([str(c) for c in columns])
        for row in range(len(preview)):
            for col in range(len(columns)):
                table.setItem(row, col, QTableWidgetItem(str(preview.iloc[row, col])))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def mapping(self):
        return (
            self.date_column.currentData(),
            self.description_column.currentData(),
            self.day_first.isChecked(),
        )


class VocabularyDialog(QDialog):
    """Edit and explicitly save a local dictionary without altering bundled data."""

    def __init__(self, vocabulary, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Construction dictionary")
        self.resize(850, 650)
        self.vocabulary = vocabulary
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Edit aliases under disciplines, activities, issues, and statuses. "
            "Canonical labels group synonyms into events. Import a plant-specific JSON "
            "dictionary, or save a copy for reuse. Apply changes affects this session only."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.editor = QTextEdit()
        self.editor.setAcceptRichText(False)
        self.editor.setPlainText(json.dumps(vocabulary, ensure_ascii=False, indent=2))
        layout.addWidget(self.editor)
        actions = QHBoxLayout()
        for title, callback in (
            ("Import JSON…", self.import_dictionary),
            ("Save JSON copy…", self.save_dictionary),
            ("Restore starter dictionary", self.restore_dictionary),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply changes")
        buttons.accepted.connect(self.apply_changes)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def validated(self):
        return validate_vocabulary(json.loads(self.editor.toPlainText()))

    def report_error(self, exc):
        QMessageBox.warning(self, "Invalid construction dictionary", str(exc))

    def apply_changes(self):
        try:
            self.vocabulary = self.validated()
        except (ValueError, TypeError) as exc:
            self.report_error(exc)
            return
        self.accept()

    def import_dictionary(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import dictionary", "", "JSON (*.json)")
        if not path:
            return
        try:
            vocabulary = load_vocabulary(path)
            self.editor.setPlainText(json.dumps(vocabulary, ensure_ascii=False, indent=2))
        except (OSError, ValueError, TypeError) as exc:
            self.report_error(exc)

    def save_dictionary(self):
        try:
            vocabulary = self.validated()
        except (ValueError, TypeError) as exc:
            self.report_error(exc)
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save dictionary copy", "construction_dictionary.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            self.report_error(exc)

    def restore_dictionary(self):
        try:
            self.editor.setPlainText(json.dumps(load_vocabulary(), ensure_ascii=False, indent=2))
        except (OSError, ValueError) as exc:
            self.report_error(exc)
