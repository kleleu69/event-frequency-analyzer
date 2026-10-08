"""Explicit column mapping avoids silently guessing ambiguous source schemas."""

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout,
)


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
