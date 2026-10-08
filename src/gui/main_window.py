"""Responsive desktop workflow for loading, mapping, and analyzing local files."""

from pathlib import Path
import traceback

import pandas as pd
from PyQt6.QtCore import QObject, QThread, QUrl, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from src.core.analyzer import EventAnalyzer
from src.core.file_loader import FileLoader
from src.core.matcher import WordMatcher
from src.gui.charts import ChartWidget
from src.gui.dialogs import ColumnMappingDialog
from src.utils.dates import parse_dates


class Worker(QObject):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    @pyqtSlot()
    def run(self):
        try:
            self.succeeded.emit(self.operation())
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc) or type(exc).__name__)
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.records = None
        self.result = None
        self.thread = None
        self.worker = None
        self.setWindowTitle("Event Frequency Analyzer")
        self.resize(1220, 820)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("Event Frequency Analyzer")
        title.setStyleSheet("font-size: 25px; font-weight: 600; color: #285c91;")
        layout.addWidget(title)
        layout.addWidget(QLabel("Discover trends in local spreadsheets — no data is uploaded."))
        splitter = QSplitter()
        layout.addWidget(splitter)
        controls = QWidget()
        sidebar = QVBoxLayout(controls)
        self.load_button = QPushButton("Add Excel / CSV files…")
        self.load_button.clicked.connect(self.load_files)
        sidebar.addWidget(self.load_button)
        self.file_label = QLabel("No files loaded")
        self.file_label.setWordWrap(True)
        sidebar.addWidget(self.file_label)
        self.clear_button = QPushButton("Clear sources")
        self.clear_button.clicked.connect(self.clear_sources)
        sidebar.addWidget(self.clear_button)
        sidebar.addWidget(QLabel("Wordset — one word or phrase per line"))
        self.words = QTextEdit()
        self.words.setPlaceholderText("error\nlogin failed\ntimeout")
        self.words.setMinimumHeight(160)
        sidebar.addWidget(self.words)
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItems(["Partial", "Exact", "Fuzzy"])
        form.addRow("Matching", self.mode)
        self.threshold = QSpinBox()
        self.threshold.setRange(0, 100)
        self.threshold.setValue(80)
        self.threshold.setSuffix("%")
        self.threshold.setToolTip("Minimum fuzzy similarity. Higher values are stricter.")
        self.threshold.setEnabled(False)
        self.mode.currentTextChanged.connect(
            lambda mode: self.threshold.setEnabled(mode == "Fuzzy")
        )
        form.addRow("Fuzzy similarity", self.threshold)
        self.case_sensitive = QCheckBox("Case-sensitive matching")
        form.addRow(self.case_sensitive)
        sidebar.addLayout(form)
        self.analyze_button = QPushButton("Analyze events")
        self.analyze_button.setStyleSheet(
            "QPushButton { background: #285c91; color: white; padding: 10px; }"
        )
        self.analyze_button.clicked.connect(self.analyze)
        sidebar.addWidget(self.analyze_button)
        self.export_button = QPushButton("Export matched rows…")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        sidebar.addWidget(self.export_button)
        note = QLabel(
            "Each matching row counts once in charts. A row may contribute to multiple "
            "wordset events. Double-click a matched row to inspect its original entry."
        )
        note.setWordWrap(True)
        sidebar.addWidget(note)
        sidebar.addStretch()
        splitter.addWidget(controls)
        self.tabs = QTabWidget()
        self.monthly = ChartWidget()
        self.quarterly = ChartWidget()
        self.distribution = ChartWidget()
        self.fit_chart = ChartWidget()
        self.top_table = self.make_table(
            ["Event", "Count", "First occurrence", "Example description"]
        )
        self.window_table = self.make_table(
            ["Start", "End", "Best fit", "Previous p", "Baseline p", "Shift"]
        )
        self.window_table.itemSelectionChanged.connect(self.show_window_fit)
        distribution_page = QWidget()
        distribution_layout = QVBoxLayout(distribution_page)
        distribution_layout.addWidget(self.distribution)
        distribution_layout.addWidget(self.window_table)
        distribution_layout.addWidget(QLabel(
            "Select a window to inspect fitted laws. AIC is a heuristic ranking, not "
            "proof of fit. Overlapping windows are dependent; shift flags are exploratory."
        ))
        self.rows_table = self.make_table(["Date", "Description", "Events", "File", "Sheet", "Row"])
        self.rows_table.cellDoubleClicked.connect(self.open_entry)
        for widget, name in (
            (self.monthly, "Monthly"), (self.quarterly, "Trimester"),
            (self.top_table, "Top 10 events"), (distribution_page, "Window comparisons"),
            (self.fit_chart, "Selected window fit"), (self.rows_table, "Matched entries"),
        ):
            self.tabs.addTab(widget, name)
        splitter.addWidget(self.tabs)
        splitter.setSizes([300, 920])
        self.summary = QLabel("Choose files, confirm column mappings, then enter your wordset.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.statusBar().showMessage("Ready")

    @staticmethod
    def make_table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        return table

    def set_busy(self, busy):
        for widget in (
            self.load_button, self.clear_button, self.analyze_button,
            self.words, self.mode, self.case_sensitive,
        ):
            widget.setEnabled(not busy)
        self.threshold.setEnabled(not busy and self.mode.currentText() == "Fuzzy")
        self.export_button.setEnabled(not busy and self.result is not None)
        self.statusBar().showMessage("Working…" if busy else "Ready")

    def start_job(self, operation, on_success):
        if self.thread is not None:
            return
        self.set_busy(True)
        self.thread = QThread(self)
        self.worker = Worker(operation)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.succeeded.connect(on_success)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.job_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @pyqtSlot()
    def job_finished(self):
        self.thread = None
        self.worker = None
        self.set_busy(False)

    @pyqtSlot(str)
    def show_error(self, message):
        QMessageBox.warning(self, "Unable to complete operation", message)

    def load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select event files", "", "Event files (*.csv *.xlsx *.xls)"
        )
        if not paths:
            return

        def load():
            sheets, errors = [], []
            for path in paths:
                try:
                    sheets.extend(FileLoader.load(path))
                except Exception as exc:
                    errors.append(f"{Path(path).name}: {exc}")
            return sheets, errors

        self.start_job(load, self.map_sources)

    @pyqtSlot(object)
    def map_sources(self, payload):
        sheets, errors = payload
        frames = []
        existing = set()
        if self.records is not None:
            existing = set(zip(self.records["source"], self.records["sheet"]))
        for sheet in sheets:
            identity = (str(sheet.path), str(sheet.sheet))
            if identity in existing:
                continue
            dialog = ColumnMappingDialog(sheet, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                continue
            date_column, description_column, day_first = dialog.mapping()
            frame = pd.DataFrame({
                "date": parse_dates(sheet.frame[date_column], day_first=day_first),
                "description": sheet.frame[description_column].fillna("").astype(str),
                "source": str(sheet.path),
                "sheet": str(sheet.sheet),
                "row": range(2, len(sheet.frame) + 2),
            })
            frames.append(frame)
            existing.add(identity)
        if frames:
            if self.records is not None:
                frames.insert(0, self.records)
            self.records = pd.concat(frames, ignore_index=True)
            self.invalidate_results()
            self.file_label.setText(
                f"{len(existing)} source sheets • {len(self.records):,} rows"
            )
            self.summary.setText("Sources loaded. Enter a wordset and select Analyze events.")
        if errors:
            self.show_error("\n".join(errors))

    def invalidate_results(self):
        self.result = None
        self.export_button.setEnabled(False)
        for chart in (self.monthly, self.quarterly, self.distribution, self.fit_chart):
            chart.clear("Run an analysis to see results.")
        for table in (self.top_table, self.window_table, self.rows_table):
            table.setRowCount(0)

    def clear_sources(self):
        self.records = None
        self.invalidate_results()
        self.file_label.setText("No files loaded")
        self.summary.setText("Choose files to begin.")

    def analyze(self):
        words = [line.strip() for line in self.words.toPlainText().splitlines() if line.strip()]
        if self.records is None or not words:
            self.show_error("Load at least one source and enter at least one word or phrase.")
            return
        matcher = WordMatcher(
            words, mode=self.mode.currentText().lower(),
            threshold=self.threshold.value(), case_sensitive=self.case_sensitive.isChecked(),
        )
        records = self.records.copy()
        self.summary.setText("Analyzing the current sources and wordset…")
        self.start_job(lambda: EventAnalyzer.analyze(records, matcher), self.display_result)

    @staticmethod
    def populate(table, rows):
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                table.setItem(row_index, column, item)
        table.resizeColumnsToContents()

    @pyqtSlot(object)
    def display_result(self, result):
        self.result = result
        self.monthly.plot_counts(result.monthly, "Monthly event frequency")
        self.quarterly.plot_counts(result.quarterly, "Trimester (calendar quarter) event frequency")
        self.distribution.plot_windows(result.windows)
        self.populate(self.top_table, [
            [r.event, r.count, str(r.first_occurrence)[:10], r.description]
            for r in result.top_events.itertuples()
        ])
        self.populate(self.window_table, [
            [str(w.start)[:10], str(w.end)[:10], w.best_fit,
             self.pvalue(w.shift_pvalue), self.pvalue(w.baseline_pvalue),
             "Yes" if w.shifted else "No"]
            for w in result.windows
        ])
        for index, window in enumerate(result.windows):
            details = "\n".join(
                f"{fit.name}: AIC={fit.aic:.2f}, parameters={fit.parameters}"
                for fit in window.fits
            )
            self.window_table.item(index, 2).setToolTip(details or "No reliable fit")
        # Keep rendering bounded; all matching rows remain available in CSV export.
        visible = result.matched.head(5000)
        self.populate(self.rows_table, [
            [str(r.date)[:10], r.description, ", ".join(r.events), r.source, r.sheet, r.row]
            for r in visible.itertuples()
        ])
        self.summary.setText(
            f"{len(result.matched):,} matching rows • {result.invalid_dates:,} rows with "
            "missing/unparseable dates excluded. Entry preview limited to 5,000 rows; export "
            "contains all matches."
        )
        if result.windows:
            self.window_table.selectRow(0)
        else:
            self.fit_chart.clear("No complete three-month window.")
        self.tabs.setCurrentIndex(0)

    @staticmethod
    def pvalue(value):
        return "—" if value is None else f"{value:.4g}"

    def show_window_fit(self):
        row = self.window_table.currentRow()
        if self.result is not None and 0 <= row < len(self.result.windows):
            self.fit_chart.plot_fit(self.result.windows[row])

    def open_entry(self, row, column):
        if self.result is None or row >= len(self.result.matched):
            return
        record = self.result.matched.iloc[row]
        QMessageBox.information(
            self, "Source entry",
            f"File: {record['source']}\nSheet: {record['sheet'] or 'CSV'}\n"
            f"Row: {record['row']}\nDate: {record['date']}\n\n{record['description']}",
        )
        answer = QMessageBox.question(
            self, "Open original file?",
            "Open this local file in its default application? "
            "Use the sheet and row above to locate the entry.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(record["source"]).resolve()))):
                self.show_error("No application could open this file.")

    def export_results(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export matches", "matches.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            exported = self.result.matched.copy()
            exported["events"] = exported["events"].apply("; ".join)
            # Prevent spreadsheet formula execution when exported text is opened in Excel.
            for column in exported.select_dtypes(include=["object", "string"]).columns:
                exported[column] = exported[column].map(
                    lambda value: "'" + value if isinstance(value, str)
                    and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n"))
                    else value
                )
            exported.to_csv(path, index=False, encoding="utf-8-sig")
            self.statusBar().showMessage(f"Exported {len(exported):,} rows")
        except (OSError, ValueError) as exc:
            self.show_error(str(exc))

    def closeEvent(self, event):
        if self.thread is not None:
            self.statusBar().showMessage("Please wait for the current operation to finish before closing.")
            event.ignore()
        else:
            event.accept()
