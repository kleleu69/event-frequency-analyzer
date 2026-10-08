"""Offscreen regression tests for the desktop's own presentation workflow."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication

from src.gui.dialogs import ColumnMappingDialog
from src.gui.main_window import MainWindow


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    @staticmethod
    def result():
        window = SimpleNamespace(
            start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-03-31"),
            best_fit="Poisson", shift_pvalue=None, baseline_pvalue=None,
            shifted=False, counts=np.array([0, 1, 2, 0, 1]),
            fits=[SimpleNamespace(name="Poisson", aic=12.5, parameters=(0.8,))],
        )
        return SimpleNamespace(
            matched=pd.DataFrame({
                "date": [pd.Timestamp("2024-01-02")], "description": ["=danger"],
                "events": [["error"]], "source": ["/tmp/events.csv"],
                "sheet": [""], "row": [2],
            }),
            monthly=pd.Series([1], index=pd.period_range("2024-01", periods=1, freq="M")),
            quarterly=pd.Series([1], index=pd.period_range("2024Q1", periods=1, freq="Q")),
            top_events=pd.DataFrame({
                "event": ["error"], "count": [1],
                "first_occurrence": [pd.Timestamp("2024-01-02")],
                "description": ["=danger"],
            }),
            windows=[window], invalid_dates=1,
        )

    def test_results_display_and_clear(self):
        self.window.display_result(self.result())
        self.assertEqual(self.window.rows_table.rowCount(), 1)
        self.assertEqual(self.window.top_table.item(0, 0).text(), "error")
        self.assertEqual(self.window.window_table.item(0, 2).text(), "Poisson")
        self.assertIn("1 rows with", self.window.summary.text())
        self.window.clear_sources()
        self.assertEqual(self.window.rows_table.rowCount(), 0)
        self.assertIsNone(self.window.result)

    def test_empty_results_clear_all_charts(self):
        result = self.result()
        result.matched = result.matched.iloc[:0]
        result.top_events = result.top_events.iloc[:0]
        result.monthly = result.monthly.iloc[:0]
        result.quarterly = result.quarterly.iloc[:0]
        result.windows = []
        self.window.display_result(result)
        self.assertEqual(self.window.rows_table.rowCount(), 0)
        self.assertEqual(self.window.window_table.rowCount(), 0)

    def test_reanalysis_refreshes_selected_fit(self):
        self.window.display_result(self.result())
        next_result = self.result()
        next_result.windows[0].best_fit = "Updated fit"
        with patch.object(self.window.fit_chart, "plot_fit") as plot:
            self.window.display_result(next_result)
        plot.assert_called_with(next_result.windows[0])

    def test_column_suggestions_are_editable(self):
        sheet = SimpleNamespace(
            path="/tmp/example.csv", sheet="",
            frame=pd.DataFrame({"details": ["error"], "reported": ["2024-01-01"]}),
            date_candidates=["reported"], text_candidates=["details"],
        )
        dialog = ColumnMappingDialog(sheet)
        self.assertEqual(dialog.mapping(), ("reported", "details", False))
        dialog.day_first.setChecked(True)
        self.assertTrue(dialog.mapping()[2])
        dialog.close()

    def test_export_neutralizes_formulas_and_keeps_all_rows(self):
        self.window.result = self.result()
        with TemporaryDirectory() as folder:
            destination = str(Path(folder) / "matches.csv")
            with patch("src.gui.main_window.QFileDialog.getSaveFileName", return_value=(destination, "")):
                self.window.export_results()
            exported = pd.read_csv(destination)
            self.assertEqual(exported.loc[0, "description"], "'=danger")
            self.assertEqual(exported.loc[0, "events"], "error")

    def test_worker_completes_and_restores_controls(self):
        values = []
        loop = QEventLoop()
        self.window.start_job(lambda: 42, values.append)
        self.window.thread.finished.connect(loop.quit)
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        self.assertEqual(values, [42])
        self.assertIsNone(self.window.thread)
        self.assertTrue(self.window.analyze_button.isEnabled())

    def test_worker_failure_restores_controls(self):
        messages = []
        loop = QEventLoop()

        def fail():
            raise ValueError("Invalid input")

        with patch.object(self.window, "show_error", side_effect=messages.append):
            self.window.start_job(fail, lambda value: None)
            self.window.thread.finished.connect(loop.quit)
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
        self.assertEqual(messages, ["Invalid input"])
        self.assertIsNone(self.window.thread)
        self.assertTrue(self.window.load_button.isEnabled())

    def test_fuzzy_threshold_is_only_enabled_for_fuzzy_mode(self):
        self.assertFalse(self.window.threshold.isEnabled())
        self.window.mode.setCurrentText("Fuzzy")
        self.assertTrue(self.window.threshold.isEnabled())
        self.window.set_busy(True)
        self.assertFalse(self.window.threshold.isEnabled())
        self.assertFalse(self.window.words.isEnabled())
        self.window.set_busy(False)
        self.assertTrue(self.window.threshold.isEnabled())


if __name__ == "__main__":
    unittest.main()
