import math
import struct
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core import EventAnalyzer, FileLoadError, FileLoader, WordMatcher
from src.core.distribution import analyze_windows, fit_distributions
from src.utils import parse_dates


ROOT = Path(__file__).parent


class DateTests(unittest.TestCase):
    def test_mixed_dates_preserve_index(self):
        values = pd.Series(["2024-01-02", "03/02/2024", "February 4, 2024",
                            45326, "bad", None, 123, 2024, "45326", True],
                           index=list("abcdefghij"), name="date")
        result = parse_dates(values, day_first=True)
        self.assertEqual(result.index.tolist(), values.index.tolist())
        self.assertEqual(result.name, "date")
        self.assertEqual(result.iloc[0], pd.Timestamp("2024-01-02"))
        self.assertEqual(result.iloc[1], pd.Timestamp("2024-02-03"))
        self.assertEqual(result.iloc[2], pd.Timestamp("2024-02-04"))
        self.assertEqual(result.iloc[3], pd.Timestamp("2024-02-04"))
        self.assertTrue(result.iloc[4:].isna().all())
        self.assertEqual(str(result.dtype), "datetime64[ns]")

    def test_month_first_and_timezone(self):
        result = parse_dates(pd.Series(["03/02/2024", "2024-02-01T10:00:00+02:00"]))
        self.assertEqual(result.iloc[0], pd.Timestamp("2024-03-02"))
        self.assertEqual(result.iloc[1], pd.Timestamp("2024-02-01 10:00:00"))

    def test_fractional_excel_serial_and_nonfinite(self):
        result = parse_dates(pd.Series([45292.5, np.inf, -10, np.nan]))
        self.assertEqual(result.iloc[0], pd.Timestamp("2024-01-01 12:00"))
        self.assertTrue(result.iloc[1:].isna().all())

    def test_empty_dates(self):
        self.assertEqual(str(parse_dates(pd.Series(dtype="object")).dtype),
                         "datetime64[ns]")


class MatcherTests(unittest.TestCase):
    def test_exact_word_phrase_and_punctuation_boundaries(self):
        matcher = WordMatcher(["power", "power outage", "C++"], mode="exact")
        self.assertEqual(matcher.match("Power outage; C++."), ["power", "power outage", "C++"])
        self.assertEqual(matcher.match("powerful outage"), [])
        self.assertEqual(matcher.match("lowpower outage"), [])

    def test_partial_and_deduplication(self):
        matcher = WordMatcher(["power", "POWER", "", " outage "])
        self.assertEqual(matcher.match("POWERFUL outage"), ["power", "outage"])
        self.assertEqual(matcher.match(None), [])
        self.assertEqual(matcher.match(np.nan), [])

    def test_case_sensitive_and_fuzzy(self):
        self.assertEqual(WordMatcher(["POWER"], case_sensitive=True).match("power"), [])
        self.assertEqual(WordMatcher(["outage"], mode="fuzzy").match("outag happened"),
                         ["outage"])
        self.assertEqual(WordMatcher(["POWER"], mode="fuzzy", case_sensitive=True,
                                    threshold=100).match("power"), [])

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            WordMatcher(["event"], mode="other")
        with self.assertRaises(ValueError):
            WordMatcher(["event"], threshold=101)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.paths = []

    def tearDown(self):
        for path in self.paths:
            path.unlink(missing_ok=True)

    def path(self, name):
        path = ROOT / name
        self.paths.append(path)
        return path

    def test_sample_csv_and_metadata(self):
        sheets = FileLoader.load(ROOT / "sample_events.csv")
        self.assertEqual(len(sheets), 1)
        sheet = sheets[0]
        self.assertEqual(sheet.path, str(ROOT / "sample_events.csv"))
        self.assertEqual(sheet.sheet, "CSV")
        self.assertEqual(sheet.date_candidates, ["date"])
        self.assertIn("description", sheet.text_candidates)
        self.assertEqual(len(sheet.frame), 4)

    def test_cp1252_semicolon_csv_and_blank_row_preservation(self):
        path = self.path("_test_encoding.csv")
        path.write_bytes("Date;Description\n2024-01-01;café\n\n2024-02-01;résumé\n".encode("cp1252"))
        sheet = FileLoader().load(path)[0]
        self.assertEqual(sheet.frame.loc[0, "Description"], "café")
        self.assertEqual(len(sheet.frame), 3)
        self.assertTrue(sheet.frame.iloc[1].isna().all())
        self.assertEqual(sheet.frame.index[-1], 2)

    def test_utf16_tab_and_utf8_bom(self):
        for name, encoding, separator in [("_test_utf16.csv", "utf-16", "\t"),
                                          ("_test_bom.csv", "utf-8-sig", ",")]:
            path = self.path(name)
            path.write_text("Date{0}Description\n2024-01-01{0}event\n".format(separator),
                            encoding=encoding)
            self.assertEqual(FileLoader.load(path)[0].frame.columns.tolist(),
                             ["Date", "Description"])

    def test_xlsx_all_nonempty_sheets_and_serial_candidates(self):
        path = self.path("_test_workbook.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame({"Date": [45292, 45293], "ID": [45292, 45293],
                          "Description": ["first", "second"]}).to_excel(
                              writer, sheet_name="Events", index=False)
            pd.DataFrame({"When": ["2024-03-01"], "Details": ["third"]}).to_excel(
                writer, sheet_name="Other", index=False)
            pd.DataFrame().to_excel(writer, sheet_name="Empty", index=False)
        sheets = FileLoader.load(path)
        self.assertEqual([sheet.sheet for sheet in sheets], ["Events", "Other"])
        self.assertEqual(sheets[0].date_candidates, ["Date"])
        self.assertEqual(sheets[1].date_candidates, ["When"])
        self.assertEqual(sheets[0].frame.loc[1, "Description"], "second")

    def test_legacy_xls(self):
        # A small raw BIFF8 workbook avoids a test-only Excel writer dependency.
        def record(kind, payload):
            return struct.pack("<HH", kind, len(payload)) + payload

        def bof(kind):
            return record(0x0809, struct.pack("<HHHHII", 0x0600, kind, 0x0DBB,
                                            0x07CC, 0x41, 0x06))

        def label(row, column, text):
            return record(0x0204, struct.pack("<HHHHB", row, column, 0, len(text), 0)
                          + text.encode("ascii"))

        eof = record(0x000A, b"")
        name = b"Events"
        boundsheet = record(0x0085, struct.pack("<IBBBB", 0, 0, 0, len(name), 0) + name)
        globals_prefix = bof(0x0005) + record(0x0042, struct.pack("<H", 1252))
        offset = len(globals_prefix + boundsheet + eof)
        boundsheet = record(0x0085, struct.pack("<IBBBB", offset, 0, 0, len(name), 0) + name)
        sheet = (bof(0x0010) + record(0x0200, struct.pack("<IIHHH", 0, 2, 0, 2, 0))
                 + label(0, 0, "Date") + label(0, 1, "Description")
                 + record(0x0203, struct.pack("<HHHd", 1, 0, 0, 45292))
                 + label(1, 1, "Legacy outage") + eof)
        path = self.path("_test_legacy.xls")
        path.write_bytes(globals_prefix + boundsheet + eof + sheet)
        loaded = FileLoader.load(path)
        self.assertEqual(loaded[0].sheet, "Events")
        self.assertEqual(loaded[0].frame.loc[0, "Description"], "Legacy outage")

    def test_safe_errors_and_empty_data(self):
        with self.assertRaisesRegex(FileLoadError, "not found"):
            FileLoader.load(ROOT / "_missing.csv")
        with self.assertRaisesRegex(FileLoadError, "Unsupported"):
            FileLoader.load("events.pdf")
        empty = self.path("_test_empty.csv")
        empty.write_text("Date,Description\n")
        self.assertEqual(FileLoader.load(empty), [])
        broken = self.path("_test_broken.xlsx")
        broken.write_bytes(b"not an Excel file")
        with self.assertRaises(FileLoadError):
            FileLoader.load(broken)
        with patch("src.core.file_loader.pd.read_excel", side_effect=ImportError):
            with self.assertRaisesRegex(FileLoadError, "dependency"):
                FileLoader.load(broken)


def records(dates, descriptions):
    return pd.DataFrame({"date": parse_dates(pd.Series(dates)),
                         "description": descriptions, "source": "events.csv",
                         "sheet": "CSV", "row": range(2, len(dates) + 2)})


class AnalyzerTests(unittest.TestCase):
    def test_aggregation_overlap_provenance_and_first_occurrence(self):
        frame = records(["2024-01-03", "2024-03-02", "invalid", "2024-04-02"],
                        ["Power outage", "outage again", "outage", "unrelated"])
        original = frame.copy(deep=True)
        result = EventAnalyzer.analyze(frame, WordMatcher(["outage", "power"]))
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(result.invalid_dates, 1)
        self.assertEqual(result.matched["row"].tolist(), [2, 3])
        self.assertEqual(result.matched.iloc[0]["events"], ["outage", "power"])
        self.assertEqual(result.monthly.tolist(), [1, 0, 1, 0])
        self.assertEqual(result.quarterly.tolist(), [2, 0])
        self.assertEqual(result.top_events["count"].tolist(), [2, 1])
        self.assertEqual(result.top_events.iloc[0]["first_occurrence"],
                         pd.Timestamp("2024-01-03"))
        self.assertEqual(result.top_events.iloc[0]["description"], "Power outage")
        self.assertEqual(len(result.windows), 2)
        self.assertEqual(result.windows[0].daily_counts.sum(), 2)

    def test_empty_no_match_and_all_invalid(self):
        matcher = WordMatcher(["outage"])
        result = EventAnalyzer.analyze(records(["2024-01-01", "2024-04-01"],
                                             ["nothing", None]), matcher)
        self.assertTrue(result.matched.empty)
        self.assertEqual(result.monthly.tolist(), [0, 0, 0, 0])
        self.assertTrue(result.top_events.empty)
        self.assertEqual(result.top_events.columns.tolist(),
                         ["event", "count", "first_occurrence", "description"])
        self.assertEqual(len(result.windows), 2)
        self.assertEqual(result.windows[1].shift_pvalue, 1.0)
        for frame in (records(["bad"], ["outage"]), records([], [])):
            result = EventAnalyzer.analyze(frame, matcher)
            self.assertTrue(result.monthly.empty)
            self.assertTrue(result.quarterly.empty)
            self.assertEqual(result.windows, [])

    def test_first_occurrence_not_restricted_to_last_window(self):
        result = EventAnalyzer.analyze(
            records(["2024-01-01", "2024-06-01"], ["outage first", "outage last"]),
            WordMatcher(["outage"]))
        self.assertEqual(result.top_events.iloc[0]["first_occurrence"],
                         pd.Timestamp("2024-01-01"))

    def test_record_schema_error(self):
        with self.assertRaises(ValueError):
            EventAnalyzer.analyze(pd.DataFrame(), WordMatcher(["outage"]))

    def test_top_events_limited_to_ten(self):
        words = ["event{:02d}".format(index) for index in range(12)]
        result = EventAnalyzer.analyze(
            records(["2024-01-01"], [" ".join(words)]), WordMatcher(words, mode="exact"))
        self.assertEqual(len(result.top_events), 10)
        self.assertEqual(result.top_events["event"].tolist(), words[:10])
        self.assertEqual(len(result.matched.iloc[0]["events"]), 12)
        self.assertEqual(result.monthly.sum(), 1)


class DistributionTests(unittest.TestCase):
    def test_calendar_windows_zero_days_and_fixed_baseline(self):
        dates = parse_dates(pd.Series(["2024-01-31", "2024-03-03", "2024-05-01"]))
        windows = analyze_windows(dates)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].start, pd.Timestamp("2024-01-01"))
        self.assertEqual(windows[0].end, pd.Timestamp("2024-03-31"))
        self.assertEqual(len(windows[0].daily_counts), 91)
        self.assertEqual(windows[0].daily_counts.sum(), 2)
        self.assertIsInstance(windows[0].counts, np.ndarray)
        np.testing.assert_array_equal(windows[0].counts,
                                      windows[0].daily_counts.to_numpy())
        self.assertIsNone(windows[0].shift_pvalue)
        self.assertIsNone(windows[0].baseline_pvalue)
        self.assertIsNotNone(windows[2].baseline_pvalue)
        self.assertIn("Overlapping", windows[0].limitation)
        for window in windows[1:]:
            self.assertGreaterEqual(window.shift_pvalue, window.raw_shift_pvalue)
            self.assertGreaterEqual(window.baseline_pvalue, window.raw_baseline_pvalue)
            self.assertTrue(0 <= window.shift_pvalue <= 1)

    def test_short_extent_and_empty(self):
        self.assertEqual(analyze_windows(parse_dates(pd.Series(["2024-01-01"]))), [])
        self.assertEqual(analyze_windows(pd.Series(dtype="datetime64[ns]")), [])

    def test_constant_and_unsupported_fits(self):
        for value in (0, 3):
            fits = fit_distributions([value] * 10)
            self.assertEqual([fit.name for fit in fits], ["Poisson"])
            self.assertTrue(math.isfinite(fits[0].aic))
        self.assertEqual(fit_distributions([]), [])
        self.assertEqual(fit_distributions([np.nan]), [])
        self.assertEqual(fit_distributions([-1, 0]), [])
        self.assertEqual(fit_distributions([1.5, 1.5]), [])

    def test_likelihood_aic_and_all_four_models(self):
        sample = np.array([1, 2, 3, 4, 6, 8])
        fits = fit_distributions(sample)
        by_name = {fit.name: fit for fit in fits}
        self.assertEqual(set(by_name), {"Normal", "Poisson", "Exponential", "Weibull"})
        self.assertEqual(by_name["Poisson"].parameters, (sample.mean(),))
        self.assertEqual(len(by_name["Normal"].parameters), 2)
        self.assertEqual(len(by_name["Exponential"].parameters), 2)
        self.assertEqual(len(by_name["Weibull"].parameters), 3)
        from scipy import stats
        expected = 2 - 2 * stats.poisson.logpmf(sample, sample.mean()).sum()
        self.assertAlmostEqual(by_name["Poisson"].aic, expected)
        self.assertEqual([fit.aic for fit in fits], sorted(fit.aic for fit in fits))
        self.assertNotIn("Weibull", {fit.name for fit in fit_distributions([0, 1, 2])})

    def test_significant_shift_with_tied_counts(self):
        # A quiet Jan–Mar baseline followed by two dense months.
        dates = pd.Series(pd.date_range("2024-04-01", "2024-05-31").repeat(20))
        windows = analyze_windows(dates, start="2024-01-01", end="2024-05-31")
        self.assertEqual(len(windows), 3)
        self.assertTrue(windows[-1].shifted)
        self.assertLess(windows[-1].baseline_pvalue, 0.05)
        self.assertTrue(all(math.isfinite(fit.aic) for fit in windows[-1].fits))


if __name__ == "__main__":
    unittest.main()
