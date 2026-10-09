import copy
import json
import unittest
from pathlib import Path

import pandas as pd

from src.core.analyzer import AnalysisResult
from src.core.construction import (ConstructionAnalyzer, ConstructionRecognizer,
                                   load_vocabulary, validate_vocabulary)


class VocabularyTests(unittest.TestCase):
    def test_default_vocabulary_and_module_relative_path(self):
        vocabulary = load_vocabulary()
        self.assertEqual(set(vocabulary), {
            "version", "disciplines", "activities", "issues", "statuses"})
        self.assertIn("electrical", vocabulary["disciplines"])
        self.assertIs(validate_vocabulary(vocabulary), vocabulary)

    def test_custom_file_edits(self):
        path = Path(__file__).with_name("_construction_vocabulary_test.json")
        vocabulary = load_vocabulary()
        vocabulary["issues"]["delay"].append("waiting for supplier")
        try:
            path.write_text(json.dumps(vocabulary), encoding="utf-8")
            recognizer = ConstructionRecognizer(load_vocabulary(path))
            result = recognizer.classify("Electrical waiting for supplier")
            self.assertEqual(result["issue"], "delay")
            self.assertEqual(result["disposition"], "recognized")
        finally:
            path.unlink(missing_ok=True)

    def test_bad_schema(self):
        for change in (None, [], {}, {"version": "1"}, {"version": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ConstructionRecognizer(change if change is not None else [])
        mutations = [
            ("issues", {"delay": "delay"}),
            ("issues", {"delay": []}),
            ("issues", {"delay": [""]}),
            ("issues", {"delay": ["---"]}),
            ("issues", {"delay": [1]}),
            ("issues", {"": ["delay"]}),
            ("issues", {"delay": ["défaut"], "defect": ["defaut"]}),
            ("issues", {"delay": ["defect"], "defect": ["broken"]}),
        ]
        for axis, replacement in mutations:
            vocabulary = load_vocabulary()
            vocabulary[axis] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                ConstructionRecognizer(vocabulary)

    def test_read_error_and_json_error_are_value_errors(self):
        path = Path(__file__).with_name("_construction_bad_test.json")
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            load_vocabulary(path)
        try:
            path.write_text("{bad json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cannot read"):
                load_vocabulary(path)
        finally:
            path.unlink(missing_ok=True)

    def test_duplicate_same_category_alias_is_legal_and_exact(self):
        vocabulary = load_vocabulary()
        vocabulary["issues"]["delay"] += ["DELAY", "delay"]
        result = ConstructionRecognizer(vocabulary).classify("Electrical delay delays")
        self.assertEqual(result["events"], ["delay"])

    def test_threshold_validation(self):
        for value in (-1, 101, "85", True, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ConstructionRecognizer(threshold=value)


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = ConstructionRecognizer()

    def classify(self, text):
        result = self.recognizer.classify(text)
        self.assertIsInstance(result["confidence"], float)
        self.assertTrue(0 <= result["confidence"] <= 1)
        self.assertEqual(set(result), {"discipline", "activity", "issue", "status",
                                       "confidence", "review_reason", "evidence",
                                       "disposition", "events"})
        return result

    def test_disciplines_activities_and_contractor_codes(self):
        examples = [
            ("GC excavation completed", "civil works", "excavation"),
            ("STR steel erection completed", "structural steel", "erection"),
            ("MECH piping welding finished", "mechanical piping", "welding"),
            ("ELEC cable pulling done", "electrical", "cable installation"),
            ("INST loop check completed", "instrumentation", "loop checking"),
            ("COMM handover completed", "commissioning", "handover"),
        ]
        for text, discipline, activity in examples:
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["disposition"], "recognized")
                self.assertEqual(result["discipline"], discipline)
                self.assertEqual(result["activity"], activity)
                self.assertEqual(result["status"], "completed")

    def test_french_accents_and_punctuation_codes(self):
        examples = [
            ("Charpente métallique : soudage terminé", "structural steel", "welding"),
            ("ÉLECTRIQUE — pose de câbles terminée", "electrical", "cable installation"),
            ("Génie civil, terrassement achevé", "civil works", "excavation"),
            ("I&C inspection", "instrumentation", "inspection"),
            ("E&I cable_installation completed", "electrical", "cable installation"),
        ]
        for text, discipline, activity in examples:
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["discipline"], discipline)
                self.assertEqual(result["activity"], activity)
                self.assertEqual(result["disposition"], "recognized")

    def test_issues_and_faults(self):
        examples = [
            ("Civil excavation delayed", "delay"),
            ("Mechanical missing materials", "missing materials"),
            ("Piping welding defect", "defect"),
            ("Structural steel rework", "rework"),
            ("ELEC NCR", "nonconformity"),
            ("Civil no access", "access constraint"),
            ("Short circuit repaired", "short circuit"),
            ("Earth fault", "earth fault"),
            ("Breaker tripped", "breaker trip"),
            ("Insulation failure", "insulation failure"),
            ("Panne électrique", "power outage"),
            ("Défaut électrique", "electrical fault"),
        ]
        for text, issue in examples:
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["disposition"], "recognized")
                self.assertEqual(result["issue"], issue)
                self.assertIn(issue, result["events"])

    def test_inspection_and_maintenance_do_not_imply_fault(self):
        for text in ("Cable inspection completed", "Electrical maintenance ongoing"):
            result = self.classify(text)
            self.assertEqual(result["issue"], "")
            self.assertNotIn("electrical fault", result["events"])
            self.assertEqual(result["disposition"], "recognized")
        result = self.classify("Cable fault found")
        self.assertEqual(result["issue"], "defect")
        result = self.classify("Cable installation incomplete")
        self.assertEqual(result["status"], "pending")
        self.assertNotEqual(result["status"], "completed")

    def test_fuzzy_windows_require_review(self):
        result = self.classify("Electrical cable installaton completed")
        self.assertEqual(result["activity"], "cable installation")
        self.assertEqual(result["disposition"], "review")
        self.assertIn("Fuzzy", result["review_reason"])
        self.assertIn("fuzzy", result["evidence"])
        self.assertLess(result["confidence"], 0.96)
        strict = ConstructionRecognizer(threshold=100).classify(
            "Electrical cable installaton completed")
        self.assertEqual(strict["activity"], "")

    def test_no_partial_substrings_or_fuzzy_acronyms(self):
        for text in ("electrically cablelessness testingly", "NCRR", "GCC",
                     "reinspectional", "ELX", "PIPP"):
            with self.subTest(text=text):
                self.assertEqual(self.classify(text)["disposition"], "unfamiliar")

    def test_unknown_and_generic_context_retained(self):
        self.assertEqual(self.classify("The weather is sunny")["disposition"], "unfamiliar")
        for text in ("Testing completed", "Electrical", "Welding finished"):
            self.assertEqual(self.classify(text)["disposition"], "review")
        self.assertEqual(self.classify(
            "Electrical and instrumentation inspection completed")["disposition"], "review")
        for text in (None, "", "   ", float("nan"), pd.NA, "!!!"):
            self.assertEqual(self.classify(text)["review_reason"], "Blank description.")

    def test_completed_delayed_and_negated_are_distinct(self):
        completed = self.classify("Cable installation completed")
        delayed = self.classify("Cable installation delayed")
        negated = self.classify("No cable defect found")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(delayed["status"], "delayed")
        self.assertEqual(delayed["issue"], "delay")
        self.assertEqual(negated["status"], "negated")
        self.assertEqual(negated["issue"], "")
        self.assertEqual(negated["events"], [])
        self.assertIn("negated", negated["evidence"])

    def test_not_completed_never_counts_as_completion(self):
        for text in ("Cable installation not completed",
                     "La pose de câbles n'est pas terminée",
                     "No cable installation completed"):
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["status"], "negated")
                self.assertEqual(result["disposition"], "review")
                self.assertIn("Negated", result["review_reason"])

    def test_issue_negation_does_not_spread_to_activity(self):
        result = self.classify("No delays during cable installation completed")
        self.assertEqual(result["issue"], "")
        self.assertEqual(result["activity"], "cable installation")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["disposition"], "recognized")
        result = self.classify("Cable inspection without defects")
        self.assertEqual(result["events"], ["inspection"])
        self.assertEqual(result["status"], "negated")
        self.assertEqual(self.classify("Cable defect not found")["issue"], "")
        result = self.classify("Cable inspection: no defects or delays found")
        self.assertEqual(result["issue"], "")
        self.assertEqual(result["events"], ["inspection"])

    def test_clause_scope_and_ambiguous_status_review(self):
        for text in (
            "Cable installation completed but piping welding delayed",
            "Cable inspection completed; cable defect found",
            "Cable installation completed and cable testing delayed",
            "Cable testing completed failed",
            "No cable defect found but short circuit repaired",
        ):
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["disposition"], "review")
                self.assertIn("clause statuses", result["review_reason"])
        result = self.classify("No cable defect found but short circuit repaired")
        self.assertNotIn("defect", result["events"])
        self.assertIn("short circuit", result["events"])

    def test_multiple_compatible_events_and_specific_phrase_preference(self):
        result = self.classify(
            "Cable installation completed and cable testing completed")
        self.assertEqual(result["disposition"], "recognized")
        self.assertEqual(result["events"], ["cable installation", "cable testing"])
        self.assertNotIn("testing", result["events"])
        self.assertEqual(self.classify(
            "Cable inspection completed; unfamiliar jargon")["disposition"], "review")


class ConstructionAggregationTests(unittest.TestCase):
    def records(self):
        return pd.DataFrame({
            "date": ["2024-01-10", "2024-01-02", "2024-03-05", "invalid",
                     "2024-04-01", "2024-07-01", "invalid"],
            "description": [
                "Cable installation completed and cable testing completed",
                "ELEC cable pulling done", "Cable installation delayed",
                "Cable inspection completed", "", "Banana weather", None],
            "source": ["source.csv"] * 7, "sheet": ["CSV"] * 7,
            "row": list(range(2, 9)),
        }, index=[0, 0, 1, 1, 2, 2, 3])

    def test_all_rows_provenance_invalid_dates_and_blank_descriptions(self):
        records = self.records()
        result = ConstructionAnalyzer.analyze(records, ConstructionRecognizer())
        self.assertEqual(len(result.classifications), len(records))
        self.assertEqual(result.classifications.index.tolist(), records.index.tolist())
        for field in ("source", "sheet", "row", "description"):
            self.assertEqual(result.classifications[field].tolist(), records[field].tolist())
        self.assertEqual(result.invalid_dates, 2)
        self.assertEqual(len(result.matched), 3)
        self.assertEqual(len(result.review), 4)
        invalid = result.review.loc[result.review["date"].isna()]
        self.assertTrue(invalid["review_reason"].str.contains("Invalid date").all())
        self.assertEqual(invalid.iloc[0]["disposition"], "recognized")
        self.assertEqual(invalid.iloc[0]["original_date"], "invalid")
        self.assertIn("Blank description", invalid.iloc[1]["review_reason"])
        self.assertEqual(records.iloc[3]["date"], "invalid")
        for field in ("discipline", "activity", "issue", "status", "confidence",
                      "evidence", "review_reason", "disposition"):
            self.assertIn(field, result.matched)

    def test_trusted_rows_once_zero_periods_alias_grouping_and_first_occurrence(self):
        result = ConstructionAnalyzer.analyze(self.records(), ConstructionRecognizer())
        self.assertEqual(result.monthly.tolist(), [2, 0, 1, 0, 0, 0, 0])
        self.assertEqual(result.quarterly.tolist(), [3, 0, 0])
        events = result.top_events.set_index("event")
        self.assertEqual(events.loc["cable installation", "count"], 3)
        self.assertEqual(events.loc["cable testing", "count"], 1)
        self.assertEqual(events.loc["delay", "count"], 1)
        self.assertEqual(events.loc["cable installation", "first_occurrence"],
                         pd.Timestamp("2024-01-02"))
        self.assertEqual(events.loc["cable installation", "description"],
                         "ELEC cable pulling done")

    def test_fuzzy_review_excluded_but_valid_dates_extend_coverage(self):
        records = self.records().iloc[:2].copy()
        records.iloc[1, records.columns.get_loc("date")] = "2024-06-01"
        records.iloc[1, records.columns.get_loc("description")] = (
            "Electrical cable installaton completed")
        result = ConstructionAnalyzer.analyze(records, ConstructionRecognizer())
        self.assertEqual(len(result.matched), 1)
        self.assertEqual(result.monthly.tolist(), [1, 0, 0, 0, 0, 0])
        self.assertEqual(result.review.iloc[0]["disposition"], "review")

    def test_empty_and_all_invalid(self):
        for records in (self.records().iloc[:0], self.records().iloc[[3, 6]]):
            result = ConstructionAnalyzer.analyze(records, ConstructionRecognizer())
            self.assertTrue(result.matched.empty)
            self.assertTrue(result.monthly.empty)
            self.assertTrue(result.quarterly.empty)
            self.assertTrue(result.top_events.empty)
            self.assertEqual(len(result.classifications), len(records))

    def test_required_columns_and_optional_dataclass_defaults(self):
        with self.assertRaises(ValueError):
            ConstructionAnalyzer.analyze(pd.DataFrame(), ConstructionRecognizer())
        result = AnalysisResult(pd.DataFrame(), pd.Series(dtype="int64"),
                                pd.Series(dtype="int64"), pd.DataFrame(), [], 0)
        self.assertTrue(result.classifications.empty)
        self.assertTrue(result.review.empty)
        other = copy.copy(result)
        self.assertIsInstance(other, AnalysisResult)


if __name__ == "__main__":
    unittest.main()
