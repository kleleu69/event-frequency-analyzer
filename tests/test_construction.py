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
        self.assertEqual(result["events"], ["electrical | delay (delayed)"])

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
            ("MECH equipment installation finished", "mechanical", "mechanical installation"),
            ("PIP piping welding finished", "piping", "welding"),
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
            ("Installation mécanique terminée", "mechanical", "mechanical installation"),
            ("Tuyauterie: pose de tuyaux terminée", "piping", "pipe installation"),
        ]
        for text, discipline, activity in examples:
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["discipline"], discipline)
                self.assertEqual(result["activity"], activity)
                self.assertEqual(result["disposition"], "recognized")

    def test_mechanical_and_piping_remain_distinct(self):
        for text, discipline, activity in (
            ("Pump installation completed", "mechanical", "mechanical installation"),
            ("Equipment installation completed", "mechanical", "mechanical installation"),
            ("Pipe installation completed", "piping", "pipe installation"),
        ):
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
                self.assertTrue(any(event.split(" | ", 1)[1].startswith(issue + " (")
                                    for event in result["events"]))

    def test_inspection_and_maintenance_do_not_imply_fault(self):
        for text in ("Cable inspection completed", "Electrical maintenance ongoing"):
            result = self.classify(text)
            self.assertEqual(result["issue"], "")
            self.assertFalse(any(event.split(" | ", 1)[1].startswith("electrical fault (")
                                 for event in result["events"]))
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
        self.assertEqual(completed["events"], ["electrical | cable installation (completed)"])
        self.assertEqual(delayed["events"],
                         ["electrical | cable installation (delayed)",
                          "electrical | delay (delayed)"])
        self.assertEqual(negated["events"], ["electrical | defect (negated)"])
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
        self.assertEqual(result["events"], [
            "electrical | inspection (unspecified)", "electrical | defect (negated)"])
        self.assertEqual(result["status"], "negated")
        self.assertEqual(self.classify("Cable defect not found")["issue"], "")
        result = self.classify("Cable inspection: no defects or delays found")
        self.assertEqual(result["issue"], "")
        self.assertCountEqual(result["events"], [
            "electrical | inspection (unspecified)", "electrical | defect (negated)",
            "electrical | delay (negated)"])

    def test_postposed_negative_findings_and_copulas(self):
        for text in (
            "Cable defect was not found", "Cable defect not detected",
            "Cable defect is absent", "Cable defects have not been observed",
            "Cable defect has never been observed",
            "Cable defect has not yet been detected",
            "Cable fault was not present", "Cable defect was ruled out",
            "Cable defect wasn't found", "Cable defect isn’t present",
            "Aucun défaut électrique n'a été détecté",
            "Défaut électrique non détecté",
            "Défaut électrique n'est pas présent",
            "Défaut électrique n'a jamais été observé",
            "Défaut électrique n'a pas encore été détecté",
        ):
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["issue"], "")
                self.assertEqual(result["status"], "negated")
                self.assertEqual(result["disposition"], "recognized")
                self.assertTrue(result["events"])
                self.assertTrue(all(event.endswith("(negated)") for event in result["events"]))
        result = self.classify("Cable defect was not repaired")
        self.assertEqual(result["issue"], "defect")
        self.assertEqual(result["disposition"], "review")
        result = self.classify("Cable defect was not detected but piping defect found")
        self.assertEqual(result["disposition"], "review")
        self.assertEqual(result["issue"], "defect")
        self.assertEqual(result["events"], [
            "electrical | defect (negated)", "piping | defect (unspecified)"])

    def test_unparsed_postposed_negative_finding_is_review_only(self):
        for text in (
            "Cable defect has not conclusively been detected",
            "Cable defect has never conclusively been observed",
            "Défaut électrique n'a pas formellement été détecté",
        ):
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["issue"], "")
                self.assertEqual(result["disposition"], "review")
                self.assertIn("Unresolved negative finding context", result["review_reason"])
                self.assertFalse(any(event.split(" | ", 1)[1].startswith("defect (")
                                     for event in result["events"]))
        for text in ("Cable defect has been observed",
                     "Electrical inspection has detected a defect"):
            result = self.classify(text)
            self.assertEqual(result["issue"], "defect")
            self.assertEqual(result["disposition"], "recognized")
        result = self.classify("Civil no access")
        self.assertEqual(result["issue"], "access constraint")
        self.assertEqual(result["disposition"], "recognized")

    def test_hypothetical_and_search_for_issues_are_not_actual_faults(self):
        for text in (
            "Electrical inspection for possible defects",
            "Cable testing for short circuits", "Cable inspection for defects",
            "Cable inspection to detect defects", "Cable defect suspected",
            "Inspection électrique pour défauts potentiels",
            "Electrical inspection for potential insulation failure",
        ):
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["issue"], "")
                self.assertEqual(result["disposition"], "review")
                self.assertIn("finding not established", result["review_reason"])
                self.assertIn("uncertain", result["evidence"])
                self.assertFalse(any(event.split(" | ", 1)[1].startswith((
                    "defect (", "short circuit (", "insulation failure ("))
                                     for event in result["events"]))
        result = self.classify("Cable inspection: defects were found")
        self.assertEqual(result["issue"], "defect")
        self.assertEqual(result["disposition"], "recognized")

    def test_test_targets_require_affirmative_findings_before_or_after_issue(self):
        for text in (
            "Cable short circuit test completed", "Breaker trip test passed",
            "Cable defect inspection completed", "Electrical insulation failure testing",
        ):
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["issue"], "")
                self.assertEqual(result["disposition"], "review")
                self.assertIn("finding not established", result["review_reason"])
                self.assertFalse(any(event.split(" | ", 1)[1].startswith((
                    "defect (", "short circuit (", "breaker trip (", "insulation failure ("))
                                     for event in result["events"]))
        for text in (
            "Electrical inspection found a defect",
            "Electrical inspection detected a defect",
            "Electrical inspection identified a defect",
            "Electrical inspection confirmed a defect",
            "Electrical inspection defect found",
            "Electrical inspection defect observed",
            "Electrical inspection defect present",
        ):
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["issue"], "defect")
                self.assertEqual(result["disposition"], "recognized")
                self.assertIn("electrical | defect (unspecified)", result["events"])
        result = self.classify("Cable testing delayed")
        self.assertEqual(result["issue"], "delay")
        self.assertEqual(result["disposition"], "recognized")
        result = self.classify("Electrical inspection defect not confirmed")
        self.assertEqual(result["issue"], "")
        self.assertTrue(any(event.endswith("defect (negated)")
                            for event in result["events"]))
        vocabulary = load_vocabulary()
        vocabulary["issues"]["arc flash"] = ["arc flash"]
        recognizer = ConstructionRecognizer(vocabulary)
        result = recognizer.classify("Electrical arc flash test completed")
        self.assertEqual(result["issue"], "")
        self.assertEqual(result["disposition"], "review")
        result = recognizer.classify("Electrical inspection identified arc flash")
        self.assertEqual(result["issue"], "arc flash")
        self.assertEqual(result["disposition"], "recognized")

    def test_status_inside_negated_issue_phrase_is_not_an_independent_status(self):
        result = self.classify("Cable testing completed without any insulation failure")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["issue"], "")
        self.assertEqual(result["disposition"], "recognized")
        self.assertEqual(result["events"], [
            "electrical | cable testing (completed)", "electrical | insulation failure (negated)"])
        self.assertNotIn("statuses:failed", result["evidence"])

    def test_coordinated_postposed_negative_findings_are_not_trusted(self):
        for text in (
            "Cable defects or short circuits were not detected",
            "Cable defects nor short circuits were not observed",
        ):
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["disposition"], "review")
                self.assertIn("negative finding context", result["review_reason"])
                self.assertIn("electrical | short circuit (negated)", result["events"])
        result = self.classify("Cable defect found with short circuit not detected")
        self.assertEqual(result["issue"], "defect")
        self.assertEqual(result["disposition"], "review")
        self.assertIn("Mixed positive and negated issue scope", result["review_reason"])

    def test_clause_scope_and_ambiguous_status_review(self):
        for text in (
            "Cable installation completed but piping welding delayed",
            "Cable inspection completed; cable defect found",
            "Cable installation completed and cable testing delayed",
            "Cable testing completed failed",
            "Cable testing completed; failed",
            "No cable defect found but short circuit repaired",
        ):
            with self.subTest(text=text):
                result = self.classify(text)
                self.assertEqual(result["disposition"], "review")
                self.assertIn("clause statuses", result["review_reason"])
        result = self.classify("No cable defect found but short circuit repaired")
        self.assertEqual(result["events"], [
            "electrical | defect (negated)", "electrical | short circuit (resolved)"])
        result = self.classify("No cable defect found but piping defect found")
        self.assertEqual(result["disposition"], "review")
        self.assertEqual(result["issue"], "defect")
        self.assertEqual(result["events"], [
            "electrical | defect (negated)", "piping | defect (unspecified)"])
        self.assertIn("clause statuses", result["review_reason"])

    def test_multiple_compatible_events_and_specific_phrase_preference(self):
        result = self.classify(
            "Cable installation completed and cable testing completed")
        self.assertEqual(result["disposition"], "recognized")
        self.assertEqual(result["events"], [
            "electrical | cable installation (completed)", "electrical | cable testing (completed)"])
        self.assertFalse(any(event.split(" | ", 1)[1].startswith("testing (")
                             for event in result["events"]))
        self.assertEqual(self.classify(
            "Cable inspection completed; unfamiliar jargon")["disposition"], "review")

    def test_event_labels_use_local_domains_and_only_unambiguous_global_fallback(self):
        result = self.classify("Electrical; testing completed")
        self.assertEqual(result["events"], ["electrical | testing (completed)"])
        self.assertEqual(result["disposition"], "recognized")
        result = self.classify("Electrical testing completed; piping testing completed")
        self.assertEqual(result["events"], [
            "electrical | testing (completed)", "piping | testing (completed)"])
        self.assertEqual(result["disposition"], "review")
        result = self.classify("Testing completed")
        self.assertEqual(result["events"], ["unspecified discipline | testing (completed)"])
        self.assertEqual(result["disposition"], "review")


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
        self.assertEqual(events.loc["electrical | cable installation (completed)", "count"], 2)
        self.assertEqual(events.loc["electrical | cable installation (delayed)", "count"], 1)
        self.assertEqual(events.loc["electrical | cable testing (completed)", "count"], 1)
        self.assertEqual(events.loc["electrical | delay (delayed)", "count"], 1)
        self.assertEqual(events.loc["electrical | cable installation (completed)", "first_occurrence"],
                         pd.Timestamp("2024-01-02"))
        self.assertEqual(events.loc["electrical | cable installation (completed)", "description"],
                         "ELEC cable pulling done")

    def test_top_events_group_aliases_but_keep_disciplines_distinct(self):
        descriptions = [
            "Electrical test completed", "ELEC tests done", "PIP testing completed",
            "Piping defect found", "ELEC defect found", "Electrical défaut found",
        ]
        records = pd.DataFrame({
            "date": ["2024-01-05", "2024-01-02", "2024-01-03",
                     "2024-01-04", "2024-01-01", "2024-01-06"],
            "description": descriptions, "source": ["events.csv"] * 6,
            "sheet": ["Daily"] * 6, "row": list(range(2, 8)),
        })
        result = ConstructionAnalyzer.analyze(records, ConstructionRecognizer())
        events = result.top_events.set_index("event")
        self.assertEqual(events["count"].to_dict(), {
            "electrical | testing (completed)": 2,
            "piping | testing (completed)": 1,
            "electrical | defect (unspecified)": 2,
            "piping | defect (unspecified)": 1,
        })
        self.assertEqual(events.loc["electrical | testing (completed)", "first_occurrence"],
                         pd.Timestamp("2024-01-02"))
        self.assertEqual(events.loc["electrical | testing (completed)", "description"],
                         "ELEC tests done")
        self.assertEqual(events.loc["piping | testing (completed)", "first_occurrence"],
                         pd.Timestamp("2024-01-03"))
        self.assertEqual(events.loc["electrical | defect (unspecified)", "first_occurrence"],
                         pd.Timestamp("2024-01-01"))
        self.assertEqual(result.monthly.tolist(), [6])
        self.assertEqual(result.matched["row"].tolist(), list(range(2, 8)))

    def test_fuzzy_review_excluded_but_valid_dates_extend_coverage(self):
        records = self.records().iloc[:2].copy()
        records.iloc[1, records.columns.get_loc("date")] = "2024-06-01"
        records.iloc[1, records.columns.get_loc("description")] = (
            "Electrical cable installaton completed")
        result = ConstructionAnalyzer.analyze(records, ConstructionRecognizer())
        self.assertEqual(len(result.matched), 1)
        self.assertEqual(result.monthly.tolist(), [1, 0, 0, 0, 0, 0])
        self.assertEqual(result.review.iloc[0]["disposition"], "review")

    def test_preexisting_original_dates_are_preserved(self):
        records = self.records()
        records["original_date"] = ["raw {}".format(value) for value in records["date"]]
        result = ConstructionAnalyzer.analyze(records, ConstructionRecognizer())
        self.assertEqual(result.classifications["original_date"].tolist(),
                         records["original_date"].tolist())
        invalid = result.review.loc[result.review["date"].isna()]
        self.assertEqual(invalid["original_date"].tolist(), ["raw invalid", "raw invalid"])

    def test_hypothetical_faults_are_review_only_not_trusted_chart_events(self):
        records = self.records().iloc[:2].copy()
        records["description"] = [
            "Electrical inspection for possible defects", "Cable testing for short circuits"]
        result = ConstructionAnalyzer.analyze(records, ConstructionRecognizer())
        self.assertTrue(result.matched.empty)
        self.assertTrue(result.top_events.empty)
        self.assertEqual(result.monthly.tolist(), [0])
        self.assertEqual(len(result.review), 2)
        self.assertTrue(result.classifications["issue"].eq("").all())

    def test_ambiguous_coordinated_negative_findings_are_excluded_from_charts(self):
        records = self.records().iloc[:1].copy()
        records["description"] = "Cable defects or short circuits were not detected"
        result = ConstructionAnalyzer.analyze(records, ConstructionRecognizer())
        self.assertTrue(result.matched.empty)
        self.assertTrue(result.top_events.empty)
        self.assertEqual(result.monthly.tolist(), [0])
        self.assertEqual(len(result.review), 1)
        self.assertIn("negative finding context",
                      result.review.iloc[0]["review_reason"])

    def test_unresolved_postposed_negation_cannot_enter_trusted_charts(self):
        records = self.records().iloc[:1].copy()
        records["description"] = "Cable defect has not conclusively been detected"
        result = ConstructionAnalyzer.analyze(records, ConstructionRecognizer())
        self.assertTrue(result.matched.empty)
        self.assertTrue(result.top_events.empty)
        self.assertEqual(result.monthly.tolist(), [0])
        self.assertEqual(len(result.review), 1)

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
