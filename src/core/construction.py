"""Lightweight, editable offline construction recognition.

Confidence is a rule-based evidence score, not a calibrated probability.
Negation and statuses are scoped to clauses; uncertain matches stay reviewable.
"""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from fuzzywuzzy import fuzz

from .analyzer import EventAnalyzer


AXES = ("disciplines", "activities", "issues", "statuses")
FIELDS = ("discipline", "activity", "issue", "status", "confidence",
          "review_reason", "evidence", "disposition", "events")


def _normalize(text):
    text = unicodedata.normalize("NFKD", str(text).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).split())


def validate_vocabulary(vocabulary):
    """Validate an in-memory editable vocabulary, returning it unchanged."""
    if not isinstance(vocabulary, dict):
        raise ValueError("Vocabulary must be a JSON object.")
    version = vocabulary.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Vocabulary version must be a nonempty string.")
    for axis in AXES:
        categories = vocabulary.get(axis)
        if not isinstance(categories, dict) or not categories:
            raise ValueError("Vocabulary {} must be a nonempty object.".format(axis))
        owners = {}
        for canonical, aliases in categories.items():
            if not isinstance(canonical, str) or not _normalize(canonical):
                raise ValueError("Vocabulary {} contains an empty category.".format(axis))
            if not isinstance(aliases, list) or not aliases:
                raise ValueError("Aliases for {} must be a nonempty list.".format(canonical))
            for alias in [canonical] + aliases:
                if not isinstance(alias, str) or not _normalize(alias):
                    raise ValueError("Aliases for {} must be nonempty strings.".format(canonical))
                normalized = _normalize(alias)
                owner = owners.get(normalized)
                if owner is not None and owner != canonical:
                    raise ValueError("Ambiguous alias {!r} in {}: {} / {}.".format(
                        alias, axis, owner, canonical))
                owners[normalized] = canonical
    return vocabulary


def load_vocabulary(path=None) -> dict:
    """Load and validate a UTF-8 JSON vocabulary beside this module by default."""
    path = Path(path) if path is not None else Path(__file__).with_name(
        "construction_vocabulary.json")
    try:
        with path.open(encoding="utf-8-sig") as handle:
            vocabulary = json.load(handle)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("Cannot read construction vocabulary: {}.".format(error)) from error
    return validate_vocabulary(vocabulary)


def _unique(items):
    return list(dict.fromkeys(items))


class ConstructionRecognizer:
    def __init__(self, vocabulary=None, threshold=85):
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 100:
            raise ValueError("Fuzzy threshold must be between 0 and 100.")
        self.threshold = threshold
        self.vocabulary = (validate_vocabulary(vocabulary)
                           if vocabulary is not None else load_vocabulary())
        self._aliases = {}
        for axis in AXES:
            aliases = []
            for canonical, values in self.vocabulary[axis].items():
                for alias in _unique([canonical] + values):
                    normalized = _normalize(alias)
                    # Contractor codes are exact-only, including longer uppercase codes.
                    fuzzy = len(normalized) >= 6 and not alias.isupper() and all(
                        len(token) >= 3 for token in normalized.split())
                    aliases.append((canonical, normalized, fuzzy))
            self._aliases[axis] = aliases

    def _matches(self, clause, axis):
        tokens = clause.split()
        matches = []
        exact_categories = set()
        for canonical, alias, _ in self._aliases[axis]:
            size = len(alias.split())
            for start in range(len(tokens) - size + 1):
                if " ".join(tokens[start:start + size]) == alias:
                    matches.append((canonical, alias, start, start + size, 100))
                    exact_categories.add(canonical)
        for canonical, alias, fuzzy in self._aliases[axis]:
            if not fuzzy or canonical in exact_categories:
                continue
            size = len(alias.split())
            for start in range(len(tokens) - size + 1):
                window = " ".join(tokens[start:start + size])
                if len(window) < 5:
                    continue
                if ((alias in window or window in alias)
                        and abs(len(alias) - len(window)) > 1):
                    continue
                score = fuzz.ratio(alias, window)
                if score >= self.threshold:
                    matches.append((canonical, window, start, start + size, score))
        # Prefer specific phrases over generic words contained in the same phrase.
        return [match for match in matches if not any(
            other[2] <= match[2] and other[3] >= match[3]
            and other[3] - other[2] > match[3] - match[2]
            for other in matches)]

    @staticmethod
    def _negated(tokens, start, end, axis, barriers):
        boundary = max([0] + [match[3] for match in barriers if match[3] <= start])
        before = tokens[max(boundary, start - 4):start]
        after = tokens[end:end + 3]
        if before[:1] in (["or"], ["nor"], ["ni"]):
            previous = [match for match in barriers if match[3] == boundary]
            if previous:
                match = previous[0]
                if ConstructionRecognizer._negated(
                        tokens, match[2], match[3], axis, barriers):
                    return True
        # "not only" is additive, not negation; "no access" is itself an issue.
        if tokens[start:end] == ["no", "access"]:
            return False
        prefix = " ".join(before)
        if re.search(r"\bnot only\b", prefix):
            return False
        if any(token in {"no", "not", "without", "never", "sans", "aucun",
                         "aucune", "pas", "jamais", "ni"} for token in before):
            return True
        if axis == "issues" and (after[:1] == ["absent"] or
                                after[:2] in (["not", "found"], ["non", "trouve"],
                                              ["non", "detecte"])):
            return True
        return False

    def classify(self, text) -> dict:
        result = {field: "" for field in FIELDS}
        result.update(confidence=0.0, disposition="unfamiliar", events=[])
        if text is None or (not isinstance(text, str) and pd.isna(text)):
            result["review_reason"] = "Blank description."
            return result
        text = str(text).strip()
        if not _normalize(text):
            result["review_reason"] = "Blank description."
            return result
        # Splitting before normalization keeps punctuation and contrast scope.
        clauses = [_normalize(part) for part in re.split(
            r"[;.!?\n]+|,\s*|\b(?:but|however|whereas|mais|cependant|and|et)\b",
            text, flags=re.IGNORECASE) if _normalize(part)]
        found = {axis: [] for axis in AXES}
        evidence, scores, clause_states, events = [], [], [], []
        negated_issues = False
        negated_status = False
        unfamiliar_clauses = 0
        for clause in clauses:
            tokens = clause.split()
            matches = {axis: self._matches(clause, axis) for axis in AXES}
            barriers = [match for axis in ("activities", "issues", "statuses")
                        for match in matches[axis]]
            if not any(matches.values()):
                unfamiliar_clauses += 1
            local = {axis: [] for axis in AXES}
            local_negated = False
            for axis in AXES:
                for canonical, alias, start, end, score in matches[axis]:
                    negated = axis in ("issues", "statuses", "activities") and self._negated(
                        tokens, start, end, axis, barriers)
                    if (axis == "statuses" and matches["activities"]
                            and not local["activities"] and local_negated):
                        negated = True
                    evidence.append("{}:{} [{}{}]".format(
                        axis, canonical, "negated " if negated else "",
                        ("exact " if score == 100 else "fuzzy ") + alias))
                    scores.append(score)
                    if negated:
                        local_negated = True
                        negated_issues |= axis == "issues"
                        # An issue alias can also be a status ("no delays").
                        issue_overlap = any(item[2:4] == (start, end)
                                            for item in matches["issues"])
                        negated_status |= axis == "activities" or (
                            axis == "statuses" and not issue_overlap)
                    else:
                        local[axis].append(canonical)
                        found[axis].append(canonical)
            if local["activities"] or local["issues"] or local_negated:
                state = _unique(local["statuses"])
                if not state:
                    state = ["negated"] if local_negated else ["unspecified"]
                clause_states.append(tuple(state))
            events.extend(local["activities"] + local["issues"])
        # Domain-specific activities and faults provide explicit domain context.
        inferred = {
            "excavation": "civil works", "concrete placement": "civil works",
            "cable installation": "electrical", "cable testing": "electrical",
            "pipe installation": "mechanical piping", "loop checking": "instrumentation",
            "electrical fault": "electrical", "short circuit": "electrical",
            "earth fault": "electrical", "breaker trip": "electrical",
            "insulation failure": "electrical", "power outage": "electrical",
        }
        for event in events:
            if event in inferred and inferred[event] in self.vocabulary["disciplines"]:
                found["disciplines"].append(inferred[event])
                evidence.append("disciplines:{} [inferred from {}]".format(
                    inferred[event], event))
        for axis, field in zip(AXES, ("discipline", "activity", "issue", "status")):
            result[field] = "; ".join(_unique(found[axis]))
        if not result["status"] and (negated_issues or negated_status):
            result["status"] = "negated"
        result["events"] = _unique(events)
        result["evidence"] = "; ".join(_unique(evidence))
        reasons = []
        if not scores:
            result["review_reason"] = "No familiar construction vocabulary."
            return result
        if any(score < 100 for score in scores):
            reasons.append("Fuzzy vocabulary match requires verification.")
        if unfamiliar_clauses:
            reasons.append("Unfamiliar clause context requires review.")
        if not result["discipline"]:
            reasons.append("Missing or ambiguous discipline context.")
        if not events and not negated_issues:
            reasons.append("Missing activity or issue context.")
        if any(len(states) > 1 for states in clause_states) or len(set(clause_states)) > 1:
            reasons.append("Different or ambiguous clause statuses; review each event.")
        if negated_status:
            reasons.append("Negated activity or status; completion is not established.")
        if len(_unique(found["disciplines"])) > 1:
            reasons.append("Multiple disciplines; review event attribution.")
        result["confidence"] = round(min(scores) / 100 * (0.75 if reasons else 0.96), 3)
        result["review_reason"] = " ".join(reasons)
        result["disposition"] = "review" if reasons else "recognized"
        return result


class ConstructionAnalyzer:
    @staticmethod
    def analyze(records, recognizer):
        required = {"date", "description", "source", "sheet", "row"}
        if not required.issubset(records.columns):
            raise ValueError("Records require columns: date, description, source, sheet, row.")
        frame = records.copy()
        frame["original_date"] = frame["date"]
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        invalid = frame["date"].isna()
        classifications = pd.DataFrame(
            [recognizer.classify(text) for text in frame["description"]],
            index=frame.index, columns=FIELDS)
        # Positional assignment also preserves duplicate source indexes.
        for field in FIELDS:
            frame[field] = classifications[field].to_numpy()
        for position in range(len(frame)):
            if invalid.iloc[position]:
                column = frame.columns.get_loc("review_reason")
                reason = frame.iloc[position, column]
                frame.iloc[position, column] = (reason + " Invalid date.").strip()
        valid = frame.loc[~invalid].copy()
        trusted = frame["disposition"].eq("recognized") & ~invalid
        matched = frame.loc[trusted].copy()
        words = _unique(event for items in matched["events"] for event in items)
        result = EventAnalyzer._aggregate(valid, matched, words, int(invalid.sum()))
        result.classifications = frame
        result.review = frame.loc[~trusted].copy()
        return result
