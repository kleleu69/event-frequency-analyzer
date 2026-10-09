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
    text = re.sub(r"\b(is|was|were|are|has|have|had|do|does|did)n['’]t\b",
                  r"\1 not", text)
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
        exact_spans = [(match[2], match[3]) for match in matches]
        for canonical, alias, fuzzy in self._aliases[axis]:
            if not fuzzy or canonical in exact_categories:
                continue
            size = len(alias.split())
            for start in range(len(tokens) - size + 1):
                if any(start < end and start + size > beginning
                       for beginning, end in exact_spans):
                    continue
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
        after = tokens[end:end + 8]
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
        if axis == "issues":
            copulas = {"was", "were", "is", "are", "has", "have", "had", "been",
                       "be", "being", "n", "ne", "a", "est", "sont", "ont", "ete"}
            while after and after[0] in copulas:
                after = after[1:]
            if after[:1] in (["absent"], ["absente"], ["absents"], ["absentes"]):
                return True
            if after[:2] == ["ruled", "out"]:
                return True
            if after and after[0] in {"not", "no", "non", "never", "pas", "jamais"}:
                after = after[1:]
                adverbs = {"yet", "ever", "still", "encore", "toujours", "deja"}
                while after and after[0] in copulas | adverbs:
                    after = after[1:]
                predicates = {
                    "found", "detected", "observed", "confirmed", "identified",
                    "present", "trouve", "trouvee", "confirme", "confirmee", "identifie",
                    "trouves", "detecte", "detectee", "detectes", "observe", "observee",
                    "observes", "presente", "constate", "constatee", "releve", "relevee",
                }
                if after and after[0] in predicates:
                    return True
        return False

    @staticmethod
    def _unresolved_negative_finding(tokens, end):
        following = tokens[end:end + 8]
        markers = {"not", "never", "pas", "jamais", "non"}
        findings = {
            "found", "detected", "observed", "confirmed", "identified", "present",
            "trouve", "trouvee", "detecte", "detectee", "observe", "observee",
            "confirme", "confirmee", "identifie", "identifiee", "constate", "presente",
        }
        return any(token in markers
                   and not any(word in findings for word in following[:index])
                   and any(word in findings for word in following[index + 1:])
                   for index, token in enumerate(following))

    @staticmethod
    def _uncertain_issue(tokens, start, end, activity_matches, canonical):
        before = tokens[max(0, start - 6):start]
        after = tokens[end:end + 4]
        uncertainty = {
            "possible", "possibly", "potential", "potentially", "suspected",
            "hypothetical", "suspect", "suspecte", "suspectee", "suspects",
            "potentiel", "potentielle", "potentiels", "eventuel", "eventuelle",
            "eventuels", "suspicion", "may", "might", "could",
        }
        if any(token in uncertainty for token in before + after):
            return True
        activities = {match[0] for match in activity_matches}
        if activities.intersection({"inspection", "testing", "cable testing", "loop checking"}):
            prefix = " ".join(before)
            searching = ("for" in before or "recherche" in before or re.search(
                r"\b(?:to (?:detect|find|identify)|afin de|pour)\b", prefix))
            actual = any(token in {
                "found", "detected", "observed", "confirmed", "identified", "present",
                "trouve", "trouvee", "detecte", "detectee", "observe", "observee",
                "constate", "constatee", "confirme", "confirmee", "identifie",
                "identifiee", "presente",
            } for token in before + after)
            logistical = {"delay", "missing materials", "rework", "access constraint"}
            if not actual and (searching or canonical not in logistical):
                return True
        return False

    def classify(self, text) -> dict:
        """Return canonical fields and ``discipline | event (status)`` labels.

        Multiple axis values use ``; `` separators. A negative finding produces
        a separate ``(negated)`` event but never a positive issue. Labels without
        explicit status use ``(unspecified)``.
        """
        result = {field: "" for field in FIELDS}
        result.update(confidence=0.0, disposition="unfamiliar", events=[])
        if text is None or (not isinstance(text, str) and pd.isna(text)):
            result["review_reason"] = "Blank description."
            return result
        text = str(text).strip()
        if not _normalize(text):
            result["review_reason"] = "Blank description."
            return result
        inferred = {
            "excavation": "civil works", "concrete placement": "civil works",
            "cable installation": "electrical", "cable testing": "electrical",
            "pipe installation": "piping", "mechanical installation": "mechanical",
            "loop checking": "instrumentation",
            "electrical fault": "electrical", "short circuit": "electrical",
            "earth fault": "electrical", "breaker trip": "electrical",
            "insulation failure": "electrical", "power outage": "electrical",
        }
        # Splitting before normalization keeps punctuation and contrast scope.
        clauses = [_normalize(part) for part in re.split(
            r"[;.!?\n]+|,\s*|\b(?:but|however|whereas|mais|cependant|and|et)\b",
            text, flags=re.IGNORECASE) if _normalize(part)]
        found = {axis: [] for axis in AXES}
        evidence, scores, clause_states, events, domain_events = [], [], [], [], []
        negated_issues = False
        negated_status = False
        uncertain_issues = False
        unresolved_negative_finding = False
        mixed_issue_scope = False
        unfamiliar_clauses = 0
        for clause in clauses:
            tokens = clause.split()
            matches = {axis: self._matches(clause, axis) for axis in AXES}
            matches["statuses"] = [match for match in matches["statuses"] if not any(
                issue[2] <= match[2] and issue[3] >= match[3]
                and issue[3] - issue[2] > match[3] - match[2]
                for issue in matches["issues"])]
            barriers = [match for axis in ("activities", "issues", "statuses")
                        for match in matches[axis]]
            if not any(matches.values()):
                unfamiliar_clauses += 1
            local = {axis: [] for axis in AXES}
            local_negated = False
            local_status_negated = False
            local_negative_issue = False
            negative_events = []
            for axis in AXES:
                for canonical, alias, start, end, score in matches[axis]:
                    negated = axis in ("issues", "statuses", "activities") and self._negated(
                        tokens, start, end, axis, barriers)
                    if (axis == "statuses" and matches["activities"]
                            and not local["activities"] and local_negated):
                        negated = True
                    unresolved = axis == "issues" and not negated and self._unresolved_negative_finding(
                        tokens, end)
                    uncertain = axis == "issues" and not negated and (
                        unresolved or self._uncertain_issue(
                            tokens, start, end, matches["activities"], canonical))
                    evidence.append("{}:{} [{}{}]".format(
                        axis, canonical, "negated " if negated else (
                            "uncertain " if uncertain else ""),
                        ("exact " if score == 100 else "fuzzy ") + alias))
                    scores.append(score)
                    if negated:
                        local_negated = True
                        negated_issues |= axis == "issues"
                        local_negative_issue |= axis == "issues"
                        # An issue alias can also be a status ("no delays").
                        issue_overlap = any(item[2:4] == (start, end)
                                            for item in matches["issues"])
                        status_negation = axis == "activities" or (
                            axis == "statuses" and not issue_overlap)
                        negated_status |= status_negation
                        local_status_negated |= status_negation
                        if axis in ("activities", "issues"):
                            negative_events.append(canonical)
                            domain_events.append(canonical)
                    elif uncertain:
                        uncertain_issues |= not unresolved
                        unresolved_negative_finding |= unresolved
                        domain_events.append(canonical)
                    else:
                        local[axis].append(canonical)
                        found[axis].append(canonical)
            mixed_issue_scope |= bool(local["issues"]) and local_negative_issue
            if local["activities"] or local["issues"] or local["statuses"] or local_negated:
                state = _unique(local["statuses"])
                if not state:
                    state = ["negated"] if local_negated else ["unspecified"]
                clause_states.append(tuple(state))
            positive_events = local["activities"] + local["issues"]
            domain_events.extend(positive_events)
            event_status = "; ".join(_unique(local["statuses"])) or (
                "negated" if local_status_negated else "unspecified")
            local_domains = _unique(local["disciplines"] + [
                inferred[match[0]] for axis in ("activities", "issues")
                for match in matches[axis] if match[0] in inferred
                and inferred[match[0]] in self.vocabulary["disciplines"]])
            events.extend((event, event_status, local_domains) for event in positive_events)
            events.extend((event, "negated", local_domains) for event in negative_events)
        # Domain-specific activities and faults provide explicit domain context.
        for event in domain_events:
            if event in inferred and inferred[event] in self.vocabulary["disciplines"]:
                found["disciplines"].append(inferred[event])
                evidence.append("disciplines:{} [inferred from {}]".format(
                    inferred[event], event))
        for axis, field in zip(AXES, ("discipline", "activity", "issue", "status")):
            result[field] = "; ".join(_unique(found[axis]))
        if not result["status"] and (negated_issues or negated_status):
            result["status"] = "negated"
        global_domains = _unique(found["disciplines"])

        def event_label(item):
            event, status, domains = item
            if not domains and len(global_domains) == 1:
                domains = global_domains
            discipline = "; ".join(domains) or "unspecified discipline"
            return "{} | {} ({})".format(discipline, event, status)

        result["events"] = _unique(event_label(event) for event in events)
        result["evidence"] = "; ".join(_unique(evidence))
        reasons = []
        if not scores:
            result["review_reason"] = "No familiar construction vocabulary."
            return result
        if any(score < 100 for score in scores):
            reasons.append("Fuzzy vocabulary match requires verification.")
        if unfamiliar_clauses:
            reasons.append("Unfamiliar clause context requires review.")
        if uncertain_issues:
            reasons.append("Hypothetical or search-for issue context; finding not established.")
        if unresolved_negative_finding:
            reasons.append("Unresolved negative finding context; finding not established.")
        if mixed_issue_scope:
            reasons.append("Mixed positive and negated issue scope in one clause; review findings.")
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
        if "original_date" not in frame:
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
