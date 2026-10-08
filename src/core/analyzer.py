"""Aggregate matched records while retaining source provenance."""

from dataclasses import dataclass
from typing import List

import pandas as pd

from .distribution import WindowResult, analyze_windows
from .matcher import WordMatcher


@dataclass
class AnalysisResult:
    matched: pd.DataFrame
    monthly: pd.Series
    quarterly: pd.Series
    top_events: pd.DataFrame
    windows: List[WindowResult]
    invalid_dates: int


class EventAnalyzer:
    @staticmethod
    def analyze(records: pd.DataFrame, matcher: WordMatcher) -> AnalysisResult:
        required = {"date", "description", "source", "sheet", "row"}
        if not required.issubset(records.columns):
            raise ValueError("Records require columns: date, description, source, sheet, row.")
        frame = records.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        invalid_dates = int(frame["date"].isna().sum())
        valid = frame.loc[frame["date"].notna()].copy()
        valid["description"] = valid["description"].fillna("").astype(str)
        valid["events"] = valid["description"].map(matcher.match)
        matched = valid.loc[valid["events"].map(bool)].copy()

        def aggregate(frequency):
            if valid.empty:
                return pd.Series([], index=pd.PeriodIndex([], freq=frequency, name="period"),
                                 dtype="int64", name="count")
            periods = pd.period_range(valid["date"].min().to_period(frequency),
                                      valid["date"].max().to_period(frequency),
                                      freq=frequency, name="period")
            counts = matched["date"].dt.to_period(frequency).value_counts()
            result = counts.reindex(periods, fill_value=0).astype("int64")
            result.name = "count"
            return result

        events = []
        for word in matcher.words:
            occurrences = matched.loc[matched["events"].map(lambda items: word in items)]
            if not occurrences.empty:
                first = occurrences.sort_values("date", kind="stable").iloc[0]
                events.append({"event": word, "count": len(occurrences),
                               "first_occurrence": first["date"],
                               "description": first["description"]})
        top_events = pd.DataFrame(events, columns=["event", "count",
                                                  "first_occurrence", "description"])
        if not top_events.empty:
            top_events = top_events.sort_values(["count", "first_occurrence", "event"],
                                                ascending=[False, True, True]).head(10).reset_index(drop=True)
        windows = [] if valid.empty else analyze_windows(
            matched["date"], valid["date"].min(), valid["date"].max())
        return AnalysisResult(matched, aggregate("M"), aggregate("Q"),
                              top_events, windows, invalid_dates)
