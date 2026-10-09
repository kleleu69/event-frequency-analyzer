"""Date parsing that preserves input rows and tolerates mixed formats."""

import math
import numbers
import re
from datetime import date, datetime

import pandas as pd


def parse_dates(series: pd.Series, day_first: bool = False) -> pd.Series:
    """Parse mixed dates, returning naive datetimes and NaT for invalid values.

    Numeric values in the plausible contemporary Excel range 20,000–80,000
    are interpreted as Excel serial days (1899-12-30 origin). Eight-digit
    YYYYMMDD values (numeric or string) with years 1900–2199 are supported.
    Excel serial strings are also supported for lossless CSV imports.
    Small integers, years, and Unix timestamps are
    not dates. Numeric dates and IDs are inherently ambiguous; candidate
    detection requires a date-like header for numeric-valued columns.
    """
    def parse(value):
        if value is None or pd.isna(value):
            return pd.NaT
        if isinstance(value, bool):
            return pd.NaT
        if isinstance(value, numbers.Number):
            numeric = float(value)
            if not math.isfinite(numeric):
                return pd.NaT
            if numeric.is_integer() and 19000101 <= numeric <= 21991231:
                return pd.to_datetime(str(int(numeric)), format="%Y%m%d", errors="coerce")
            if not 20000 <= numeric <= 80000:
                return pd.NaT
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric, unit="D")
        if not isinstance(value, (str, date, datetime, pd.Timestamp)):
            return pd.NaT
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return pd.NaT
            if re.fullmatch(r"(?:19|20|21)\d{6}", value):
                return pd.to_datetime(value, format="%Y%m%d", errors="coerce")
            try:
                return parse(float(value))
            except ValueError:
                pass
        try:
            result = pd.to_datetime(value, dayfirst=day_first, errors="coerce",
                                    format="mixed")
            if pd.isna(result):
                return pd.NaT
            if result.tzinfo is not None:
                result = result.tz_localize(None)
            return result
        except (ValueError, TypeError, OverflowError):
            return pd.NaT

    return pd.Series([parse(value) for value in series], index=series.index,
                     name=series.name, dtype="datetime64[ns]")
