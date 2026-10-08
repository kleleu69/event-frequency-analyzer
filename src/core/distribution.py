"""Likelihood fits and exploratory comparisons of overlapping count windows.

The date extent covers inclusive calendar months: both boundary months are
assumed observed in full. Each window spans three whole calendar months.
Mann–Whitney tests use asymptotic tie correction; Holm correction is applied
jointly to adjacent and baseline comparisons. Adjacent windows share two
months, so p-values are exploratory, not independent change-point evidence.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FitResult:
    name: str
    aic: float
    parameters: Tuple[float, ...]


@dataclass
class WindowResult:
    start: pd.Timestamp
    end: pd.Timestamp
    best_fit: str
    fits: List[FitResult]
    shift_pvalue: Optional[float] = None
    baseline_pvalue: Optional[float] = None
    shifted: bool = False
    daily_counts: pd.Series = field(default_factory=lambda: pd.Series(dtype="int64"))
    raw_shift_pvalue: Optional[float] = None
    raw_baseline_pvalue: Optional[float] = None
    limitation: str = "Overlapping windows share observations; comparisons are exploratory."

    @property
    def counts(self) -> np.ndarray:
        """Daily observations used for histogram plotting and model fitting."""
        return self.daily_counts.to_numpy()


def fit_distributions(sample) -> List[FitResult]:
    values = np.asarray(sample, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        return []
    fits = []

    def add(name, likelihood, parameter_count, parameters):
        if np.isfinite(likelihood):
            fits.append(FitResult(name, float(2 * parameter_count - 2 * likelihood),
                                  tuple(float(value) for value in parameters)))

    mean = float(values.mean())
    if np.equal(values, np.floor(values)).all():
        add("Poisson", stats.poisson.logpmf(values, mean).sum(), 1, (mean,))
    if np.ptp(values) == 0:
        return sorted(fits, key=lambda fit: fit.aic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = float(values.std())
        if std > 0:
            add("Normal", stats.norm.logpdf(values, loc=mean, scale=std).sum(),
                2, (mean, std))
        if mean > 0:
            add("Exponential", stats.expon.logpdf(values, loc=0, scale=mean).sum(),
                1, (0, mean))
        # Weibull density at zero is singular or zero for most shapes.
        # Do not alter zero counts or count an unsupported fit as valid.
        if (values > 0).all():
            try:
                shape, location, scale = stats.weibull_min.fit(values, floc=0)
                if shape > 0 and scale > 0:
                    add("Weibull", stats.weibull_min.logpdf(
                        values, shape, loc=location, scale=scale).sum(),
                        2, (shape, location, scale))
            except (ValueError, FloatingPointError, OverflowError):
                pass
    return sorted(fits, key=lambda fit: fit.aic)


def _pvalue(first, second):
    if np.ptp(np.concatenate((first, second))) == 0:
        return 1.0
    return float(stats.mannwhitneyu(first, second, alternative="two-sided",
                                  method="asymptotic").pvalue)


def _holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [1.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def analyze_windows(dates: pd.Series, start=None, end=None,
                    alpha: float = 0.05) -> List[WindowResult]:
    """Use event dates and, optionally, the full valid record coverage."""
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between zero and one.")
    dates = dates.dropna()
    if start is None:
        if dates.empty:
            return []
        start = dates.min()
    if end is None:
        if dates.empty:
            return []
        end = dates.max()
    first = pd.Timestamp(start).to_period("M")
    last = pd.Timestamp(end).to_period("M")
    months = pd.period_range(first, last, freq="M")
    if len(months) < 3:
        return []
    daily = dates.dt.normalize().value_counts()
    windows, comparisons = [], []
    for index in range(len(months) - 2):
        window_start = months[index].start_time
        window_end = months[index + 2].end_time.normalize()
        days = pd.date_range(window_start, window_end, freq="D")
        counts = daily.reindex(days, fill_value=0).astype("int64")
        counts.index.name = "date"
        counts.name = "count"
        fits = fit_distributions(counts.to_numpy())
        window = WindowResult(window_start, window_end,
                              fits[0].name if fits else "Unavailable", fits,
                              daily_counts=counts)
        if windows:
            window.raw_shift_pvalue = _pvalue(windows[-1].daily_counts.to_numpy(),
                                             counts.to_numpy())
            window.raw_baseline_pvalue = _pvalue(windows[0].daily_counts.to_numpy(),
                                                counts.to_numpy())
            comparisons.extend([(index, "shift_pvalue", window.raw_shift_pvalue),
                                (index, "baseline_pvalue", window.raw_baseline_pvalue)])
        windows.append(window)
    for comparison, corrected in zip(comparisons, _holm([item[2] for item in comparisons])):
        index, attribute, _ = comparison
        setattr(windows[index], attribute, corrected)
    for window in windows:
        window.shifted = any(value is not None and value < alpha
                             for value in (window.shift_pvalue, window.baseline_pvalue))
    return windows
