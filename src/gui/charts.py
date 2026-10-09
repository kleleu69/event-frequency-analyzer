"""Embedded matplotlib charts with zoom, pan, and image export."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from src.core.distribution import count_probabilities


class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)
        self.clear("Load files and run an analysis to see results.")

    def clear(self, message):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw_idle()

    def plot_counts(self, counts, title):
        if counts.empty:
            self.clear("No matching events with valid dates.")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        positions = list(range(len(counts)))
        ax.bar(positions, counts.values, color="#3979c6")
        step = max(1, (len(counts) + 11) // 12)
        ticks = positions[::step]
        ax.set_xticks(ticks, [str(counts.index[i]) for i in ticks], rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Matching rows")
        ax.grid(axis="y", alpha=0.2)
        self.canvas.draw_idle()

    def plot_windows(self, windows):
        if not windows:
            self.clear("At least three calendar months are needed for sliding windows.")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        positions = list(range(len(windows)))
        adjacent = [w.shift_pvalue if w.shift_pvalue is not None else float("nan") for w in windows]
        baseline = [w.baseline_pvalue if w.baseline_pvalue is not None else float("nan") for w in windows]
        ax.plot(positions, adjacent, "o-", label="Previous window")
        ax.plot(positions, baseline, "s--", label="First window baseline")
        for i, window in enumerate(windows):
            if window.shifted:
                ax.axvspan(i - 0.2, i + 0.2, color="#ef9a9a", alpha=0.4)
        step = max(1, (len(windows) + 7) // 8)
        ticks = positions[::step]
        ax.set_xticks(ticks, [str(windows[i].start)[:10] for i in ticks], rotation=35, ha="right")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Comparison p-value")
        ax.set_title("Three-month daily-count distributions (shaded: detected shift)")
        ax.legend()
        ax.grid(alpha=0.2)
        self.canvas.draw_idle()

    def plot_fit(self, window):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        counts = np.asarray(window.counts)
        if not len(counts):
            self.clear("No daily counts available.")
            return
        maximum = max(1, int(counts.max()))
        observed, frequencies = np.unique(counts, return_counts=True)
        ax.bar(observed, frequencies / len(counts), alpha=0.35, color="#3979c6", label="Daily counts")
        x = np.unique(np.linspace(0, maximum + 1, min(maximum + 2, 300)).astype(int))
        for fit in window.fits:
            probabilities = count_probabilities(fit.name, x, fit.parameters)
            ax.plot(x, probabilities, "--", label=f"{fit.name} (AIC {fit.aic:.1f})")
        ax.set_title(f"{str(window.start)[:10]} – {str(window.end)[:10]} • best: {window.best_fit}")
        ax.set_xlabel("Matching events per day")
        ax.set_ylabel("Daily-count probability")
        ax.legend(fontsize="small")
        self.canvas.draw_idle()
