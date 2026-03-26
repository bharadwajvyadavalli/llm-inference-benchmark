"""Strategy Comparisons: Head-to-head analysis with statistical significance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two strategies."""

    strategy_a: str
    strategy_b: str
    metric: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    difference: float
    difference_pct: float
    p_value: float
    statistically_significant: bool
    winner: str | None
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "metric": self.metric,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "std_a": self.std_a,
            "std_b": self.std_b,
            "difference": self.difference,
            "difference_pct": self.difference_pct,
            "p_value": self.p_value,
            "statistically_significant": self.statistically_significant,
            "winner": self.winner,
            "confidence_level": self.confidence_level,
        }


class StrategyComparison:
    """Compare strategies with statistical significance testing."""

    def __init__(self, confidence_level: float = 0.95) -> None:
        """Initialize strategy comparison.

        Args:
            confidence_level: Confidence level for significance testing (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def compare_latency(
        self,
        results_a: dict[str, Any],
        results_b: dict[str, Any],
        metric: str = "mean_ms",
    ) -> ComparisonResult:
        """Compare latency between two strategies.

        Args:
            results_a: Results for strategy A
            results_b: Results for strategy B
            metric: Latency metric to compare

        Returns:
            ComparisonResult with statistical analysis
        """
        strategy_a = results_a.get("strategy", "Strategy A")
        strategy_b = results_b.get("strategy", "Strategy B")

        # Get raw timings for statistical test
        timings_a = results_a.get("raw_timings", [])
        timings_b = results_b.get("raw_timings", [])

        # Convert to milliseconds if needed
        timings_a = [t * 1000 if t < 100 else t for t in timings_a]
        timings_b = [t * 1000 if t < 100 else t for t in timings_b]

        if len(timings_a) < 2 or len(timings_b) < 2:
            # Fall back to summary statistics
            latency_a = results_a.get("latency_metrics", {})
            latency_b = results_b.get("latency_metrics", {})

            mean_a = latency_a.get(metric, 0)
            mean_b = latency_b.get(metric, 0)
            std_a = latency_a.get("std_ms", 0)
            std_b = latency_b.get("std_ms", 0)

            return ComparisonResult(
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                metric=f"latency_{metric}",
                mean_a=mean_a,
                mean_b=mean_b,
                std_a=std_a,
                std_b=std_b,
                difference=mean_a - mean_b,
                difference_pct=((mean_a - mean_b) / mean_b * 100) if mean_b > 0 else 0,
                p_value=1.0,
                statistically_significant=False,
                winner=None,
                confidence_level=self.confidence_level,
            )

        return self._perform_comparison(
            strategy_a,
            strategy_b,
            timings_a,
            timings_b,
            f"latency_{metric}",
            lower_is_better=True,
        )

    def compare_throughput(
        self,
        results_a: dict[str, Any],
        results_b: dict[str, Any],
    ) -> ComparisonResult:
        """Compare throughput between two strategies.

        Args:
            results_a: Results for strategy A
            results_b: Results for strategy B

        Returns:
            ComparisonResult with statistical analysis
        """
        strategy_a = results_a.get("strategy", "Strategy A")
        strategy_b = results_b.get("strategy", "Strategy B")

        throughput_a = results_a.get("throughput_metrics", {})
        throughput_b = results_b.get("throughput_metrics", {})

        mean_a = throughput_a.get("tokens_per_sec", 0)
        mean_b = throughput_b.get("tokens_per_sec", 0)

        # Without raw data, we can't do proper statistical testing
        return ComparisonResult(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            metric="throughput_tokens_per_sec",
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=0,
            std_b=0,
            difference=mean_a - mean_b,
            difference_pct=((mean_a - mean_b) / mean_b * 100) if mean_b > 0 else 0,
            p_value=1.0,
            statistically_significant=False,
            winner=strategy_a if mean_a > mean_b else (strategy_b if mean_b > mean_a else None),
            confidence_level=self.confidence_level,
        )

    def compare_memory(
        self,
        results_a: dict[str, Any],
        results_b: dict[str, Any],
    ) -> ComparisonResult:
        """Compare memory usage between two strategies.

        Args:
            results_a: Results for strategy A
            results_b: Results for strategy B

        Returns:
            ComparisonResult with statistical analysis
        """
        strategy_a = results_a.get("strategy", "Strategy A")
        strategy_b = results_b.get("strategy", "Strategy B")

        memory_a = results_a.get("memory_metrics", {})
        memory_b = results_b.get("memory_metrics", {})

        mean_a = memory_a.get("peak_memory_gb", 0)
        mean_b = memory_b.get("peak_memory_gb", 0)

        return ComparisonResult(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            metric="peak_memory_gb",
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=0,
            std_b=0,
            difference=mean_a - mean_b,
            difference_pct=((mean_a - mean_b) / mean_b * 100) if mean_b > 0 else 0,
            p_value=1.0,
            statistically_significant=False,
            winner=strategy_b if mean_a > mean_b else (strategy_a if mean_b > mean_a else None),
            confidence_level=self.confidence_level,
        )

    def _perform_comparison(
        self,
        strategy_a: str,
        strategy_b: str,
        values_a: list[float],
        values_b: list[float],
        metric: str,
        lower_is_better: bool = True,
    ) -> ComparisonResult:
        """Perform statistical comparison using paired t-test.

        Args:
            strategy_a: Name of strategy A
            strategy_b: Name of strategy B
            values_a: Measurements for strategy A
            values_b: Measurements for strategy B
            metric: Name of the metric being compared
            lower_is_better: Whether lower values are better

        Returns:
            ComparisonResult with statistical analysis
        """
        arr_a = np.array(values_a)
        arr_b = np.array(values_b)

        mean_a = float(np.mean(arr_a))
        mean_b = float(np.mean(arr_b))
        std_a = float(np.std(arr_a, ddof=1))
        std_b = float(np.std(arr_b, ddof=1))

        # Perform paired t-test if same length, otherwise independent t-test
        if len(arr_a) == len(arr_b):
            t_stat, p_value = stats.ttest_rel(arr_a, arr_b)
        else:
            t_stat, p_value = stats.ttest_ind(arr_a, arr_b)

        p_value = float(p_value)
        statistically_significant = p_value < self.alpha

        # Determine winner
        winner = None
        if statistically_significant:
            if lower_is_better:
                winner = strategy_a if mean_a < mean_b else strategy_b
            else:
                winner = strategy_a if mean_a > mean_b else strategy_b

        return ComparisonResult(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            metric=metric,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            difference=mean_a - mean_b,
            difference_pct=((mean_a - mean_b) / mean_b * 100) if mean_b > 0 else 0,
            p_value=p_value,
            statistically_significant=statistically_significant,
            winner=winner,
            confidence_level=self.confidence_level,
        )

    def full_comparison(
        self,
        results_a: dict[str, Any],
        results_b: dict[str, Any],
    ) -> dict[str, ComparisonResult]:
        """Perform full comparison across all metrics.

        Args:
            results_a: Results for strategy A
            results_b: Results for strategy B

        Returns:
            Dictionary mapping metric to ComparisonResult
        """
        comparisons = {}

        # Latency comparison
        comparisons["latency"] = self.compare_latency(results_a, results_b)

        # Throughput comparison
        comparisons["throughput"] = self.compare_throughput(results_a, results_b)

        # Memory comparison
        comparisons["memory"] = self.compare_memory(results_a, results_b)

        return comparisons

    def generate_comparison_report(
        self,
        comparisons: dict[str, ComparisonResult],
    ) -> str:
        """Generate a text report of comparisons.

        Args:
            comparisons: Dictionary of comparison results

        Returns:
            Formatted comparison report
        """
        lines = []
        lines.append("# Strategy Comparison Report")
        lines.append("")

        for metric, result in comparisons.items():
            lines.append(f"## {metric.title()}")
            lines.append("")
            lines.append(f"**{result.strategy_a}**: {result.mean_a:.4f} (std: {result.std_a:.4f})")
            lines.append(f"**{result.strategy_b}**: {result.mean_b:.4f} (std: {result.std_b:.4f})")
            lines.append("")
            lines.append(f"Difference: {result.difference:.4f} ({result.difference_pct:+.2f}%)")
            lines.append(f"p-value: {result.p_value:.4f}")
            lines.append(f"Statistically significant: {result.statistically_significant}")

            if result.winner:
                lines.append(f"**Winner: {result.winner}**")
            else:
                lines.append("No statistically significant difference")

            lines.append("")

        return "\n".join(lines)

    def compute_effect_size(
        self,
        values_a: list[float],
        values_b: list[float],
    ) -> dict[str, float]:
        """Compute effect size (Cohen's d) between two samples.

        Args:
            values_a: Measurements for group A
            values_b: Measurements for group B

        Returns:
            Dictionary with effect size metrics
        """
        arr_a = np.array(values_a)
        arr_b = np.array(values_b)

        mean_diff = np.mean(arr_a) - np.mean(arr_b)

        # Pooled standard deviation
        n_a = len(arr_a)
        n_b = len(arr_b)
        var_a = np.var(arr_a, ddof=1)
        var_b = np.var(arr_b, ddof=1)

        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {
            "cohens_d": float(cohens_d),
            "abs_cohens_d": float(abs_d),
            "interpretation": interpretation,
            "pooled_std": float(pooled_std),
        }
