"""Latency Metrics: TTFT, per-token latency, end-to-end latency, percentiles."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class LatencyResult:
    """Container for latency measurement results."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    samples: int

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "samples": self.samples,
        }


class LatencyMetrics:
    """Compute latency metrics from timing measurements.

    Metrics computed:
    - Time to First Token (TTFT)
    - Per-token latency (inter-token interval)
    - End-to-end latency
    - Statistical measures: mean, std, P50, P90, P95, P99
    """

    def __init__(self) -> None:
        """Initialize latency metrics collector."""
        self._ttft_values: list[float] = []
        self._per_token_values: list[float] = []
        self._e2e_values: list[float] = []

    def compute(
        self,
        timings: list[float],
        unit: str = "seconds",
    ) -> dict[str, Any]:
        """Compute latency statistics from raw timings.

        Args:
            timings: List of timing measurements
            unit: Unit of input timings ("seconds" or "milliseconds")

        Returns:
            Dictionary with latency statistics
        """
        if not timings:
            return self._empty_result()

        # Filter out invalid timings
        valid_timings = [t for t in timings if t > 0 and t != float("inf")]
        if not valid_timings:
            return self._empty_result()

        # Convert to milliseconds if needed
        if unit == "seconds":
            valid_timings = [t * 1000 for t in valid_timings]

        result = self._compute_statistics(valid_timings)
        return result.to_dict()

    def compute_ttft(
        self,
        ttft_values: list[float],
        unit: str = "seconds",
    ) -> dict[str, Any]:
        """Compute Time to First Token statistics.

        Args:
            ttft_values: List of TTFT measurements
            unit: Unit of input ("seconds" or "milliseconds")

        Returns:
            Dictionary with TTFT statistics
        """
        if unit == "seconds":
            ttft_values = [t * 1000 for t in ttft_values]

        self._ttft_values.extend(ttft_values)
        result = self._compute_statistics(ttft_values)

        return {
            "ttft_" + k: v
            for k, v in result.to_dict().items()
        }

    def compute_per_token(
        self,
        token_timings: list[list[float]],
        unit: str = "seconds",
    ) -> dict[str, Any]:
        """Compute per-token latency statistics.

        Args:
            token_timings: List of per-request token timing lists
            unit: Unit of input ("seconds" or "milliseconds")

        Returns:
            Dictionary with per-token statistics
        """
        # Flatten all inter-token intervals
        intervals = []
        for timings in token_timings:
            if len(timings) < 2:
                continue
            for i in range(1, len(timings)):
                interval = timings[i] - timings[i - 1]
                if unit == "seconds":
                    interval *= 1000
                intervals.append(interval)

        if not intervals:
            return {"per_token_" + k: 0.0 for k in LatencyResult.__dataclass_fields__}

        self._per_token_values.extend(intervals)
        result = self._compute_statistics(intervals)

        return {
            "per_token_" + k: v
            for k, v in result.to_dict().items()
        }

    def compute_e2e(
        self,
        e2e_values: list[float],
        unit: str = "seconds",
    ) -> dict[str, Any]:
        """Compute end-to-end latency statistics.

        Args:
            e2e_values: List of end-to-end latency measurements
            unit: Unit of input ("seconds" or "milliseconds")

        Returns:
            Dictionary with E2E statistics
        """
        if unit == "seconds":
            e2e_values = [t * 1000 for t in e2e_values]

        self._e2e_values.extend(e2e_values)
        result = self._compute_statistics(e2e_values)

        return {
            "e2e_" + k: v
            for k, v in result.to_dict().items()
        }

    def _compute_statistics(self, values: list[float]) -> LatencyResult:
        """Compute statistical measures from a list of values.

        Args:
            values: List of measurements in milliseconds

        Returns:
            LatencyResult with computed statistics
        """
        if not values:
            return LatencyResult(
                mean_ms=0.0,
                std_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                samples=0,
            )

        sorted_values = sorted(values)
        n = len(sorted_values)

        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else 0.0

        return LatencyResult(
            mean_ms=mean,
            std_ms=std,
            min_ms=min(values),
            max_ms=max(values),
            p50_ms=self._percentile(sorted_values, 50),
            p90_ms=self._percentile(sorted_values, 90),
            p95_ms=self._percentile(sorted_values, 95),
            p99_ms=self._percentile(sorted_values, 99),
            samples=n,
        )

    def _percentile(self, sorted_values: list[float], p: float) -> float:
        """Compute percentile from sorted values.

        Args:
            sorted_values: Sorted list of values
            p: Percentile (0-100)

        Returns:
            Percentile value
        """
        if not sorted_values:
            return 0.0

        n = len(sorted_values)
        k = (n - 1) * (p / 100)
        f = int(k)
        c = f + 1

        if c >= n:
            return sorted_values[-1]

        # Linear interpolation
        d = k - f
        return sorted_values[f] * (1 - d) + sorted_values[c] * d

    def _empty_result(self) -> dict[str, float]:
        """Return empty result dictionary."""
        return {
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "samples": 0,
        }

    def reset(self) -> None:
        """Reset accumulated values."""
        self._ttft_values = []
        self._per_token_values = []
        self._e2e_values = []

    def get_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of all collected latency metrics.

        Returns:
            Dictionary with TTFT, per-token, and E2E summaries
        """
        summary = {}

        if self._ttft_values:
            result = self._compute_statistics(self._ttft_values)
            summary["ttft"] = result.to_dict()

        if self._per_token_values:
            result = self._compute_statistics(self._per_token_values)
            summary["per_token"] = result.to_dict()

        if self._e2e_values:
            result = self._compute_statistics(self._e2e_values)
            summary["e2e"] = result.to_dict()

        return summary
