"""Throughput Metrics: Tokens/sec, requests/sec, batch throughput."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ThroughputResult:
    """Container for throughput measurement results."""

    tokens_per_sec: float
    prefill_tokens_per_sec: float
    generation_tokens_per_sec: float
    requests_per_sec: float
    batch_throughput: float
    scaling_efficiency: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "tokens_per_sec": self.tokens_per_sec,
            "prefill_tokens_per_sec": self.prefill_tokens_per_sec,
            "generation_tokens_per_sec": self.generation_tokens_per_sec,
            "requests_per_sec": self.requests_per_sec,
            "batch_throughput": self.batch_throughput,
            "scaling_efficiency": self.scaling_efficiency,
        }


class ThroughputMetrics:
    """Compute throughput metrics from benchmark results.

    Metrics computed:
    - Generation tokens/sec
    - Prefill tokens/sec
    - Requests/sec
    - Batch throughput
    - Throughput scaling efficiency
    """

    def __init__(self) -> None:
        """Initialize throughput metrics collector."""
        self._measurements: list[dict[str, Any]] = []

    def compute(
        self,
        timings: list[float],
        token_counts: list[int],
        num_requests: int,
        prefill_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Compute throughput metrics from timing and token data.

        Args:
            timings: List of timing measurements in seconds
            token_counts: List of generated token counts per run
            num_requests: Number of requests in the workload
            prefill_tokens: Number of prefill (input) tokens (optional)

        Returns:
            Dictionary with throughput metrics
        """
        if not timings or not token_counts:
            return self._empty_result()

        # Filter out invalid measurements
        valid_pairs = [
            (t, c)
            for t, c in zip(timings, token_counts)
            if t > 0 and t != float("inf") and c > 0
        ]

        if not valid_pairs:
            return self._empty_result()

        # Calculate averages
        total_time = sum(t for t, _ in valid_pairs)
        total_tokens = sum(c for _, c in valid_pairs)
        num_runs = len(valid_pairs)

        # Average time per run
        avg_time = total_time / num_runs
        avg_tokens = total_tokens / num_runs

        # Tokens per second
        tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0

        # Requests per second
        requests_per_sec = num_requests / avg_time if avg_time > 0 else 0

        # Prefill throughput (if provided)
        prefill_tps = 0.0
        if prefill_tokens is not None and prefill_tokens > 0:
            # Estimate prefill time as a fraction of total
            # In practice, this should be measured separately
            estimated_prefill_time = avg_time * 0.1  # Rough estimate
            prefill_tps = prefill_tokens / estimated_prefill_time if estimated_prefill_time > 0 else 0

        # Generation throughput (tokens generated per second)
        gen_tps = tokens_per_sec  # Same as total for generation-only

        result = ThroughputResult(
            tokens_per_sec=tokens_per_sec,
            prefill_tokens_per_sec=prefill_tps,
            generation_tokens_per_sec=gen_tps,
            requests_per_sec=requests_per_sec,
            batch_throughput=tokens_per_sec,  # Same for single batch
            scaling_efficiency=1.0,  # Baseline
        )

        return result.to_dict()

    def compute_batch_scaling(
        self,
        batch_results: dict[int, dict[str, float]],
    ) -> dict[str, Any]:
        """Compute throughput scaling across batch sizes.

        Args:
            batch_results: Dictionary mapping batch_size to throughput results

        Returns:
            Dictionary with scaling analysis
        """
        if not batch_results:
            return {}

        batch_sizes = sorted(batch_results.keys())

        # Get baseline (batch_size=1) throughput
        baseline_tps = batch_results.get(1, {}).get("tokens_per_sec", 0)
        if baseline_tps == 0 and batch_sizes:
            baseline_tps = batch_results[batch_sizes[0]].get("tokens_per_sec", 0)

        scaling_data = {}
        for batch_size in batch_sizes:
            result = batch_results[batch_size]
            tps = result.get("tokens_per_sec", 0)

            # Ideal scaling: linear with batch size
            ideal_tps = baseline_tps * batch_size

            # Actual scaling efficiency
            efficiency = (tps / ideal_tps) if ideal_tps > 0 else 0

            scaling_data[batch_size] = {
                "tokens_per_sec": tps,
                "ideal_tokens_per_sec": ideal_tps,
                "scaling_efficiency": efficiency,
                "speedup_vs_baseline": (tps / baseline_tps) if baseline_tps > 0 else 0,
            }

        return {
            "batch_scaling": scaling_data,
            "optimal_batch_size": self._find_optimal_batch_size(scaling_data),
        }

    def _find_optimal_batch_size(
        self,
        scaling_data: dict[int, dict[str, float]],
    ) -> int:
        """Find the optimal batch size based on throughput.

        Args:
            scaling_data: Scaling data from compute_batch_scaling

        Returns:
            Optimal batch size
        """
        if not scaling_data:
            return 1

        # Find batch size with highest throughput
        best_batch_size = 1
        best_throughput = 0

        for batch_size, data in scaling_data.items():
            tps = data.get("tokens_per_sec", 0)
            if tps > best_throughput:
                best_throughput = tps
                best_batch_size = batch_size

        return best_batch_size

    def compute_prefill_vs_decode(
        self,
        prefill_times: list[float],
        prefill_tokens: list[int],
        decode_times: list[float],
        decode_tokens: list[int],
    ) -> dict[str, float]:
        """Compute separate prefill and decode throughput.

        Args:
            prefill_times: List of prefill phase timings
            prefill_tokens: List of prefill token counts
            decode_times: List of decode phase timings
            decode_tokens: List of decode token counts

        Returns:
            Dictionary with prefill and decode throughput
        """
        # Prefill throughput
        total_prefill_time = sum(prefill_times)
        total_prefill_tokens = sum(prefill_tokens)
        prefill_tps = (
            total_prefill_tokens / total_prefill_time
            if total_prefill_time > 0
            else 0
        )

        # Decode throughput
        total_decode_time = sum(decode_times)
        total_decode_tokens = sum(decode_tokens)
        decode_tps = (
            total_decode_tokens / total_decode_time if total_decode_time > 0 else 0
        )

        return {
            "prefill_tokens_per_sec": prefill_tps,
            "decode_tokens_per_sec": decode_tps,
            "prefill_total_time": total_prefill_time,
            "decode_total_time": total_decode_time,
            "prefill_pct_time": (
                total_prefill_time / (total_prefill_time + total_decode_time) * 100
                if (total_prefill_time + total_decode_time) > 0
                else 0
            ),
        }

    def _empty_result(self) -> dict[str, float]:
        """Return empty result dictionary."""
        return {
            "tokens_per_sec": 0.0,
            "prefill_tokens_per_sec": 0.0,
            "generation_tokens_per_sec": 0.0,
            "requests_per_sec": 0.0,
            "batch_throughput": 0.0,
            "scaling_efficiency": 0.0,
        }

    def add_measurement(
        self,
        batch_size: int,
        time_seconds: float,
        tokens_generated: int,
        num_requests: int,
    ) -> None:
        """Add a throughput measurement.

        Args:
            batch_size: Batch size used
            time_seconds: Total time in seconds
            tokens_generated: Total tokens generated
            num_requests: Number of requests processed
        """
        self._measurements.append(
            {
                "batch_size": batch_size,
                "time_seconds": time_seconds,
                "tokens_generated": tokens_generated,
                "num_requests": num_requests,
                "tokens_per_sec": tokens_generated / time_seconds if time_seconds > 0 else 0,
                "requests_per_sec": num_requests / time_seconds if time_seconds > 0 else 0,
            }
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all throughput measurements.

        Returns:
            Dictionary with aggregated throughput statistics
        """
        if not self._measurements:
            return {}

        # Group by batch size
        by_batch_size: dict[int, list[dict]] = {}
        for m in self._measurements:
            bs = m["batch_size"]
            if bs not in by_batch_size:
                by_batch_size[bs] = []
            by_batch_size[bs].append(m)

        # Compute averages per batch size
        summary = {}
        for batch_size, measurements in by_batch_size.items():
            avg_tps = sum(m["tokens_per_sec"] for m in measurements) / len(measurements)
            avg_rps = sum(m["requests_per_sec"] for m in measurements) / len(measurements)

            summary[f"batch_{batch_size}"] = {
                "avg_tokens_per_sec": avg_tps,
                "avg_requests_per_sec": avg_rps,
                "num_measurements": len(measurements),
            }

        return summary

    def reset(self) -> None:
        """Reset accumulated measurements."""
        self._measurements = []
