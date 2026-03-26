"""Cost Metrics: $/1M tokens, tokens per GPU-hour, cost efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CostResult:
    """Container for cost measurement results."""

    dollars_per_million_tokens: float
    tokens_per_gpu_hour: float
    gpu_cost_per_hour: float
    cost_efficiency_score: float
    pareto_optimal: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dollars_per_million_tokens": self.dollars_per_million_tokens,
            "tokens_per_gpu_hour": self.tokens_per_gpu_hour,
            "gpu_cost_per_hour": self.gpu_cost_per_hour,
            "cost_efficiency_score": self.cost_efficiency_score,
            "pareto_optimal": self.pareto_optimal,
        }


# Default GPU costs ($/hour) for common GPUs
DEFAULT_GPU_COSTS = {
    "a100_40gb": 3.00,
    "a100_80gb": 4.50,
    "h100_80gb": 8.00,
    "h100_sxm": 10.00,
    "rtx_4090": 1.50,
    "rtx_3090": 0.80,
    "v100_16gb": 1.20,
    "v100_32gb": 1.50,
    "default": 2.00,
}


class CostMetrics:
    """Compute cost metrics for LLM inference.

    Metrics computed:
    - $/1M tokens
    - Tokens per GPU-hour
    - Cost-quality Pareto frontier
    """

    def __init__(
        self,
        gpu_cost_per_hour: float | None = None,
        gpu_type: str = "default",
    ) -> None:
        """Initialize cost metrics collector.

        Args:
            gpu_cost_per_hour: GPU rental cost in $/hour
            gpu_type: GPU type for default cost lookup
        """
        if gpu_cost_per_hour is not None:
            self.gpu_cost_per_hour = gpu_cost_per_hour
        else:
            self.gpu_cost_per_hour = DEFAULT_GPU_COSTS.get(
                gpu_type.lower(), DEFAULT_GPU_COSTS["default"]
            )

        self._measurements: list[dict[str, Any]] = []

    def compute(
        self,
        tokens_per_second: float,
        num_gpus: int = 1,
        gpu_cost_per_hour: float | None = None,
    ) -> dict[str, float]:
        """Compute cost metrics from throughput data.

        Args:
            tokens_per_second: Generation throughput
            num_gpus: Number of GPUs used
            gpu_cost_per_hour: Override GPU cost (optional)

        Returns:
            Dictionary with cost metrics
        """
        if gpu_cost_per_hour is None:
            gpu_cost_per_hour = self.gpu_cost_per_hour

        total_gpu_cost = gpu_cost_per_hour * num_gpus

        # Tokens per GPU-hour
        tokens_per_hour = tokens_per_second * 3600
        tokens_per_gpu_hour = tokens_per_hour / num_gpus if num_gpus > 0 else 0

        # Cost per million tokens
        if tokens_per_hour > 0:
            cost_per_token = total_gpu_cost / tokens_per_hour
            cost_per_million = cost_per_token * 1_000_000
        else:
            cost_per_million = float("inf")

        # Cost efficiency score (higher is better)
        # Normalized to tokens per dollar
        tokens_per_dollar = tokens_per_hour / total_gpu_cost if total_gpu_cost > 0 else 0

        result = CostResult(
            dollars_per_million_tokens=cost_per_million,
            tokens_per_gpu_hour=tokens_per_gpu_hour,
            gpu_cost_per_hour=total_gpu_cost,
            cost_efficiency_score=tokens_per_dollar,
            pareto_optimal=False,  # Determined later
        )

        return result.to_dict()

    def compute_with_quality(
        self,
        tokens_per_second: float,
        quality_score: float,
        num_gpus: int = 1,
    ) -> dict[str, float]:
        """Compute cost metrics including quality consideration.

        Args:
            tokens_per_second: Generation throughput
            quality_score: Quality score (0-100)
            num_gpus: Number of GPUs used

        Returns:
            Dictionary with cost-quality metrics
        """
        base_metrics = self.compute(tokens_per_second, num_gpus)

        # Quality-adjusted cost efficiency
        # Higher quality and lower cost both improve the score
        quality_factor = quality_score / 100
        cost_factor = 1 / (base_metrics["dollars_per_million_tokens"] + 0.001)

        quality_cost_score = quality_factor * cost_factor * 100

        base_metrics.update(
            {
                "quality_score": quality_score,
                "quality_cost_score": quality_cost_score,
            }
        )

        return base_metrics

    def estimate_monthly_cost(
        self,
        tokens_per_day: int,
        tokens_per_second: float,
        num_gpus: int = 1,
        hours_per_day: float = 24,
    ) -> dict[str, float]:
        """Estimate monthly infrastructure cost.

        Args:
            tokens_per_day: Daily token generation requirement
            tokens_per_second: Current throughput
            num_gpus: Number of GPUs
            hours_per_day: Operating hours per day

        Returns:
            Dictionary with monthly cost estimates
        """
        # How many hours needed to generate required tokens?
        tokens_per_hour = tokens_per_second * 3600
        hours_needed = tokens_per_day / tokens_per_hour if tokens_per_hour > 0 else 24

        # Daily GPU hours (capped at operating hours)
        daily_gpu_hours = min(hours_needed, hours_per_day) * num_gpus

        # Monthly costs (30 days)
        monthly_gpu_hours = daily_gpu_hours * 30
        monthly_cost = monthly_gpu_hours * self.gpu_cost_per_hour

        # Monthly tokens generated
        monthly_tokens = tokens_per_hour * min(hours_needed, hours_per_day) * 30

        return {
            "hours_needed_per_day": hours_needed,
            "daily_gpu_hours": daily_gpu_hours,
            "monthly_gpu_hours": monthly_gpu_hours,
            "monthly_cost_usd": monthly_cost,
            "monthly_tokens": monthly_tokens,
            "cost_per_million_tokens": (
                monthly_cost / monthly_tokens * 1_000_000 if monthly_tokens > 0 else 0
            ),
        }

    def compare_strategies(
        self,
        strategy_results: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Compare cost metrics across strategies.

        Args:
            strategy_results: Dictionary mapping strategy name to metrics

        Returns:
            Dictionary with comparison analysis
        """
        if not strategy_results:
            return {}

        # Find best and worst for each metric
        strategies = list(strategy_results.keys())

        costs = {s: r.get("dollars_per_million_tokens", float("inf")) for s, r in strategy_results.items()}
        throughputs = {s: r.get("tokens_per_gpu_hour", 0) for s, r in strategy_results.items()}

        lowest_cost = min(costs.values())
        highest_throughput = max(throughputs.values())

        comparison = {
            "strategies": strategies,
            "lowest_cost_strategy": min(costs, key=costs.get),
            "highest_throughput_strategy": max(throughputs, key=throughputs.get),
            "lowest_cost": lowest_cost,
            "highest_throughput": highest_throughput,
            "strategy_rankings": {},
        }

        # Rank strategies by cost efficiency
        sorted_by_cost = sorted(strategies, key=lambda s: costs[s])
        for rank, strategy in enumerate(sorted_by_cost, 1):
            comparison["strategy_rankings"][strategy] = {
                "cost_rank": rank,
                "cost": costs[strategy],
                "throughput": throughputs[strategy],
                "cost_vs_best": costs[strategy] / lowest_cost if lowest_cost > 0 else 0,
                "throughput_vs_best": (
                    throughputs[strategy] / highest_throughput if highest_throughput > 0 else 0
                ),
            }

        return comparison

    def add_measurement(
        self,
        strategy: str,
        tokens_per_second: float,
        quality_score: float,
        num_gpus: int = 1,
    ) -> None:
        """Add a cost measurement.

        Args:
            strategy: Strategy name
            tokens_per_second: Throughput
            quality_score: Quality score
            num_gpus: Number of GPUs
        """
        metrics = self.compute_with_quality(tokens_per_second, quality_score, num_gpus)
        metrics["strategy"] = strategy

        self._measurements.append(metrics)

    def get_pareto_frontier(self) -> list[dict[str, Any]]:
        """Identify Pareto-optimal strategies.

        Returns strategies that are not dominated on cost-quality tradeoff.

        Returns:
            List of Pareto-optimal measurement dictionaries
        """
        if not self._measurements:
            return []

        # Sort by cost (lower is better)
        sorted_measurements = sorted(
            self._measurements, key=lambda m: m["dollars_per_million_tokens"]
        )

        pareto_frontier = []
        max_quality_seen = -float("inf")

        for measurement in sorted_measurements:
            quality = measurement.get("quality_score", 0)

            # A point is Pareto-optimal if it has higher quality than
            # all points with lower cost
            if quality > max_quality_seen:
                pareto_frontier.append(measurement)
                max_quality_seen = quality

        # Mark Pareto-optimal strategies
        pareto_strategies = {m["strategy"] for m in pareto_frontier}
        for measurement in self._measurements:
            measurement["pareto_optimal"] = measurement["strategy"] in pareto_strategies

        return pareto_frontier

    def recommend_strategy(
        self,
        priority: str = "cost",
    ) -> dict[str, Any] | None:
        """Recommend best strategy based on priority.

        Args:
            priority: What to optimize ("cost", "quality", or "balanced")

        Returns:
            Recommended strategy metrics or None if no measurements
        """
        pareto = self.get_pareto_frontier()
        if not pareto:
            return None

        if priority == "cost":
            # Lowest cost from Pareto frontier
            return min(pareto, key=lambda m: m["dollars_per_million_tokens"])

        elif priority == "quality":
            # Highest quality from Pareto frontier
            return max(pareto, key=lambda m: m.get("quality_score", 0))

        else:  # balanced
            # Best quality-cost score
            return max(pareto, key=lambda m: m.get("quality_cost_score", 0))

    def get_summary(self) -> dict[str, Any]:
        """Get summary of cost measurements.

        Returns:
            Dictionary with cost statistics
        """
        if not self._measurements:
            return {}

        costs = [m["dollars_per_million_tokens"] for m in self._measurements]
        throughputs = [m["tokens_per_gpu_hour"] for m in self._measurements]

        return {
            "num_strategies": len(self._measurements),
            "avg_cost_per_million": sum(costs) / len(costs),
            "min_cost_per_million": min(costs),
            "max_cost_per_million": max(costs),
            "avg_tokens_per_gpu_hour": sum(throughputs) / len(throughputs),
            "pareto_optimal_count": sum(1 for m in self._measurements if m.get("pareto_optimal")),
        }

    def reset(self) -> None:
        """Reset accumulated measurements."""
        self._measurements = []
