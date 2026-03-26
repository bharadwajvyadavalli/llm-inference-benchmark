"""Pareto Analysis: Compute Pareto frontiers and recommend strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """A point in the Pareto analysis."""

    strategy: str
    objectives: dict[str, float]
    is_pareto_optimal: bool
    dominates: list[str]
    dominated_by: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy,
            "objectives": self.objectives,
            "is_pareto_optimal": self.is_pareto_optimal,
            "dominates": self.dominates,
            "dominated_by": self.dominated_by,
        }


class ParetoAnalyzer:
    """Analyze Pareto frontiers for multi-objective optimization."""

    def __init__(self) -> None:
        """Initialize Pareto analyzer."""
        self.points: list[ParetoPoint] = []

    def compute_pareto_frontier(
        self,
        results: list[dict[str, Any]],
        objectives: dict[str, str],
    ) -> list[ParetoPoint]:
        """Compute Pareto-optimal set from results.

        Args:
            results: List of benchmark result dictionaries
            objectives: Dictionary mapping objective name to direction
                       ("minimize" or "maximize")

        Returns:
            List of ParetoPoint with Pareto analysis
        """
        if not results:
            return []

        # Extract objective values for each result
        points = []
        for r in results:
            strategy = r.get("strategy", "unknown")
            obj_values = {}

            for obj_name, direction in objectives.items():
                value = self._extract_objective(r, obj_name)
                # Normalize direction: convert all to "minimize"
                if direction == "maximize" and value is not None:
                    value = -value
                obj_values[obj_name] = value

            points.append(
                {
                    "strategy": strategy,
                    "objectives": obj_values,
                    "original_result": r,
                }
            )

        # Compute dominance relationships
        n = len(points)
        is_dominated = [False] * n
        dominates = [[] for _ in range(n)]
        dominated_by = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                if self._dominates(points[i]["objectives"], points[j]["objectives"]):
                    dominates[i].append(points[j]["strategy"])
                    dominated_by[j].append(points[i]["strategy"])
                    is_dominated[j] = True

        # Create ParetoPoint objects
        pareto_points = []
        for i, p in enumerate(points):
            # Convert back objective values to original direction for display
            display_objectives = {}
            for obj_name, value in p["objectives"].items():
                if objectives.get(obj_name) == "maximize" and value is not None:
                    display_objectives[obj_name] = -value
                else:
                    display_objectives[obj_name] = value

            pareto_points.append(
                ParetoPoint(
                    strategy=p["strategy"],
                    objectives=display_objectives,
                    is_pareto_optimal=not is_dominated[i],
                    dominates=dominates[i],
                    dominated_by=dominated_by[i],
                )
            )

        self.points = pareto_points
        return pareto_points

    def _dominates(
        self,
        obj_a: dict[str, float | None],
        obj_b: dict[str, float | None],
    ) -> bool:
        """Check if point A dominates point B.

        A dominates B if A is at least as good in all objectives
        and strictly better in at least one.

        Args:
            obj_a: Objectives for point A
            obj_b: Objectives for point B

        Returns:
            True if A dominates B
        """
        at_least_as_good = True
        strictly_better = False

        for key in obj_a:
            val_a = obj_a.get(key)
            val_b = obj_b.get(key)

            if val_a is None or val_b is None:
                continue

            if val_a > val_b:  # Remember: we normalized to minimize
                at_least_as_good = False
                break
            elif val_a < val_b:
                strictly_better = True

        return at_least_as_good and strictly_better

    def _extract_objective(
        self,
        result: dict[str, Any],
        objective: str,
    ) -> float | None:
        """Extract objective value from result.

        Args:
            result: Benchmark result dictionary
            objective: Objective name (e.g., "latency", "cost", "quality")

        Returns:
            Objective value or None if not found
        """
        # Map common objective names to result fields
        objective_map = {
            "latency": ("latency_metrics", "mean_ms"),
            "ttft": ("latency_metrics", "mean_ms"),
            "throughput": ("throughput_metrics", "tokens_per_sec"),
            "memory": ("memory_metrics", "peak_memory_gb"),
            "quality": ("quality_metrics", "perplexity"),
            "perplexity": ("quality_metrics", "perplexity"),
            "cost": ("cost_metrics", "dollars_per_million_tokens"),
        }

        if objective in objective_map:
            category, field = objective_map[objective]
            return result.get(category, {}).get(field)

        # Try direct access
        for category in ["latency_metrics", "throughput_metrics", "memory_metrics",
                         "quality_metrics", "cost_metrics"]:
            if objective in result.get(category, {}):
                return result[category][objective]

        return None

    def get_pareto_frontier(self) -> list[ParetoPoint]:
        """Get Pareto-optimal points.

        Returns:
            List of Pareto-optimal points
        """
        return [p for p in self.points if p.is_pareto_optimal]

    def recommend_strategy(
        self,
        priority: str = "balanced",
        weights: dict[str, float] | None = None,
    ) -> ParetoPoint | None:
        """Recommend a strategy from the Pareto frontier.

        Args:
            priority: Priority mode ("latency", "throughput", "cost", "quality", "balanced")
            weights: Custom weights for each objective (for balanced mode)

        Returns:
            Recommended ParetoPoint or None
        """
        frontier = self.get_pareto_frontier()
        if not frontier:
            return None

        if priority == "latency":
            return min(frontier, key=lambda p: p.objectives.get("latency", float("inf")))

        elif priority == "throughput":
            return max(frontier, key=lambda p: p.objectives.get("throughput", 0))

        elif priority == "cost":
            return min(frontier, key=lambda p: p.objectives.get("cost", float("inf")))

        elif priority == "quality":
            # Lower perplexity is better
            return min(frontier, key=lambda p: p.objectives.get("quality", float("inf")))

        else:  # balanced
            # Use weighted sum (with default equal weights)
            if weights is None:
                weights = {obj: 1.0 for obj in frontier[0].objectives}

            def score(p: ParetoPoint) -> float:
                total = 0.0
                for obj, weight in weights.items():
                    value = p.objectives.get(obj, 0)
                    if value is not None:
                        # Normalize and weight
                        total += weight * value
                return total

            # For balanced, we want minimum weighted sum (assuming minimization)
            return min(frontier, key=score)

    def compute_hypervolume(
        self,
        reference_point: dict[str, float] | None = None,
    ) -> float:
        """Compute hypervolume indicator for the Pareto frontier.

        The hypervolume is the volume of objective space dominated by
        the Pareto frontier, bounded by a reference point.

        Args:
            reference_point: Reference point for hypervolume calculation

        Returns:
            Hypervolume value
        """
        frontier = self.get_pareto_frontier()
        if not frontier or len(frontier[0].objectives) != 2:
            # Only support 2D for simplicity
            logger.warning("Hypervolume only supported for 2 objectives")
            return 0.0

        # Get objectives
        obj_names = list(frontier[0].objectives.keys())
        if len(obj_names) != 2:
            return 0.0

        # Extract points
        points = []
        for p in frontier:
            x = p.objectives.get(obj_names[0], 0) or 0
            y = p.objectives.get(obj_names[1], 0) or 0
            points.append((x, y))

        # Set reference point if not provided
        if reference_point is None:
            max_x = max(p[0] for p in points) * 1.1
            max_y = max(p[1] for p in points) * 1.1
            ref = (max_x, max_y)
        else:
            ref = (
                reference_point.get(obj_names[0], 0),
                reference_point.get(obj_names[1], 0),
            )

        # Sort points by x coordinate
        sorted_points = sorted(points, key=lambda p: p[0])

        # Compute hypervolume using sweep line
        hypervolume = 0.0
        prev_x = 0.0

        for x, y in sorted_points:
            width = x - prev_x
            height = ref[1] - y
            if width > 0 and height > 0:
                hypervolume += width * height
            prev_x = x

        # Add final rectangle to reference point
        if sorted_points:
            final_x, final_y = sorted_points[-1]
            width = ref[0] - final_x
            height = ref[1] - final_y
            if width > 0 and height > 0:
                hypervolume += width * height

        return hypervolume

    def generate_pareto_report(self) -> str:
        """Generate a text report of Pareto analysis.

        Returns:
            Formatted Pareto analysis report
        """
        lines = []
        lines.append("# Pareto Analysis Report")
        lines.append("")

        frontier = self.get_pareto_frontier()
        lines.append(f"Total strategies analyzed: {len(self.points)}")
        lines.append(f"Pareto-optimal strategies: {len(frontier)}")
        lines.append("")

        lines.append("## Pareto Frontier")
        lines.append("")

        for p in frontier:
            lines.append(f"### {p.strategy}")
            for obj, value in p.objectives.items():
                lines.append(f"  - {obj}: {value:.4f}" if value else f"  - {obj}: N/A")
            if p.dominates:
                lines.append(f"  - Dominates: {', '.join(p.dominates)}")
            lines.append("")

        lines.append("## Dominated Strategies")
        lines.append("")

        dominated = [p for p in self.points if not p.is_pareto_optimal]
        for p in dominated:
            lines.append(f"- **{p.strategy}** (dominated by: {', '.join(p.dominated_by)})")

        lines.append("")
        lines.append("## Recommendations")
        lines.append("")

        for priority in ["latency", "throughput", "cost", "quality", "balanced"]:
            rec = self.recommend_strategy(priority)
            if rec:
                lines.append(f"- Best for **{priority}**: {rec.strategy}")

        return "\n".join(lines)
