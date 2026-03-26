"""Plot Generator: Create visualizations for benchmark results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generate plots and visualizations for benchmark results."""

    def __init__(self, output_dir: str | Path = "reports/plots") -> None:
        """Initialize plot generator.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Try to import plotting libraries
        self._matplotlib_available = False
        self._plotly_available = False

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            self.plt = plt
            self.sns = sns
            self._matplotlib_available = True
        except ImportError:
            logger.warning("matplotlib/seaborn not available. Some plots disabled.")

        try:
            import plotly.express as px
            import plotly.graph_objects as go

            self.px = px
            self.go = go
            self._plotly_available = True
        except ImportError:
            logger.warning("plotly not available. Interactive plots disabled.")

    def latency_comparison_chart(
        self,
        results: list[dict[str, Any]],
        filename: str = "latency_comparison",
    ) -> Path | None:
        """Create grouped bar chart comparing latency across strategies.

        Args:
            results: Benchmark results
            filename: Output filename (without extension)

        Returns:
            Path to saved plot or None if failed
        """
        if not self._matplotlib_available:
            return None

        strategies = [r.get("strategy", "unknown") for r in results]
        ttft = [r.get("latency_metrics", {}).get("mean_ms", 0) for r in results]
        p95 = [r.get("latency_metrics", {}).get("p95_ms", 0) for r in results]
        p99 = [r.get("latency_metrics", {}).get("p99_ms", 0) for r in results]

        fig, ax = self.plt.subplots(figsize=(12, 6))

        x = range(len(strategies))
        width = 0.25

        bars1 = ax.bar([i - width for i in x], ttft, width, label="Mean", color="#007bff")
        bars2 = ax.bar(x, p95, width, label="P95", color="#28a745")
        bars3 = ax.bar([i + width for i in x], p99, width, label="P99", color="#dc3545")

        ax.set_xlabel("Strategy")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Comparison by Strategy")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.legend()

        self.plt.tight_layout()

        # Save as PNG
        png_path = self.output_dir / f"{filename}.png"
        self.plt.savefig(png_path, dpi=150)
        self.plt.close()

        # Also save as interactive Plotly HTML if available
        if self._plotly_available:
            self._save_plotly_latency(results, filename)

        return png_path

    def _save_plotly_latency(
        self, results: list[dict[str, Any]], filename: str
    ) -> None:
        """Save interactive Plotly latency chart."""
        import pandas as pd

        data = []
        for r in results:
            latency = r.get("latency_metrics", {})
            data.append(
                {
                    "Strategy": r.get("strategy", "unknown"),
                    "Mean": latency.get("mean_ms", 0),
                    "P95": latency.get("p95_ms", 0),
                    "P99": latency.get("p99_ms", 0),
                }
            )

        df = pd.DataFrame(data)
        fig = self.px.bar(
            df,
            x="Strategy",
            y=["Mean", "P95", "P99"],
            barmode="group",
            title="Latency Comparison by Strategy",
            labels={"value": "Latency (ms)", "variable": "Metric"},
        )

        html_path = self.output_dir / f"{filename}.html"
        fig.write_html(str(html_path))

    def throughput_scaling_chart(
        self,
        results: dict[int, dict[str, float]],
        filename: str = "throughput_scaling",
    ) -> Path | None:
        """Create line chart showing throughput vs batch size.

        Args:
            results: Dictionary mapping batch_size to metrics
            filename: Output filename

        Returns:
            Path to saved plot or None
        """
        if not self._matplotlib_available:
            return None

        batch_sizes = sorted(results.keys())
        throughputs = [results[bs].get("tokens_per_sec", 0) for bs in batch_sizes]
        ideal = [results[batch_sizes[0]].get("tokens_per_sec", 0) * bs for bs in batch_sizes]

        fig, ax = self.plt.subplots(figsize=(10, 6))

        ax.plot(batch_sizes, throughputs, "o-", label="Actual", color="#007bff", linewidth=2)
        ax.plot(batch_sizes, ideal, "--", label="Ideal (linear)", color="#aaa", linewidth=1)

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("Throughput Scaling with Batch Size")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.plt.tight_layout()

        png_path = self.output_dir / f"{filename}.png"
        self.plt.savefig(png_path, dpi=150)
        self.plt.close()

        return png_path

    def memory_breakdown_chart(
        self,
        breakdown: dict[str, dict[str, float]],
        filename: str = "memory_breakdown",
    ) -> Path | None:
        """Create stacked bar chart showing memory breakdown.

        Args:
            breakdown: Memory breakdown by category
            filename: Output filename

        Returns:
            Path to saved plot or None
        """
        if not self._matplotlib_available:
            return None

        categories = ["model_weights", "kv_cache", "other"]
        values = [breakdown.get(cat, {}).get("gb", 0) for cat in categories]
        colors = ["#007bff", "#28a745", "#aaa"]

        fig, ax = self.plt.subplots(figsize=(8, 6))

        ax.bar(categories, values, color=colors)
        ax.set_xlabel("Category")
        ax.set_ylabel("Memory (GB)")
        ax.set_title("GPU Memory Breakdown")

        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + 0.1, f"{v:.2f} GB", ha="center")

        self.plt.tight_layout()

        png_path = self.output_dir / f"{filename}.png"
        self.plt.savefig(png_path, dpi=150)
        self.plt.close()

        return png_path

    def quality_vs_speedup_scatter(
        self,
        results: list[dict[str, Any]],
        filename: str = "quality_vs_speedup",
    ) -> Path | None:
        """Create scatter plot of quality vs speedup.

        Args:
            results: Benchmark results with quality and speedup metrics
            filename: Output filename

        Returns:
            Path to saved plot or None
        """
        if not self._matplotlib_available:
            return None

        strategies = []
        speedups = []
        qualities = []

        for r in results:
            strategy = r.get("strategy", "unknown")
            quality = r.get("quality_metrics", {}).get("perplexity", None)
            throughput = r.get("throughput_metrics", {}).get("tokens_per_sec", 0)

            if quality is not None and throughput > 0:
                strategies.append(strategy)
                qualities.append(quality)
                speedups.append(throughput)

        if not strategies:
            logger.warning("No data for quality vs speedup plot")
            return None

        # Normalize speedup to first result
        baseline = speedups[0] if speedups else 1
        speedups = [s / baseline for s in speedups]

        fig, ax = self.plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(speedups, qualities, s=100, c=range(len(strategies)), cmap="viridis")

        # Add labels
        for i, strategy in enumerate(strategies):
            ax.annotate(
                strategy,
                (speedups[i], qualities[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax.set_xlabel("Relative Speedup")
        ax.set_ylabel("Perplexity (lower is better)")
        ax.set_title("Quality vs Speedup Tradeoff")
        ax.grid(True, alpha=0.3)

        # Ideal point indicator
        ax.axhline(y=min(qualities), color="g", linestyle="--", alpha=0.5, label="Best quality")
        ax.axvline(x=max(speedups), color="b", linestyle="--", alpha=0.5, label="Best speedup")

        self.plt.tight_layout()

        png_path = self.output_dir / f"{filename}.png"
        self.plt.savefig(png_path, dpi=150)
        self.plt.close()

        return png_path

    def pareto_frontier_plot(
        self,
        results: list[dict[str, Any]],
        x_metric: str = "cost",
        y_metric: str = "quality",
        filename: str = "pareto_frontier",
    ) -> Path | None:
        """Create Pareto frontier visualization.

        Args:
            results: Benchmark results
            x_metric: Metric for X axis
            y_metric: Metric for Y axis
            filename: Output filename

        Returns:
            Path to saved plot or None
        """
        if not self._matplotlib_available:
            return None

        # Extract metrics
        strategies = []
        x_values = []
        y_values = []
        is_pareto = []

        for r in results:
            strategy = r.get("strategy", "unknown")

            if x_metric == "cost":
                x = r.get("cost_metrics", {}).get("dollars_per_million_tokens", 0)
            elif x_metric == "latency":
                x = r.get("latency_metrics", {}).get("mean_ms", 0)
            else:
                x = r.get("throughput_metrics", {}).get("tokens_per_sec", 0)

            if y_metric == "quality":
                y = 100 - r.get("quality_metrics", {}).get("perplexity", 0)  # Invert for "higher is better"
            else:
                y = r.get("throughput_metrics", {}).get("tokens_per_sec", 0)

            strategies.append(strategy)
            x_values.append(x)
            y_values.append(y)
            is_pareto.append(r.get("pareto_optimal", False))

        fig, ax = self.plt.subplots(figsize=(10, 8))

        # Plot non-Pareto points
        non_pareto_x = [x for x, p in zip(x_values, is_pareto) if not p]
        non_pareto_y = [y for y, p in zip(y_values, is_pareto) if not p]
        ax.scatter(non_pareto_x, non_pareto_y, s=80, c="gray", alpha=0.5, label="Dominated")

        # Plot Pareto points
        pareto_x = [x for x, p in zip(x_values, is_pareto) if p]
        pareto_y = [y for y, p in zip(y_values, is_pareto) if p]
        ax.scatter(pareto_x, pareto_y, s=120, c="green", label="Pareto Optimal")

        # Draw Pareto frontier line
        if pareto_x:
            sorted_pareto = sorted(zip(pareto_x, pareto_y))
            pareto_x_sorted = [p[0] for p in sorted_pareto]
            pareto_y_sorted = [p[1] for p in sorted_pareto]
            ax.plot(pareto_x_sorted, pareto_y_sorted, "g--", alpha=0.7)

        # Add labels
        for i, strategy in enumerate(strategies):
            ax.annotate(
                strategy,
                (x_values[i], y_values[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax.set_xlabel(f"{x_metric.title()}")
        ax.set_ylabel(f"{y_metric.title()} Score")
        ax.set_title(f"Pareto Frontier: {y_metric.title()} vs {x_metric.title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.plt.tight_layout()

        png_path = self.output_dir / f"{filename}.png"
        self.plt.savefig(png_path, dpi=150)
        self.plt.close()

        return png_path

    def generate_all_plots(
        self,
        results: list[dict[str, Any]],
        batch_scaling: dict[int, dict[str, float]] | None = None,
        memory_breakdown: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Path]:
        """Generate all available plots.

        Args:
            results: Benchmark results
            batch_scaling: Optional batch scaling data
            memory_breakdown: Optional memory breakdown data

        Returns:
            Dictionary mapping plot name to path
        """
        plots = {}

        # Latency comparison
        path = self.latency_comparison_chart(results)
        if path:
            plots["latency_comparison"] = path

        # Throughput scaling (if data provided)
        if batch_scaling:
            path = self.throughput_scaling_chart(batch_scaling)
            if path:
                plots["throughput_scaling"] = path

        # Memory breakdown (if data provided)
        if memory_breakdown:
            path = self.memory_breakdown_chart(memory_breakdown)
            if path:
                plots["memory_breakdown"] = path

        # Quality vs speedup
        path = self.quality_vs_speedup_scatter(results)
        if path:
            plots["quality_vs_speedup"] = path

        # Pareto frontier
        path = self.pareto_frontier_plot(results)
        if path:
            plots["pareto_frontier"] = path

        return plots
