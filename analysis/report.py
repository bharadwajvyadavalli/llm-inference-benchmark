"""Report Generator: Create markdown and HTML benchmark reports."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate benchmark reports in Markdown and HTML formats."""

    def __init__(self, output_dir: str | Path = "reports") -> None:
        """Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown(
        self,
        results: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate a Markdown report from benchmark results.

        Args:
            results: List of benchmark result dictionaries
            metadata: Optional metadata about the benchmark run

        Returns:
            Markdown report string
        """
        lines = []

        # Header
        lines.append("# LLM Inference Benchmark Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Metadata section
        if metadata:
            lines.append("## Benchmark Configuration")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for key, value in metadata.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")

        # Environment info
        if results and results[0].get("environment"):
            env = results[0]["environment"]
            lines.append("## Environment")
            lines.append("")
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for key, value in env.items():
                if not isinstance(value, (list, dict)):
                    lines.append(f"| {key} | {value} |")
            lines.append("")

        # Summary table
        lines.append("## Results Summary")
        lines.append("")
        lines.append(self._create_summary_table(results))
        lines.append("")

        # Detailed results by strategy
        lines.append("## Detailed Results")
        lines.append("")

        strategies = set(r.get("strategy", "unknown") for r in results)
        for strategy in sorted(strategies):
            strategy_results = [r for r in results if r.get("strategy") == strategy]
            lines.append(f"### {strategy}")
            lines.append("")
            lines.append(self._create_strategy_table(strategy_results))
            lines.append("")

        # Key findings
        lines.append("## Key Findings")
        lines.append("")
        findings = self._generate_findings(results)
        for finding in findings:
            lines.append(f"- {finding}")
        lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        recommendations = self._generate_recommendations(results)
        for rec in recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        return "\n".join(lines)

    def generate_html(
        self,
        results: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        include_plots: bool = True,
    ) -> str:
        """Generate an HTML report with interactive charts.

        Args:
            results: List of benchmark result dictionaries
            metadata: Optional metadata about the benchmark run
            include_plots: Whether to include Plotly charts

        Returns:
            HTML report string
        """
        html_parts = []

        # HTML header
        html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LLM Inference Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #444; margin-top: 30px; }
        h3 { color: #555; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th { background-color: #f8f9fa; font-weight: 600; }
        tr:hover { background-color: #f5f5f5; }
        .metric-card {
            display: inline-block;
            padding: 15px 25px;
            margin: 10px;
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            border-radius: 8px;
        }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { font-size: 12px; opacity: 0.9; }
        .chart-container { margin: 30px 0; }
        .finding { padding: 10px; margin: 5px 0; background: #e8f4fd; border-left: 4px solid #007bff; }
        .recommendation { padding: 10px; margin: 5px 0; background: #e8fde8; border-left: 4px solid #28a745; }
    </style>
</head>
<body>
<div class="container">
""")

        # Title and timestamp
        html_parts.append(f"""
    <h1>LLM Inference Benchmark Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
""")

        # Summary metrics cards
        if results:
            summary = self._compute_summary_stats(results)
            html_parts.append("""
    <h2>Summary</h2>
    <div class="metric-cards">
""")
            for label, value in summary.items():
                html_parts.append(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
""")
            html_parts.append("    </div>")

        # Results table
        html_parts.append("""
    <h2>Results</h2>
""")
        html_parts.append(self._create_html_table(results))

        # Plotly charts
        if include_plots:
            html_parts.append("""
    <h2>Visualizations</h2>
    <div class="chart-container" id="latency-chart"></div>
    <div class="chart-container" id="throughput-chart"></div>
    <div class="chart-container" id="memory-chart"></div>
""")
            html_parts.append(self._generate_plotly_scripts(results))

        # Findings and recommendations
        html_parts.append("""
    <h2>Key Findings</h2>
""")
        for finding in self._generate_findings(results):
            html_parts.append(f'    <div class="finding">{finding}</div>\n')

        html_parts.append("""
    <h2>Recommendations</h2>
""")
        for rec in self._generate_recommendations(results):
            html_parts.append(f'    <div class="recommendation">{rec}</div>\n')

        # Close HTML
        html_parts.append("""
</div>
</body>
</html>
""")

        return "".join(html_parts)

    def _create_summary_table(self, results: list[dict[str, Any]]) -> str:
        """Create a Markdown summary table."""
        if not results:
            return "No results available."

        lines = []
        lines.append("| Strategy | Model | TTFT (ms) | Tokens/sec | Memory (GB) | Perplexity |")
        lines.append("|----------|-------|-----------|------------|-------------|------------|")

        for r in results:
            strategy = r.get("strategy", "N/A")
            model = r.get("model", "N/A")
            latency = r.get("latency_metrics", {})
            throughput = r.get("throughput_metrics", {})
            memory = r.get("memory_metrics", {})
            quality = r.get("quality_metrics", {})

            ttft = latency.get("mean_ms", latency.get("p50_ms", "-"))
            tps = throughput.get("tokens_per_sec", "-")
            mem = memory.get("peak_memory_gb", "-")
            ppl = quality.get("perplexity", "-")

            ttft_str = f"{ttft:.2f}" if isinstance(ttft, (int, float)) else ttft
            tps_str = f"{tps:.2f}" if isinstance(tps, (int, float)) else tps
            mem_str = f"{mem:.2f}" if isinstance(mem, (int, float)) else mem
            ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else ppl

            lines.append(f"| {strategy} | {model} | {ttft_str} | {tps_str} | {mem_str} | {ppl_str} |")

        return "\n".join(lines)

    def _create_strategy_table(self, results: list[dict[str, Any]]) -> str:
        """Create detailed table for a single strategy."""
        if not results:
            return "No results."

        lines = []
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        # Aggregate metrics from first result (or average if multiple)
        r = results[0]

        for category in ["latency_metrics", "throughput_metrics", "memory_metrics"]:
            metrics = r.get(category, {})
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"| {category}.{key} | {value:.4f} |")
                else:
                    lines.append(f"| {category}.{key} | {value} |")

        return "\n".join(lines)

    def _create_html_table(self, results: list[dict[str, Any]]) -> str:
        """Create HTML results table."""
        html = ["<table>"]
        html.append("<thead><tr>")
        html.append("<th>Strategy</th><th>Model</th><th>Workload</th>")
        html.append("<th>TTFT (ms)</th><th>Tokens/sec</th><th>Memory (GB)</th>")
        html.append("</tr></thead>")
        html.append("<tbody>")

        for r in results:
            latency = r.get("latency_metrics", {})
            throughput = r.get("throughput_metrics", {})
            memory = r.get("memory_metrics", {})

            html.append("<tr>")
            html.append(f"<td>{r.get('strategy', 'N/A')}</td>")
            html.append(f"<td>{r.get('model', 'N/A')}</td>")
            html.append(f"<td>{r.get('workload', 'N/A')}</td>")
            html.append(f"<td>{latency.get('mean_ms', '-'):.2f}</td>")
            html.append(f"<td>{throughput.get('tokens_per_sec', '-'):.2f}</td>")
            html.append(f"<td>{memory.get('peak_memory_gb', '-'):.2f}</td>")
            html.append("</tr>")

        html.append("</tbody></table>")
        return "\n".join(html)

    def _generate_plotly_scripts(self, results: list[dict[str, Any]]) -> str:
        """Generate Plotly JavaScript for charts."""
        strategies = [r.get("strategy", "unknown") for r in results]
        latencies = [r.get("latency_metrics", {}).get("mean_ms", 0) for r in results]
        throughputs = [r.get("throughput_metrics", {}).get("tokens_per_sec", 0) for r in results]
        memories = [r.get("memory_metrics", {}).get("peak_memory_gb", 0) for r in results]

        script = f"""
<script>
// Latency Chart
Plotly.newPlot('latency-chart', [{{
    x: {json.dumps(strategies)},
    y: {json.dumps(latencies)},
    type: 'bar',
    marker: {{ color: '#007bff' }}
}}], {{
    title: 'Latency by Strategy (ms)',
    xaxis: {{ title: 'Strategy' }},
    yaxis: {{ title: 'Latency (ms)' }}
}});

// Throughput Chart
Plotly.newPlot('throughput-chart', [{{
    x: {json.dumps(strategies)},
    y: {json.dumps(throughputs)},
    type: 'bar',
    marker: {{ color: '#28a745' }}
}}], {{
    title: 'Throughput by Strategy (tokens/sec)',
    xaxis: {{ title: 'Strategy' }},
    yaxis: {{ title: 'Tokens/sec' }}
}});

// Memory Chart
Plotly.newPlot('memory-chart', [{{
    x: {json.dumps(strategies)},
    y: {json.dumps(memories)},
    type: 'bar',
    marker: {{ color: '#dc3545' }}
}}], {{
    title: 'Peak Memory by Strategy (GB)',
    xaxis: {{ title: 'Strategy' }},
    yaxis: {{ title: 'Memory (GB)' }}
}});
</script>
"""
        return script

    def _compute_summary_stats(self, results: list[dict[str, Any]]) -> dict[str, str]:
        """Compute summary statistics."""
        if not results:
            return {}

        strategies = set(r.get("strategy") for r in results)
        models = set(r.get("model") for r in results)

        latencies = [r.get("latency_metrics", {}).get("mean_ms", 0) for r in results]
        throughputs = [r.get("throughput_metrics", {}).get("tokens_per_sec", 0) for r in results]

        return {
            "Strategies Tested": str(len(strategies)),
            "Models Tested": str(len(models)),
            "Best Latency": f"{min(latencies):.2f} ms" if latencies else "N/A",
            "Best Throughput": f"{max(throughputs):.2f} tok/s" if throughputs else "N/A",
        }

    def _generate_findings(self, results: list[dict[str, Any]]) -> list[str]:
        """Generate key findings from results."""
        if not results:
            return ["No results to analyze."]

        findings = []

        # Find best strategy for each metric
        best_latency = min(
            results, key=lambda r: r.get("latency_metrics", {}).get("mean_ms", float("inf"))
        )
        best_throughput = max(
            results, key=lambda r: r.get("throughput_metrics", {}).get("tokens_per_sec", 0)
        )
        best_memory = min(
            results, key=lambda r: r.get("memory_metrics", {}).get("peak_memory_gb", float("inf"))
        )

        findings.append(
            f"**Best latency**: {best_latency.get('strategy')} "
            f"({best_latency.get('latency_metrics', {}).get('mean_ms', 0):.2f} ms)"
        )
        findings.append(
            f"**Best throughput**: {best_throughput.get('strategy')} "
            f"({best_throughput.get('throughput_metrics', {}).get('tokens_per_sec', 0):.2f} tokens/sec)"
        )
        findings.append(
            f"**Lowest memory**: {best_memory.get('strategy')} "
            f"({best_memory.get('memory_metrics', {}).get('peak_memory_gb', 0):.2f} GB)"
        )

        return findings

    def _generate_recommendations(self, results: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations based on results."""
        if not results:
            return ["Run benchmarks to get recommendations."]

        recommendations = []

        # Analyze results and provide recommendations
        strategies = set(r.get("strategy") for r in results)

        if "speculative_decoding" in strategies:
            recommendations.append(
                "Consider speculative decoding for latency-sensitive applications."
            )

        if any("quantization" in s for s in strategies if s):
            recommendations.append(
                "Quantization provides good memory-throughput tradeoffs. "
                "Evaluate quality impact for your use case."
            )

        recommendations.append(
            "Run production workloads to validate benchmark findings."
        )

        return recommendations

    def save_markdown(
        self,
        results: list[dict[str, Any]],
        filename: str = "benchmark_report.md",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save Markdown report to file.

        Args:
            results: Benchmark results
            filename: Output filename
            metadata: Optional metadata

        Returns:
            Path to saved file
        """
        content = self.generate_markdown(results, metadata)
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write(content)

        logger.info(f"Saved Markdown report to {output_path}")
        return output_path

    def save_html(
        self,
        results: list[dict[str, Any]],
        filename: str = "benchmark_report.html",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save HTML report to file.

        Args:
            results: Benchmark results
            filename: Output filename
            metadata: Optional metadata

        Returns:
            Path to saved file
        """
        content = self.generate_html(results, metadata)
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write(content)

        logger.info(f"Saved HTML report to {output_path}")
        return output_path
