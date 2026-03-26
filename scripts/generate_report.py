#!/usr/bin/env python3
"""Generate reports from saved benchmark results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reports from benchmark results",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input results file or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["markdown", "html", "all"],
        help="Output format",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Generate plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Include Pareto analysis",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_results(input_path: str) -> list[dict]:
    """Load results from file or directory."""
    path = Path(input_path)

    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        return data.get("results", [data] if "strategy" in data else [])

    elif path.is_dir():
        results = []
        for json_file in path.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            if "results" in data:
                results.extend(data["results"])
            elif "strategy" in data:
                results.append(data)
        return results

    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Report Generator")
    logger.info("=" * 50)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Format: {args.format}")

    # Load results
    try:
        results = load_results(args.input)
        logger.info(f"Loaded {len(results)} results")
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return 1

    if not results:
        logger.error("No results found")
        return 1

    try:
        from analysis.report import ReportGenerator
        from analysis.plots import PlotGenerator
        from analysis.pareto import ParetoAnalyzer
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate reports
    report_gen = ReportGenerator(output_dir=output_dir)

    if args.format in ["markdown", "all"]:
        md_path = report_gen.save_markdown(results)
        logger.info(f"Markdown report: {md_path}")

    if args.format in ["html", "all"]:
        html_path = report_gen.save_html(results)
        logger.info(f"HTML report: {html_path}")

    # Generate plots
    if args.plots and not args.no_plots:
        logger.info("Generating plots...")
        plot_gen = PlotGenerator(output_dir=output_dir / "plots")
        plots = plot_gen.generate_all_plots(results)

        for name, path in plots.items():
            logger.info(f"  {name}: {path}")

    # Pareto analysis
    if args.pareto:
        logger.info("Performing Pareto analysis...")
        pareto = ParetoAnalyzer()

        # Define objectives
        objectives = {
            "latency": "minimize",
            "throughput": "maximize",
            "quality": "minimize",  # perplexity, lower is better
            "cost": "minimize",
        }

        pareto.compute_pareto_frontier(results, objectives)
        frontier = pareto.get_pareto_frontier()

        logger.info(f"Pareto-optimal strategies: {len(frontier)}")
        for p in frontier:
            logger.info(f"  - {p.strategy}")

        # Save Pareto report
        pareto_report = pareto.generate_pareto_report()
        pareto_path = output_dir / "pareto_analysis.md"
        with open(pareto_path, "w") as f:
            f.write(pareto_report)
        logger.info(f"Pareto analysis: {pareto_path}")

        # Recommendations
        logger.info("\nRecommendations:")
        for priority in ["latency", "throughput", "cost", "quality", "balanced"]:
            rec = pareto.recommend_strategy(priority)
            if rec:
                logger.info(f"  Best for {priority}: {rec.strategy}")

    logger.info("\nReport generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
