#!/usr/bin/env python3
"""Head-to-head comparison between two strategies with statistical significance."""

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
        description="Compare two strategies head-to-head",
    )

    parser.add_argument(
        "--strategy-a",
        type=str,
        required=True,
        help="First strategy to compare",
    )
    parser.add_argument(
        "--strategy-b",
        type=str,
        required=True,
        help="Second strategy to compare",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pythia-70m",
        help="Model to benchmark",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "vllm", "tgi"],
        help="Backend to use",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="short",
        help="Workload type",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of measurement runs for statistical significance",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for significance testing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def create_strategy(name: str):
    """Create strategy instance from name."""
    from strategies.base import BaselineStrategy
    from strategies.quantization import QuantizationStrategy
    from strategies.speculative_decoding import SpeculativeDecodingStrategy
    from strategies.kv_cache import KVCacheStrategy
    from strategies.batching import BatchingStrategy
    from strategies.attention import AttentionStrategy

    name_lower = name.lower()

    if name_lower == "baseline":
        return BaselineStrategy()
    elif name_lower.startswith("quantization"):
        method = name_lower.replace("quantization_", "") or "int8"
        return QuantizationStrategy(method=method)
    elif name_lower == "speculative":
        return SpeculativeDecodingStrategy()
    elif name_lower.startswith("kv_cache"):
        method = name_lower.replace("kv_cache_", "") or "paged_attention"
        return KVCacheStrategy(method=method)
    elif name_lower.startswith("batching"):
        method = name_lower.replace("batching_", "") or "static"
        return BatchingStrategy(method=method)
    elif name_lower.startswith("attention"):
        method = name_lower.replace("attention_", "") or "flash_attention_2"
        return AttentionStrategy(method=method)
    else:
        raise ValueError(f"Unknown strategy: {name}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Strategy Comparison")
    logger.info("=" * 50)
    logger.info(f"Strategy A: {args.strategy_a}")
    logger.info(f"Strategy B: {args.strategy_b}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Runs: {args.runs}")
    logger.info(f"Confidence: {args.confidence}")

    try:
        from benchmark.runner import BenchmarkRunner, BenchmarkConfig
        from benchmark.workloads import WorkloadGenerator
        from serving.hf_backend import HuggingFaceBackend
        from serving.vllm_backend import VLLMBackend
        from serving.tgi_backend import TGIBackend
        from analysis.comparisons import StrategyComparison
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create strategies
    try:
        strategy_a = create_strategy(args.strategy_a)
        strategy_b = create_strategy(args.strategy_b)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Create backend
    if args.backend == "hf":
        backend = HuggingFaceBackend()
    elif args.backend == "vllm":
        backend = VLLMBackend()
    elif args.backend == "tgi":
        backend = TGIBackend()
    else:
        logger.error(f"Unknown backend: {args.backend}")
        return 1

    # Create workload
    workload_gen = WorkloadGenerator()
    workload = workload_gen.create(args.workload)

    # Create runner with more runs for statistical significance
    config = BenchmarkConfig(
        warmup_runs=5,
        measurement_runs=args.runs,
        output_dir=str(output_dir),
    )
    runner = BenchmarkRunner(config=config)

    # Run benchmarks for both strategies
    logger.info(f"\nBenchmarking {args.strategy_a}...")
    result_a = runner.run_single(
        strategy=strategy_a,
        model=args.model,
        workload=workload,
        backend=backend,
    )

    logger.info(f"\nBenchmarking {args.strategy_b}...")
    result_b = runner.run_single(
        strategy=strategy_b,
        model=args.model,
        workload=workload,
        backend=backend,
    )

    # Perform statistical comparison
    logger.info("\nPerforming statistical comparison...")
    comparator = StrategyComparison(confidence_level=args.confidence)

    comparisons = comparator.full_comparison(
        result_a.to_dict(),
        result_b.to_dict(),
    )

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 50)

    for metric, comparison in comparisons.items():
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  {args.strategy_a}: {comparison.mean_a:.4f} (std: {comparison.std_a:.4f})")
        logger.info(f"  {args.strategy_b}: {comparison.mean_b:.4f} (std: {comparison.std_b:.4f})")
        logger.info(f"  Difference: {comparison.difference:.4f} ({comparison.difference_pct:+.2f}%)")
        logger.info(f"  p-value: {comparison.p_value:.4f}")

        if comparison.statistically_significant:
            logger.info(f"  WINNER: {comparison.winner} (p < {1 - args.confidence})")
        else:
            logger.info(f"  No statistically significant difference (p >= {1 - args.confidence})")

    # Generate report
    report = comparator.generate_comparison_report(comparisons)
    report_path = output_dir / f"comparison_{args.strategy_a}_vs_{args.strategy_b}.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"\nReport saved to: {report_path}")

    # Save raw results
    results_path = output_dir / f"comparison_{args.strategy_a}_vs_{args.strategy_b}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "strategy_a": result_a.to_dict(),
                "strategy_b": result_b.to_dict(),
                "comparisons": {k: v.to_dict() for k, v in comparisons.items()},
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
