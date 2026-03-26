#!/usr/bin/env python3
"""Run benchmark for a single strategy in isolation."""

from __future__ import annotations

import argparse
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
        description="Benchmark a single strategy in isolation",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy to benchmark (e.g., baseline, quantization, speculative)",
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
        choices=["short", "long", "batch", "multi_turn"],
        help="Workload type",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory",
    )

    # Strategy-specific options
    parser.add_argument(
        "--quantization-method",
        type=str,
        default="int8",
        help="Quantization method (for quantization strategy)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Draft model for speculative decoding",
    )
    parser.add_argument(
        "--draft-length",
        type=int,
        default=5,
        help="Draft length for speculative decoding",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--measurement-runs",
        type=int,
        default=10,
        help="Number of measurement runs",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info(f"Single Strategy Benchmark: {args.strategy}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Workload: {args.workload}")

    try:
        from benchmark.runner import BenchmarkRunner, BenchmarkConfig
        from benchmark.workloads import WorkloadGenerator
        from strategies.base import BaselineStrategy
        from strategies.quantization import QuantizationStrategy
        from strategies.speculative_decoding import SpeculativeDecodingStrategy
        from strategies.kv_cache import KVCacheStrategy
        from strategies.batching import BatchingStrategy
        from strategies.attention import AttentionStrategy
        from serving.hf_backend import HuggingFaceBackend
        from serving.vllm_backend import VLLMBackend
        from serving.tgi_backend import TGIBackend
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        return 1

    # Create strategy
    strategy_name = args.strategy.lower()
    if strategy_name == "baseline":
        strategy = BaselineStrategy()
    elif strategy_name == "quantization":
        strategy = QuantizationStrategy(method=args.quantization_method)
    elif strategy_name == "speculative":
        strategy = SpeculativeDecodingStrategy(
            draft_model_name=args.draft_model,
            draft_length=args.draft_length,
        )
    elif strategy_name == "kv_cache":
        strategy = KVCacheStrategy()
    elif strategy_name == "batching":
        strategy = BatchingStrategy()
    elif strategy_name == "attention":
        strategy = AttentionStrategy()
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
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

    # Create runner
    config = BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
        output_dir=args.output,
    )
    runner = BenchmarkRunner(config=config)

    # Run benchmark
    logger.info("Starting benchmark...")
    try:
        result = runner.run_single(
            strategy=strategy,
            model=args.model,
            workload=workload,
            backend=backend,
        )

        # Print results
        logger.info("=" * 50)
        logger.info("Results:")
        logger.info(f"  Strategy: {result.strategy}")
        logger.info(f"  Model: {result.model}")
        logger.info(f"  Workload: {result.workload}")
        logger.info(f"  Backend: {result.backend}")
        logger.info("")
        logger.info("Latency:")
        for k, v in result.latency_metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        logger.info("")
        logger.info("Throughput:")
        for k, v in result.throughput_metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        logger.info("")
        logger.info("Memory:")
        for k, v in result.memory_metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # Save result
        runner.results = [result]
        results_path = runner.save_results(f"single_{strategy_name}_{args.model}.json")
        logger.info(f"Results saved to: {results_path}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
