#!/usr/bin/env python3
"""Main CLI for running LLM inference benchmarks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LLM inference benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark suite
  python run_benchmark.py --strategies all --models all --workloads all --backends all

  # Run selective benchmark
  python run_benchmark.py --strategies speculative,quantization --models mistral-7b --workloads short

  # Run with specific backend
  python run_benchmark.py --models llama-7b --backends vllm --output results/
        """,
    )

    parser.add_argument(
        "--strategies",
        type=str,
        default="baseline",
        help="Strategies to benchmark (comma-separated or 'all')",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="pythia-70m",
        help="Models to benchmark (comma-separated or 'all')",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default="short",
        help="Workloads to run (comma-separated or 'all'): short,long,batch,multi_turn",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="hf",
        help="Backends to use (comma-separated or 'all'): hf,vllm,tgi",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_config.yaml",
        help="Path to benchmark configuration file",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=None,
        help="Number of warmup runs (overrides config)",
    )
    parser.add_argument(
        "--measurement-runs",
        type=int,
        default=None,
        help="Number of measurement runs (overrides config)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running benchmarks",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from YAML."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def parse_list(value: str, available: list[str]) -> list[str]:
    """Parse comma-separated list or 'all'."""
    if value.lower() == "all":
        return available
    return [v.strip() for v in value.split(",")]


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("LLM Inference Benchmark")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI args
    if args.warmup_runs is not None:
        config["warmup_runs"] = args.warmup_runs
    if args.measurement_runs is not None:
        config["measurement_runs"] = args.measurement_runs
    config["output_dir"] = args.output

    # Available options
    available_strategies = [
        "baseline",
        "speculative",
        "quantization_int8",
        "quantization_int4",
        "quantization_gptq",
        "quantization_awq",
        "kv_cache_paged",
        "kv_cache_sliding",
        "batching_static",
        "batching_dynamic",
        "batching_continuous",
        "attention_flash",
        "attention_sdpa",
    ]
    available_models = [
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "phi-2",
        "llama-2-7b",
        "llama-2-13b",
        "mistral-7b",
    ]
    available_workloads = ["short", "long", "batch", "multi_turn"]
    available_backends = ["hf", "vllm", "tgi"]

    # Parse selections
    strategies = parse_list(args.strategies, available_strategies)
    models = parse_list(args.models, available_models)
    workloads = parse_list(args.workloads, available_workloads)
    backends = parse_list(args.backends, available_backends)

    logger.info(f"Strategies: {strategies}")
    logger.info(f"Models: {models}")
    logger.info(f"Workloads: {workloads}")
    logger.info(f"Backends: {backends}")
    logger.info(f"Output directory: {args.output}")

    if args.dry_run:
        logger.info("Dry run - not executing benchmarks")
        total = len(strategies) * len(models) * len(workloads) * len(backends)
        logger.info(f"Would run {total} benchmark combinations")
        return 0

    # Import benchmark components
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
        from analysis.report import ReportGenerator
    except ImportError as e:
        logger.error(f"Failed to import benchmark modules: {e}")
        logger.error("Make sure you have installed the package: pip install -e .")
        return 1

    # Create benchmark configuration
    benchmark_config = BenchmarkConfig(
        warmup_runs=config.get("warmup_runs", 3),
        measurement_runs=config.get("measurement_runs", 10),
        save_intermediate=config.get("save_intermediate", True),
        output_dir=config.get("output_dir", "results"),
        random_seed=config.get("random_seed", 42),
    )

    # Create runner
    runner = BenchmarkRunner(config=benchmark_config)

    # Create strategy instances
    strategy_instances = []
    for s in strategies:
        if s == "baseline":
            strategy_instances.append(BaselineStrategy())
        elif s == "speculative":
            strategy_instances.append(SpeculativeDecodingStrategy())
        elif s.startswith("quantization"):
            method = s.replace("quantization_", "")
            strategy_instances.append(QuantizationStrategy(method=method))
        elif s.startswith("kv_cache"):
            method = s.replace("kv_cache_", "")
            strategy_instances.append(KVCacheStrategy(method=method))
        elif s.startswith("batching"):
            method = s.replace("batching_", "")
            strategy_instances.append(BatchingStrategy(method=method))
        elif s.startswith("attention"):
            method = s.replace("attention_", "")
            strategy_instances.append(AttentionStrategy(method=method))

    # Create workload instances
    workload_gen = WorkloadGenerator(seed=benchmark_config.random_seed)
    workload_instances = [workload_gen.create(w) for w in workloads]

    # Create backend instances
    backend_instances = []
    for b in backends:
        if b == "hf":
            backend_instances.append(HuggingFaceBackend())
        elif b == "vllm":
            backend_instances.append(VLLMBackend())
        elif b == "tgi":
            backend_instances.append(TGIBackend())

    # Run benchmarks
    logger.info("Starting benchmark suite...")
    try:
        results = runner.run_suite(
            strategies=strategy_instances,
            models=models,
            workloads=workload_instances,
            backends=backend_instances,
        )

        # Save results
        results_path = runner.save_results()
        logger.info(f"Results saved to: {results_path}")

        # Generate reports
        report_gen = ReportGenerator(output_dir=args.output)
        md_path = report_gen.save_markdown([r.to_dict() for r in results])
        html_path = report_gen.save_html([r.to_dict() for r in results])

        logger.info(f"Markdown report: {md_path}")
        logger.info(f"HTML report: {html_path}")

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    logger.info("Benchmark completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
