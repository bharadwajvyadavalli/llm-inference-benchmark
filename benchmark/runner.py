"""BenchmarkRunner: Orchestrates all benchmark suites."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import torch
import yaml

from benchmark.profiler import GPUProfiler
from benchmark.workloads import WorkloadGenerator
from metrics.latency import LatencyMetrics
from metrics.memory import MemoryMetrics
from metrics.throughput import ThroughputMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    warmup_runs: int = 3
    measurement_runs: int = 10
    save_intermediate: bool = True
    output_dir: str = "results"
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    strategy: str
    model: str
    workload: str
    backend: str
    timestamp: str
    config: dict[str, Any]
    latency_metrics: dict[str, float] = field(default_factory=dict)
    throughput_metrics: dict[str, float] = field(default_factory=dict)
    memory_metrics: dict[str, float] = field(default_factory=dict)
    quality_metrics: dict[str, float] = field(default_factory=dict)
    cost_metrics: dict[str, float] = field(default_factory=dict)
    raw_timings: list[float] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy": self.strategy,
            "model": self.model,
            "workload": self.workload,
            "backend": self.backend,
            "timestamp": self.timestamp,
            "config": self.config,
            "latency_metrics": self.latency_metrics,
            "throughput_metrics": self.throughput_metrics,
            "memory_metrics": self.memory_metrics,
            "quality_metrics": self.quality_metrics,
            "cost_metrics": self.cost_metrics,
            "raw_timings": self.raw_timings,
            "environment": self.environment,
        }


class BenchmarkRunner:
    """Orchestrates benchmark execution across strategies, models, and workloads."""

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            config: BenchmarkConfig instance
            config_path: Path to YAML config file (used if config is None)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = BenchmarkConfig.from_yaml(config_path)
        else:
            self.config = BenchmarkConfig()

        self.profiler = GPUProfiler()
        self.workload_generator = WorkloadGenerator(seed=self.config.random_seed)
        self.results: list[BenchmarkResult] = []

        # Set random seeds for reproducibility
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

    def _get_environment_info(self) -> dict[str, Any]:
        """Collect environment information for reproducibility."""
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            env_info.update(
                {
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_names": [
                        torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())
                    ],
                    "gpu_memory": [
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count())
                    ],
                }
            )

        return env_info

    def _sync_cuda(self) -> None:
        """Synchronize CUDA for accurate timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_single(
        self,
        strategy: Any,
        model: Any,
        workload: Any,
        backend: Any,
    ) -> BenchmarkResult:
        """Run a single benchmark configuration.

        Args:
            strategy: Strategy instance to benchmark
            model: Model configuration
            workload: Workload to run
            backend: Serving backend instance

        Returns:
            BenchmarkResult with all collected metrics
        """
        logger.info(
            f"Running benchmark: strategy={strategy.name}, "
            f"model={model}, workload={workload.name}, backend={backend.name}"
        )

        result = BenchmarkResult(
            strategy=strategy.name,
            model=str(model),
            workload=workload.name,
            backend=backend.name,
            timestamp=datetime.now().isoformat(),
            config={
                "warmup_runs": self.config.warmup_runs,
                "measurement_runs": self.config.measurement_runs,
            },
            environment=self._get_environment_info(),
        )

        # Load model with strategy applied
        try:
            backend.load_model(model, strategy.get_config())
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            result.latency_metrics["error"] = str(e)
            return result

        # Generate workload prompts
        prompts = workload.generate()

        # Warmup runs
        logger.info(f"Running {self.config.warmup_runs} warmup iterations")
        for _ in range(self.config.warmup_runs):
            self._sync_cuda()
            try:
                backend.generate(prompts[:10])  # Use subset for warmup
            except Exception as e:
                logger.warning(f"Warmup run failed: {e}")
            self._sync_cuda()

        # Clear GPU cache before measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Measurement runs
        logger.info(f"Running {self.config.measurement_runs} measurement iterations")
        timings: list[float] = []
        token_counts: list[int] = []

        with self.profiler:
            for i in range(self.config.measurement_runs):
                self._sync_cuda()
                start_time = time.perf_counter()

                try:
                    outputs = backend.generate(prompts)
                    self._sync_cuda()
                    end_time = time.perf_counter()

                    elapsed = end_time - start_time
                    timings.append(elapsed)

                    # Count generated tokens
                    total_tokens = sum(len(out.get("tokens", [])) for out in outputs)
                    token_counts.append(total_tokens)

                    logger.debug(f"Run {i + 1}: {elapsed:.3f}s, {total_tokens} tokens")

                except Exception as e:
                    logger.error(f"Measurement run {i + 1} failed: {e}")
                    timings.append(float("inf"))
                    token_counts.append(0)

        # Collect metrics
        result.raw_timings = timings

        # Latency metrics
        latency_collector = LatencyMetrics()
        result.latency_metrics = latency_collector.compute(timings)

        # Throughput metrics
        throughput_collector = ThroughputMetrics()
        result.throughput_metrics = throughput_collector.compute(
            timings, token_counts, len(prompts)
        )

        # Memory metrics
        memory_collector = MemoryMetrics()
        result.memory_metrics = memory_collector.compute(self.profiler.get_memory_stats())

        return result

    def run_suite(
        self,
        strategies: list[Any],
        models: list[Any],
        workloads: list[Any],
        backends: list[Any],
    ) -> list[BenchmarkResult]:
        """Run full benchmark suite across all combinations.

        Args:
            strategies: List of strategy instances
            models: List of model configurations
            workloads: List of workload instances
            backends: List of backend instances

        Returns:
            List of BenchmarkResult for all combinations
        """
        total_combinations = len(strategies) * len(models) * len(workloads) * len(backends)
        logger.info(f"Running benchmark suite with {total_combinations} combinations")

        results: list[BenchmarkResult] = []

        for i, (strategy, model, workload, backend) in enumerate(
            product(strategies, models, workloads, backends)
        ):
            logger.info(f"Combination {i + 1}/{total_combinations}")

            try:
                result = self.run_single(strategy, model, workload, backend)
                results.append(result)

                # Save intermediate results
                if self.config.save_intermediate:
                    self._save_intermediate(results)

            except Exception as e:
                logger.error(f"Benchmark combination failed: {e}")
                # Create error result
                results.append(
                    BenchmarkResult(
                        strategy=strategy.name,
                        model=str(model),
                        workload=workload.name,
                        backend=backend.name,
                        timestamp=datetime.now().isoformat(),
                        config={"error": str(e)},
                    )
                )

        self.results = results
        return results

    def _save_intermediate(self, results: list[BenchmarkResult]) -> None:
        """Save intermediate results to JSON file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "intermediate_results.json"
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        logger.debug(f"Saved intermediate results to {output_path}")

    def save_results(self, filename: str | None = None) -> Path:
        """Save all results to JSON file.

        Args:
            filename: Output filename (default: results_<timestamp>.json)

        Returns:
            Path to saved file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "config": {
                            "warmup_runs": self.config.warmup_runs,
                            "measurement_runs": self.config.measurement_runs,
                            "random_seed": self.config.random_seed,
                        },
                        "environment": self._get_environment_info(),
                    },
                    "results": [r.to_dict() for r in self.results],
                },
                f,
                indent=2,
            )

        logger.info(f"Saved results to {output_path}")
        return output_path

    @classmethod
    def load_results(cls, path: str | Path) -> list[BenchmarkResult]:
        """Load results from JSON file.

        Args:
            path: Path to results JSON file

        Returns:
            List of BenchmarkResult
        """
        with open(path) as f:
            data = json.load(f)

        results = []
        for r in data.get("results", []):
            results.append(
                BenchmarkResult(
                    strategy=r["strategy"],
                    model=r["model"],
                    workload=r["workload"],
                    backend=r["backend"],
                    timestamp=r["timestamp"],
                    config=r["config"],
                    latency_metrics=r.get("latency_metrics", {}),
                    throughput_metrics=r.get("throughput_metrics", {}),
                    memory_metrics=r.get("memory_metrics", {}),
                    quality_metrics=r.get("quality_metrics", {}),
                    cost_metrics=r.get("cost_metrics", {}),
                    raw_timings=r.get("raw_timings", []),
                    environment=r.get("environment", {}),
                )
            )

        return results
