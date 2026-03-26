"""Memory Metrics: Peak GPU memory, KV cache memory, memory efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryResult:
    """Container for memory measurement results."""

    peak_memory_bytes: int
    peak_memory_gb: float
    allocated_memory_bytes: int
    allocated_memory_gb: float
    reserved_memory_bytes: int
    reserved_memory_gb: float
    kv_cache_memory_bytes: int
    kv_cache_memory_gb: float
    model_memory_bytes: int
    model_memory_gb: float
    memory_efficiency: float  # useful / total

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_gb": self.peak_memory_gb,
            "allocated_memory_bytes": self.allocated_memory_bytes,
            "allocated_memory_gb": self.allocated_memory_gb,
            "reserved_memory_bytes": self.reserved_memory_bytes,
            "reserved_memory_gb": self.reserved_memory_gb,
            "kv_cache_memory_bytes": self.kv_cache_memory_bytes,
            "kv_cache_memory_gb": self.kv_cache_memory_gb,
            "model_memory_bytes": self.model_memory_bytes,
            "model_memory_gb": self.model_memory_gb,
            "memory_efficiency": self.memory_efficiency,
        }


class MemoryMetrics:
    """Compute memory metrics from GPU profiler data.

    Metrics computed:
    - Peak GPU memory
    - KV cache memory footprint
    - Model weights memory
    - Memory efficiency ratio
    - Memory vs. sequence length curves
    """

    def __init__(self) -> None:
        """Initialize memory metrics collector."""
        self._measurements: list[dict[str, Any]] = []

    def compute(
        self,
        gpu_stats: dict[int, Any],
    ) -> dict[str, Any]:
        """Compute memory metrics from GPU profiler stats.

        Args:
            gpu_stats: Dictionary mapping device ID to GPUStats

        Returns:
            Dictionary with memory metrics
        """
        if not gpu_stats:
            return self._empty_result()

        # Aggregate across all GPUs
        total_peak_allocated = 0
        total_peak_reserved = 0

        for device_id, stats in gpu_stats.items():
            if hasattr(stats, "peak_memory_allocated_bytes"):
                total_peak_allocated += stats.peak_memory_allocated_bytes
                total_peak_reserved += stats.peak_memory_reserved_bytes
            elif isinstance(stats, dict):
                total_peak_allocated += stats.get("peak_memory_allocated_bytes", 0)
                total_peak_reserved += stats.get("peak_memory_reserved_bytes", 0)

        # Estimate efficiency (allocated vs reserved)
        efficiency = (
            total_peak_allocated / total_peak_reserved
            if total_peak_reserved > 0
            else 0
        )

        result = MemoryResult(
            peak_memory_bytes=total_peak_allocated,
            peak_memory_gb=total_peak_allocated / (1024**3),
            allocated_memory_bytes=total_peak_allocated,
            allocated_memory_gb=total_peak_allocated / (1024**3),
            reserved_memory_bytes=total_peak_reserved,
            reserved_memory_gb=total_peak_reserved / (1024**3),
            kv_cache_memory_bytes=0,  # Would need separate tracking
            kv_cache_memory_gb=0.0,
            model_memory_bytes=0,  # Would need separate tracking
            model_memory_gb=0.0,
            memory_efficiency=efficiency,
        )

        return result.to_dict()

    def estimate_kv_cache_memory(
        self,
        model_config: dict[str, Any],
        sequence_length: int,
        batch_size: int,
        dtype_bytes: int = 2,  # FP16
    ) -> dict[str, float]:
        """Estimate KV cache memory requirements.

        Args:
            model_config: Model configuration dictionary
            sequence_length: Maximum sequence length
            batch_size: Batch size
            dtype_bytes: Bytes per element (2 for FP16, 4 for FP32)

        Returns:
            Dictionary with KV cache memory estimates
        """
        # Extract model parameters
        num_layers = model_config.get("num_layers", 32)
        num_kv_heads = model_config.get("num_kv_heads", model_config.get("num_heads", 32))
        head_dim = model_config.get(
            "head_dim", model_config.get("hidden_size", 4096) // model_config.get("num_heads", 32)
        )

        # KV cache: 2 (K+V) * layers * kv_heads * head_dim * seq_len * batch * dtype
        kv_cache_bytes = (
            2  # K and V
            * num_layers
            * num_kv_heads
            * head_dim
            * sequence_length
            * batch_size
            * dtype_bytes
        )

        return {
            "kv_cache_bytes": kv_cache_bytes,
            "kv_cache_gb": kv_cache_bytes / (1024**3),
            "kv_cache_mb": kv_cache_bytes / (1024**2),
            "per_token_bytes": (
                2 * num_layers * num_kv_heads * head_dim * dtype_bytes
            ),
            "sequence_length": sequence_length,
            "batch_size": batch_size,
        }

    def estimate_model_memory(
        self,
        num_params: int,
        dtype_bytes: int = 2,
    ) -> dict[str, float]:
        """Estimate model weights memory requirements.

        Args:
            num_params: Number of model parameters
            dtype_bytes: Bytes per parameter (2 for FP16, 4 for FP32, 0.5 for INT4)

        Returns:
            Dictionary with model memory estimates
        """
        model_bytes = num_params * dtype_bytes

        return {
            "model_bytes": model_bytes,
            "model_gb": model_bytes / (1024**3),
            "model_mb": model_bytes / (1024**2),
            "num_params": num_params,
            "dtype_bytes": dtype_bytes,
        }

    def compute_memory_breakdown(
        self,
        total_memory_bytes: int,
        model_memory_bytes: int,
        kv_cache_bytes: int,
    ) -> dict[str, Any]:
        """Compute memory breakdown by category.

        Args:
            total_memory_bytes: Total GPU memory used
            model_memory_bytes: Memory for model weights
            kv_cache_bytes: Memory for KV cache

        Returns:
            Dictionary with memory breakdown
        """
        known_memory = model_memory_bytes + kv_cache_bytes
        other_memory = max(0, total_memory_bytes - known_memory)

        total = total_memory_bytes if total_memory_bytes > 0 else 1

        return {
            "model_weights": {
                "bytes": model_memory_bytes,
                "gb": model_memory_bytes / (1024**3),
                "percentage": model_memory_bytes / total * 100,
            },
            "kv_cache": {
                "bytes": kv_cache_bytes,
                "gb": kv_cache_bytes / (1024**3),
                "percentage": kv_cache_bytes / total * 100,
            },
            "other": {
                "bytes": other_memory,
                "gb": other_memory / (1024**3),
                "percentage": other_memory / total * 100,
            },
            "total": {
                "bytes": total_memory_bytes,
                "gb": total_memory_bytes / (1024**3),
            },
        }

    def compute_memory_vs_sequence_length(
        self,
        model_config: dict[str, Any],
        sequence_lengths: list[int],
        batch_size: int = 1,
    ) -> dict[int, dict[str, float]]:
        """Compute memory requirements across sequence lengths.

        Args:
            model_config: Model configuration
            sequence_lengths: List of sequence lengths to analyze
            batch_size: Batch size

        Returns:
            Dictionary mapping sequence length to memory estimates
        """
        results = {}

        for seq_len in sequence_lengths:
            kv_cache = self.estimate_kv_cache_memory(
                model_config, seq_len, batch_size
            )
            results[seq_len] = {
                "kv_cache_gb": kv_cache["kv_cache_gb"],
                "kv_cache_bytes": kv_cache["kv_cache_bytes"],
            }

        return results

    def _empty_result(self) -> dict[str, float]:
        """Return empty result dictionary."""
        return {
            "peak_memory_bytes": 0,
            "peak_memory_gb": 0.0,
            "allocated_memory_bytes": 0,
            "allocated_memory_gb": 0.0,
            "reserved_memory_bytes": 0,
            "reserved_memory_gb": 0.0,
            "kv_cache_memory_bytes": 0,
            "kv_cache_memory_gb": 0.0,
            "model_memory_bytes": 0,
            "model_memory_gb": 0.0,
            "memory_efficiency": 0.0,
        }

    def add_measurement(
        self,
        sequence_length: int,
        batch_size: int,
        peak_memory_bytes: int,
        allocated_memory_bytes: int,
    ) -> None:
        """Add a memory measurement.

        Args:
            sequence_length: Sequence length for this measurement
            batch_size: Batch size for this measurement
            peak_memory_bytes: Peak memory usage
            allocated_memory_bytes: Allocated memory
        """
        self._measurements.append(
            {
                "sequence_length": sequence_length,
                "batch_size": batch_size,
                "peak_memory_bytes": peak_memory_bytes,
                "peak_memory_gb": peak_memory_bytes / (1024**3),
                "allocated_memory_bytes": allocated_memory_bytes,
                "allocated_memory_gb": allocated_memory_bytes / (1024**3),
            }
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all memory measurements.

        Returns:
            Dictionary with aggregated memory statistics
        """
        if not self._measurements:
            return {}

        max_peak = max(m["peak_memory_bytes"] for m in self._measurements)
        avg_peak = sum(m["peak_memory_bytes"] for m in self._measurements) / len(
            self._measurements
        )

        return {
            "max_peak_memory_gb": max_peak / (1024**3),
            "avg_peak_memory_gb": avg_peak / (1024**3),
            "num_measurements": len(self._measurements),
        }

    def reset(self) -> None:
        """Reset accumulated measurements."""
        self._measurements = []
