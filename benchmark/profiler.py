"""GPUProfiler: Track GPU memory, utilization, and power draw."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Try to import pynvml for detailed GPU monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available. GPU power/utilization monitoring disabled.")


@dataclass
class GPUSnapshot:
    """Snapshot of GPU state at a point in time."""

    timestamp: float
    device_id: int
    memory_allocated_bytes: int
    memory_reserved_bytes: int
    memory_free_bytes: int
    utilization_percent: float | None = None
    power_watts: float | None = None
    temperature_celsius: float | None = None


@dataclass
class MemoryBreakdown:
    """Breakdown of GPU memory usage by category."""

    model_weights_bytes: int = 0
    kv_cache_bytes: int = 0
    activations_bytes: int = 0
    optimizer_states_bytes: int = 0
    other_bytes: int = 0
    total_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "model_weights_bytes": self.model_weights_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "activations_bytes": self.activations_bytes,
            "optimizer_states_bytes": self.optimizer_states_bytes,
            "other_bytes": self.other_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass
class GPUStats:
    """Aggregated GPU statistics over a profiling session."""

    device_id: int = 0
    peak_memory_allocated_bytes: int = 0
    peak_memory_reserved_bytes: int = 0
    avg_utilization_percent: float | None = None
    max_utilization_percent: float | None = None
    avg_power_watts: float | None = None
    max_power_watts: float | None = None
    max_temperature_celsius: float | None = None
    snapshots: list[GPUSnapshot] = field(default_factory=list)
    memory_breakdown: MemoryBreakdown | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "peak_memory_allocated_bytes": self.peak_memory_allocated_bytes,
            "peak_memory_reserved_bytes": self.peak_memory_reserved_bytes,
            "peak_memory_allocated_gb": self.peak_memory_allocated_bytes / (1024**3),
            "peak_memory_reserved_gb": self.peak_memory_reserved_bytes / (1024**3),
            "avg_utilization_percent": self.avg_utilization_percent,
            "max_utilization_percent": self.max_utilization_percent,
            "avg_power_watts": self.avg_power_watts,
            "max_power_watts": self.max_power_watts,
            "max_temperature_celsius": self.max_temperature_celsius,
            "num_snapshots": len(self.snapshots),
            "memory_breakdown": (
                self.memory_breakdown.to_dict() if self.memory_breakdown else None
            ),
        }


class GPUProfiler:
    """Profile GPU memory, utilization, and power consumption.

    Can be used as a context manager:
        with GPUProfiler() as profiler:
            # Run workload
            pass
        stats = profiler.get_memory_stats()

    Or manually:
        profiler = GPUProfiler()
        profiler.start()
        # Run workload
        profiler.stop()
        stats = profiler.get_memory_stats()
    """

    def __init__(
        self,
        device_ids: list[int] | None = None,
        sampling_interval_ms: int = 100,
        track_power: bool = True,
        track_utilization: bool = True,
    ) -> None:
        """Initialize the GPU profiler.

        Args:
            device_ids: List of GPU device IDs to profile (default: all available)
            sampling_interval_ms: Interval between snapshots in milliseconds
            track_power: Whether to track power consumption (requires pynvml)
            track_utilization: Whether to track GPU utilization (requires pynvml)
        """
        self.sampling_interval_ms = sampling_interval_ms
        self.track_power = track_power and PYNVML_AVAILABLE
        self.track_utilization = track_utilization and PYNVML_AVAILABLE

        # Determine devices to profile
        if device_ids is not None:
            self.device_ids = device_ids
        elif torch.cuda.is_available():
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = []

        self._snapshots: dict[int, list[GPUSnapshot]] = {d: [] for d in self.device_ids}
        self._running = False
        self._thread: threading.Thread | None = None
        self._nvml_initialized = False

    def _init_nvml(self) -> None:
        """Initialize NVML library."""
        if PYNVML_AVAILABLE and not self._nvml_initialized:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")

    def _shutdown_nvml(self) -> None:
        """Shutdown NVML library."""
        if PYNVML_AVAILABLE and self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
            except Exception as e:
                logger.warning(f"Failed to shutdown NVML: {e}")

    def start(self) -> None:
        """Start profiling."""
        if self._running:
            logger.warning("Profiler already running")
            return

        if not self.device_ids:
            logger.warning("No GPU devices available for profiling")
            return

        # Reset state
        self._snapshots = {d: [] for d in self.device_ids}
        self._running = True

        # Initialize NVML if needed
        if self.track_power or self.track_utilization:
            self._init_nvml()

        # Reset CUDA memory stats
        for device_id in self.device_ids:
            torch.cuda.reset_peak_memory_stats(device_id)

        # Start sampling thread
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

        logger.debug("GPU profiler started")

    def stop(self) -> None:
        """Stop profiling."""
        if not self._running:
            return

        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        logger.debug("GPU profiler stopped")

    def snapshot(self) -> dict[int, GPUSnapshot]:
        """Take a snapshot of current GPU state.

        Returns:
            Dictionary mapping device ID to GPUSnapshot
        """
        snapshots = {}

        for device_id in self.device_ids:
            try:
                snapshot = self._take_device_snapshot(device_id)
                snapshots[device_id] = snapshot
                self._snapshots[device_id].append(snapshot)
            except Exception as e:
                logger.warning(f"Failed to snapshot device {device_id}: {e}")

        return snapshots

    def _take_device_snapshot(self, device_id: int) -> GPUSnapshot:
        """Take a snapshot for a single device."""
        timestamp = time.time()

        # Get PyTorch memory stats
        memory_allocated = torch.cuda.memory_allocated(device_id)
        memory_reserved = torch.cuda.memory_reserved(device_id)

        # Get total memory
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory
        memory_free = total_memory - memory_reserved

        snapshot = GPUSnapshot(
            timestamp=timestamp,
            device_id=device_id,
            memory_allocated_bytes=memory_allocated,
            memory_reserved_bytes=memory_reserved,
            memory_free_bytes=memory_free,
        )

        # Get NVML stats if available
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

                if self.track_utilization:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    snapshot.utilization_percent = util.gpu

                if self.track_power:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    snapshot.power_watts = power_mw / 1000.0

                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                snapshot.temperature_celsius = temp

            except Exception as e:
                logger.debug(f"Failed to get NVML stats for device {device_id}: {e}")

        return snapshot

    def _sampling_loop(self) -> None:
        """Background thread for periodic sampling."""
        interval_sec = self.sampling_interval_ms / 1000.0

        while self._running:
            self.snapshot()
            time.sleep(interval_sec)

    def get_memory_stats(self) -> dict[int, GPUStats]:
        """Get aggregated memory statistics.

        Returns:
            Dictionary mapping device ID to GPUStats
        """
        stats = {}

        for device_id in self.device_ids:
            device_stats = GPUStats(device_id=device_id)
            snapshots = self._snapshots.get(device_id, [])

            if snapshots:
                device_stats.snapshots = snapshots

                # Peak memory
                device_stats.peak_memory_allocated_bytes = max(
                    s.memory_allocated_bytes for s in snapshots
                )
                device_stats.peak_memory_reserved_bytes = max(
                    s.memory_reserved_bytes for s in snapshots
                )

                # Utilization stats
                util_values = [
                    s.utilization_percent
                    for s in snapshots
                    if s.utilization_percent is not None
                ]
                if util_values:
                    device_stats.avg_utilization_percent = sum(util_values) / len(
                        util_values
                    )
                    device_stats.max_utilization_percent = max(util_values)

                # Power stats
                power_values = [
                    s.power_watts for s in snapshots if s.power_watts is not None
                ]
                if power_values:
                    device_stats.avg_power_watts = sum(power_values) / len(power_values)
                    device_stats.max_power_watts = max(power_values)

                # Temperature
                temp_values = [
                    s.temperature_celsius
                    for s in snapshots
                    if s.temperature_celsius is not None
                ]
                if temp_values:
                    device_stats.max_temperature_celsius = max(temp_values)

            # Also check PyTorch's peak memory tracking
            peak_allocated = torch.cuda.max_memory_allocated(device_id)
            peak_reserved = torch.cuda.max_memory_reserved(device_id)

            device_stats.peak_memory_allocated_bytes = max(
                device_stats.peak_memory_allocated_bytes, peak_allocated
            )
            device_stats.peak_memory_reserved_bytes = max(
                device_stats.peak_memory_reserved_bytes, peak_reserved
            )

            stats[device_id] = device_stats

        return stats

    def memory_breakdown(
        self,
        model_size_bytes: int | None = None,
        kv_cache_bytes: int | None = None,
    ) -> dict[int, MemoryBreakdown]:
        """Estimate memory breakdown by category.

        Args:
            model_size_bytes: Known model size in bytes (optional)
            kv_cache_bytes: Known KV cache size in bytes (optional)

        Returns:
            Dictionary mapping device ID to MemoryBreakdown
        """
        breakdowns = {}

        for device_id in self.device_ids:
            breakdown = MemoryBreakdown()

            # Get current memory state
            allocated = torch.cuda.memory_allocated(device_id)
            breakdown.total_bytes = allocated

            # If we know model size, use it
            if model_size_bytes is not None:
                breakdown.model_weights_bytes = model_size_bytes

            # If we know KV cache size, use it
            if kv_cache_bytes is not None:
                breakdown.kv_cache_bytes = kv_cache_bytes

            # Estimate other as remainder
            known = breakdown.model_weights_bytes + breakdown.kv_cache_bytes
            breakdown.other_bytes = max(0, breakdown.total_bytes - known)

            breakdowns[device_id] = breakdown

        return breakdowns

    def __enter__(self) -> GPUProfiler:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()
        self._shutdown_nvml()

    def reset(self) -> None:
        """Reset profiler state."""
        self._snapshots = {d: [] for d in self.device_ids}
        for device_id in self.device_ids:
            torch.cuda.reset_peak_memory_stats(device_id)
