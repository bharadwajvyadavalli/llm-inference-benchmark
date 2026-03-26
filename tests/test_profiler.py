"""Tests for GPU profiler."""

import pytest
from unittest.mock import MagicMock, patch


class TestGPUProfiler:
    """Tests for GPUProfiler class."""

    def test_profiler_init(self):
        """Test profiler initialization."""
        from benchmark.profiler import GPUProfiler

        profiler = GPUProfiler(sampling_interval_ms=100)
        assert profiler.sampling_interval_ms == 100
        assert not profiler._running

    def test_profiler_context_manager(self):
        """Test profiler as context manager."""
        from benchmark.profiler import GPUProfiler

        profiler = GPUProfiler()

        with profiler:
            assert profiler._running or len(profiler.device_ids) == 0

        assert not profiler._running

    @patch("torch.cuda.is_available", return_value=False)
    def test_profiler_no_gpu(self, mock_cuda):
        """Test profiler behavior when no GPU is available."""
        from benchmark.profiler import GPUProfiler

        profiler = GPUProfiler()
        assert profiler.device_ids == []

        stats = profiler.get_memory_stats()
        assert stats == {}

    def test_memory_breakdown(self):
        """Test memory breakdown estimation."""
        from benchmark.profiler import GPUProfiler, MemoryBreakdown

        profiler = GPUProfiler()

        # Test with mock values
        breakdown = MemoryBreakdown(
            model_weights_bytes=1000,
            kv_cache_bytes=500,
            activations_bytes=200,
            other_bytes=100,
            total_bytes=1800,
        )

        result = breakdown.to_dict()
        assert result["model_weights_bytes"] == 1000
        assert result["kv_cache_bytes"] == 500
        assert result["total_bytes"] == 1800

    def test_gpu_snapshot_dataclass(self):
        """Test GPUSnapshot dataclass."""
        from benchmark.profiler import GPUSnapshot

        snapshot = GPUSnapshot(
            timestamp=1234567890.0,
            device_id=0,
            memory_allocated_bytes=1024,
            memory_reserved_bytes=2048,
            memory_free_bytes=4096,
            utilization_percent=50.0,
            power_watts=100.0,
        )

        assert snapshot.device_id == 0
        assert snapshot.memory_allocated_bytes == 1024
        assert snapshot.utilization_percent == 50.0

    def test_gpu_stats_to_dict(self):
        """Test GPUStats conversion to dictionary."""
        from benchmark.profiler import GPUStats

        stats = GPUStats(
            device_id=0,
            peak_memory_allocated_bytes=1024 * 1024 * 1024,  # 1 GB
            peak_memory_reserved_bytes=2 * 1024 * 1024 * 1024,  # 2 GB
        )

        result = stats.to_dict()
        assert result["device_id"] == 0
        assert result["peak_memory_allocated_gb"] == pytest.approx(1.0)
        assert result["peak_memory_reserved_gb"] == pytest.approx(2.0)

    def test_profiler_reset(self):
        """Test profiler reset functionality."""
        from benchmark.profiler import GPUProfiler

        profiler = GPUProfiler()
        profiler._snapshots = {0: [MagicMock()]}

        profiler.reset()
        assert profiler._snapshots == {d: [] for d in profiler.device_ids}
