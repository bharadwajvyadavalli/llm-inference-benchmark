"""Tests for metrics modules."""

import pytest


class TestLatencyMetrics:
    """Tests for LatencyMetrics class."""

    def test_compute_empty(self):
        """Test compute with empty input."""
        from metrics.latency import LatencyMetrics

        metrics = LatencyMetrics()
        result = metrics.compute([])

        assert result["mean_ms"] == 0.0
        assert result["samples"] == 0

    def test_compute_single_value(self):
        """Test compute with single value."""
        from metrics.latency import LatencyMetrics

        metrics = LatencyMetrics()
        result = metrics.compute([1.0])  # 1 second

        assert result["mean_ms"] == 1000.0
        assert result["samples"] == 1

    def test_compute_multiple_values(self):
        """Test compute with multiple values."""
        from metrics.latency import LatencyMetrics

        metrics = LatencyMetrics()
        # 100ms, 200ms, 300ms in seconds
        timings = [0.1, 0.2, 0.3]
        result = metrics.compute(timings)

        assert result["mean_ms"] == pytest.approx(200.0)
        assert result["min_ms"] == pytest.approx(100.0)
        assert result["max_ms"] == pytest.approx(300.0)
        assert result["samples"] == 3

    def test_percentile_calculation(self):
        """Test percentile calculations."""
        from metrics.latency import LatencyMetrics

        metrics = LatencyMetrics()
        # Create a range of values
        timings = [i / 1000 for i in range(1, 101)]  # 1ms to 100ms
        result = metrics.compute(timings, unit="seconds")

        assert result["p50_ms"] == pytest.approx(50.0, rel=0.1)
        assert result["p95_ms"] == pytest.approx(95.0, rel=0.1)
        assert result["p99_ms"] == pytest.approx(99.0, rel=0.1)

    def test_filter_invalid_timings(self):
        """Test filtering of invalid timing values."""
        from metrics.latency import LatencyMetrics

        metrics = LatencyMetrics()
        timings = [0.1, -0.1, float("inf"), 0.2, 0.0]
        result = metrics.compute(timings)

        # Should only include 0.1 and 0.2
        assert result["samples"] == 2
        assert result["mean_ms"] == pytest.approx(150.0)


class TestThroughputMetrics:
    """Tests for ThroughputMetrics class."""

    def test_compute_empty(self):
        """Test compute with empty input."""
        from metrics.throughput import ThroughputMetrics

        metrics = ThroughputMetrics()
        result = metrics.compute([], [], 0)

        assert result["tokens_per_sec"] == 0.0

    def test_compute_basic(self):
        """Test basic throughput computation."""
        from metrics.throughput import ThroughputMetrics

        metrics = ThroughputMetrics()
        # 100 tokens in 1 second
        result = metrics.compute([1.0], [100], num_requests=1)

        assert result["tokens_per_sec"] == pytest.approx(100.0)
        assert result["requests_per_sec"] == pytest.approx(1.0)

    def test_compute_multiple_runs(self):
        """Test throughput with multiple runs."""
        from metrics.throughput import ThroughputMetrics

        metrics = ThroughputMetrics()
        # Two runs: 100 tokens in 1s, 200 tokens in 2s
        result = metrics.compute([1.0, 2.0], [100, 200], num_requests=10)

        # Average: 150 tokens / 1.5 seconds = 100 tokens/sec
        assert result["tokens_per_sec"] == pytest.approx(100.0)


class TestMemoryMetrics:
    """Tests for MemoryMetrics class."""

    def test_estimate_kv_cache(self):
        """Test KV cache memory estimation."""
        from metrics.memory import MemoryMetrics

        metrics = MemoryMetrics()
        model_config = {
            "num_layers": 32,
            "num_kv_heads": 8,
            "hidden_size": 4096,
            "num_heads": 32,
        }

        result = metrics.estimate_kv_cache_memory(
            model_config,
            sequence_length=2048,
            batch_size=1,
            dtype_bytes=2,
        )

        assert "kv_cache_bytes" in result
        assert "kv_cache_gb" in result
        assert result["kv_cache_bytes"] > 0

    def test_estimate_model_memory(self):
        """Test model memory estimation."""
        from metrics.memory import MemoryMetrics

        metrics = MemoryMetrics()

        # 7B parameters in FP16
        result = metrics.estimate_model_memory(
            num_params=7_000_000_000,
            dtype_bytes=2,
        )

        assert result["model_gb"] == pytest.approx(14.0, rel=0.1)

    def test_memory_breakdown(self):
        """Test memory breakdown computation."""
        from metrics.memory import MemoryMetrics

        metrics = MemoryMetrics()

        breakdown = metrics.compute_memory_breakdown(
            total_memory_bytes=10 * 1024**3,  # 10 GB
            model_memory_bytes=7 * 1024**3,   # 7 GB
            kv_cache_bytes=2 * 1024**3,       # 2 GB
        )

        assert breakdown["model_weights"]["percentage"] == pytest.approx(70.0)
        assert breakdown["kv_cache"]["percentage"] == pytest.approx(20.0)
        assert breakdown["other"]["percentage"] == pytest.approx(10.0)


class TestCostMetrics:
    """Tests for CostMetrics class."""

    def test_compute_cost(self):
        """Test cost computation."""
        from metrics.cost import CostMetrics

        metrics = CostMetrics(gpu_cost_per_hour=3.0)

        # 1000 tokens/sec = 3.6M tokens/hour
        # Cost per hour = $3.00
        # Cost per million = $3.00 / 3.6 = $0.83
        result = metrics.compute(tokens_per_second=1000.0)

        assert result["tokens_per_gpu_hour"] == pytest.approx(3_600_000)
        assert result["dollars_per_million_tokens"] == pytest.approx(0.833, rel=0.01)

    def test_monthly_cost_estimate(self):
        """Test monthly cost estimation."""
        from metrics.cost import CostMetrics

        metrics = CostMetrics(gpu_cost_per_hour=3.0)

        result = metrics.estimate_monthly_cost(
            tokens_per_day=1_000_000,
            tokens_per_second=1000.0,
            num_gpus=1,
            hours_per_day=24,
        )

        assert "monthly_cost_usd" in result
        assert result["monthly_cost_usd"] > 0
