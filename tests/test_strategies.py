"""Tests for strategy implementations."""

import pytest


class TestBaseStrategy:
    """Tests for BaseStrategy and BaselineStrategy."""

    def test_baseline_strategy(self):
        """Test baseline strategy initialization."""
        from strategies.base import BaselineStrategy

        strategy = BaselineStrategy()

        assert strategy.name == "baseline"
        assert "baseline" in strategy.describe().lower()

    def test_strategy_config(self):
        """Test StrategyConfig dataclass."""
        from strategies.base import StrategyConfig

        config = StrategyConfig(enabled=True, params={"key": "value"})

        assert config.enabled
        assert config.params["key"] == "value"

        result = config.to_dict()
        assert result["enabled"] is True


class TestQuantizationStrategy:
    """Tests for QuantizationStrategy."""

    def test_quantization_methods(self):
        """Test different quantization method initialization."""
        from strategies.quantization import QuantizationStrategy, QuantizationMethod

        # Test INT8
        strategy = QuantizationStrategy(method="int8")
        assert strategy.quant_config.method == QuantizationMethod.INT8
        assert strategy.quant_config.bits == 8

        # Test INT4
        strategy = QuantizationStrategy(method="int4")
        assert strategy.quant_config.method == QuantizationMethod.INT4
        assert strategy.quant_config.bits == 4

    def test_memory_reduction_estimate(self):
        """Test memory reduction estimation."""
        from strategies.quantization import QuantizationStrategy

        strategy = QuantizationStrategy(method="int4")
        result = strategy.estimate_memory_reduction(14.0)  # 14 GB model

        assert result["original_size_gb"] == 14.0
        assert result["quantized_size_gb"] == pytest.approx(3.5)
        assert result["reduction_ratio"] == pytest.approx(0.25)

    def test_quantization_describe(self):
        """Test strategy description."""
        from strategies.quantization import QuantizationStrategy

        strategy = QuantizationStrategy(method="gptq")
        description = strategy.describe()

        assert "GPTQ" in description
        assert "4-bit" in description


class TestKVCacheStrategy:
    """Tests for KVCacheStrategy."""

    def test_kv_cache_methods(self):
        """Test different KV cache methods."""
        from strategies.kv_cache import KVCacheStrategy, KVCacheMethod

        # Test paged attention
        strategy = KVCacheStrategy(method="paged_attention")
        assert strategy.kv_config.method == KVCacheMethod.PAGED_ATTENTION

        # Test sliding window
        strategy = KVCacheStrategy(method="sliding_window", window_size=2048)
        assert strategy.kv_config.method == KVCacheMethod.SLIDING_WINDOW
        assert strategy.kv_config.window_size == 2048

    def test_memory_savings_estimate(self):
        """Test memory savings estimation."""
        from strategies.kv_cache import KVCacheStrategy

        strategy = KVCacheStrategy(method="sliding_window", window_size=2048)

        result = strategy.estimate_memory_savings(
            sequence_length=8192,
            batch_size=1,
            hidden_size=4096,
            num_layers=32,
            num_kv_heads=8,
        )

        assert result["savings_ratio"] > 0
        assert result["optimized_bytes"] < result["baseline_bytes"]


class TestSpeculativeDecodingStrategy:
    """Tests for SpeculativeDecodingStrategy."""

    def test_speculative_init(self):
        """Test speculative decoding initialization."""
        from strategies.speculative_decoding import SpeculativeDecodingStrategy

        strategy = SpeculativeDecodingStrategy(
            draft_model_name="EleutherAI/pythia-70m",
            draft_length=5,
        )

        assert strategy.name == "speculative_decoding"
        assert strategy.spec_config.draft_length == 5

    def test_default_draft_models(self):
        """Test default draft model selection."""
        from strategies.speculative_decoding import SpeculativeDecodingStrategy

        strategy = SpeculativeDecodingStrategy()

        # Test default draft model lookup
        draft = strategy._get_default_draft("llama-2-7b")
        assert draft is not None
        assert "pythia" in draft.lower()


class TestBatchingStrategy:
    """Tests for BatchingStrategy."""

    def test_batching_methods(self):
        """Test different batching methods."""
        from strategies.batching import BatchingStrategy, BatchingMethod

        # Test static batching
        strategy = BatchingStrategy(method="static", batch_size=16)
        assert strategy.batch_config.method == BatchingMethod.STATIC
        assert strategy.batch_config.batch_size == 16

        # Test continuous batching
        strategy = BatchingStrategy(method="continuous")
        assert strategy.batch_config.method == BatchingMethod.CONTINUOUS


class TestAttentionStrategy:
    """Tests for AttentionStrategy."""

    def test_attention_methods(self):
        """Test different attention methods."""
        from strategies.attention import AttentionStrategy, AttentionMethod

        # Test FlashAttention-2
        strategy = AttentionStrategy(method="flash_attention_2")
        assert strategy.attn_config.method == AttentionMethod.FLASH_ATTENTION_2

        # Test SDPA
        strategy = AttentionStrategy(method="sdpa")
        assert strategy.attn_config.method == AttentionMethod.SDPA

    def test_memory_savings_estimate(self):
        """Test memory savings estimation."""
        from strategies.attention import AttentionStrategy

        strategy = AttentionStrategy(method="flash_attention_2")

        result = strategy.estimate_memory_savings(
            sequence_length=4096,
            batch_size=1,
            hidden_size=4096,
            num_heads=32,
        )

        assert result["savings_ratio"] > 0


class TestParallelStrategy:
    """Tests for ParallelStrategy."""

    def test_parallel_methods(self):
        """Test different parallelism methods."""
        from strategies.parallel import ParallelStrategy, ParallelMethod

        # Test tensor parallelism
        strategy = ParallelStrategy(
            method="tensor_parallel",
            tensor_parallel_size=2,
        )
        assert strategy.parallel_config.method == ParallelMethod.TENSOR_PARALLEL
        assert strategy.parallel_config.tensor_parallel_size == 2

    def test_scaling_efficiency_estimate(self):
        """Test scaling efficiency estimation."""
        from strategies.parallel import ParallelStrategy

        strategy = ParallelStrategy(method="tensor_parallel", tensor_parallel_size=4)

        result = strategy.estimate_scaling_efficiency(
            num_gpus=4,
            model_size_params=7_000_000_000,
        )

        assert 0 < result["scaling_efficiency"] <= 1.0
        assert result["effective_throughput_multiplier"] > 1.0
