"""KV Cache Optimization Strategies.

Strategies for optimizing key-value cache memory usage:
- Paged Attention (vLLM-style)
- Sliding Window Attention
- Token Eviction (H2O-style)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from strategies.base import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)


class KVCacheMethod(Enum):
    """Available KV cache optimization methods."""

    NONE = "none"
    PAGED_ATTENTION = "paged_attention"
    SLIDING_WINDOW = "sliding_window"
    H2O_EVICTION = "h2o_eviction"


@dataclass
class KVCacheConfig:
    """Configuration for KV cache optimization."""

    method: KVCacheMethod = KVCacheMethod.NONE
    # Paged attention settings
    block_size: int = 16
    num_blocks: int | None = None  # Auto-calculate if None
    # Sliding window settings
    window_size: int = 4096
    # H2O eviction settings
    heavy_hitter_ratio: float = 0.2  # Fraction of tokens to keep as heavy hitters
    recent_ratio: float = 0.2  # Fraction of recent tokens to keep


class KVCacheStrategy(BaseStrategy):
    """KV cache optimization strategy.

    Supports multiple methods:
    - paged_attention: vLLM-style paged attention for efficient memory management
    - sliding_window: Keep only recent tokens in cache
    - h2o_eviction: Heavy-hitter oracle eviction policy
    """

    def __init__(
        self,
        method: str | KVCacheMethod = KVCacheMethod.PAGED_ATTENTION,
        window_size: int = 4096,
        block_size: int = 16,
        heavy_hitter_ratio: float = 0.2,
        config: StrategyConfig | None = None,
    ) -> None:
        """Initialize KV cache strategy.

        Args:
            method: KV cache optimization method
            window_size: Window size for sliding window attention
            block_size: Block size for paged attention
            heavy_hitter_ratio: Ratio of heavy hitter tokens to keep (H2O)
            config: Base strategy configuration
        """
        super().__init__(config)

        if isinstance(method, str):
            method = KVCacheMethod(method.lower())

        self.kv_config = KVCacheConfig(
            method=method,
            window_size=window_size,
            block_size=block_size,
            heavy_hitter_ratio=heavy_hitter_ratio,
        )

    @property
    def name(self) -> str:
        return f"kv_cache_{self.kv_config.method.value}"

    def setup(self, model: Any, **kwargs: Any) -> Any:
        """Apply KV cache optimization to model.

        Args:
            model: Model to optimize
            **kwargs: Additional options

        Returns:
            Optimized model
        """
        method = self.kv_config.method

        if method == KVCacheMethod.NONE:
            logger.info("KV cache optimization: disabled (baseline)")
            self._is_applied = True
            return model

        elif method == KVCacheMethod.PAGED_ATTENTION:
            return self._setup_paged_attention(model, **kwargs)

        elif method == KVCacheMethod.SLIDING_WINDOW:
            return self._setup_sliding_window(model, **kwargs)

        elif method == KVCacheMethod.H2O_EVICTION:
            return self._setup_h2o_eviction(model, **kwargs)

        else:
            logger.warning(f"Unknown KV cache method: {method}")
            return model

    def _setup_paged_attention(self, model: Any, **kwargs: Any) -> Any:
        """Set up paged attention (vLLM-style).

        Note: Full paged attention typically requires serving framework support.
        This configures the model for use with vLLM or similar.
        """
        logger.info(
            f"Configuring paged attention with block_size={self.kv_config.block_size}"
        )

        # Store config for serving backend to use
        if hasattr(model, "config"):
            model.config.kv_cache_config = {
                "method": "paged_attention",
                "block_size": self.kv_config.block_size,
                "num_blocks": self.kv_config.num_blocks,
            }

        self._is_applied = True
        return model

    def _setup_sliding_window(self, model: Any, **kwargs: Any) -> Any:
        """Set up sliding window attention.

        For models that support it natively (like Mistral), this is a no-op.
        For others, we configure the attention to use sliding window.
        """
        window_size = self.kv_config.window_size
        logger.info(f"Configuring sliding window attention with window_size={window_size}")

        # Check if model already has sliding window
        if hasattr(model, "config"):
            if hasattr(model.config, "sliding_window"):
                existing_window = model.config.sliding_window
                if existing_window is not None:
                    logger.info(
                        f"Model has native sliding window: {existing_window}. "
                        f"Overriding with: {window_size}"
                    )
            model.config.sliding_window = window_size
            model.config.kv_cache_config = {
                "method": "sliding_window",
                "window_size": window_size,
            }

        self._is_applied = True
        return model

    def _setup_h2o_eviction(self, model: Any, **kwargs: Any) -> Any:
        """Set up H2O-style heavy hitter eviction.

        H2O (Heavy-Hitter Oracle) keeps:
        - A fraction of heavy hitter tokens (most attended)
        - A fraction of recent tokens
        """
        logger.info(
            f"Configuring H2O eviction with heavy_hitter_ratio="
            f"{self.kv_config.heavy_hitter_ratio}, recent_ratio="
            f"{self.kv_config.recent_ratio}"
        )

        if hasattr(model, "config"):
            model.config.kv_cache_config = {
                "method": "h2o_eviction",
                "heavy_hitter_ratio": self.kv_config.heavy_hitter_ratio,
                "recent_ratio": self.kv_config.recent_ratio,
            }

        self._is_applied = True
        return model

    def describe(self) -> str:
        method = self.kv_config.method

        if method == KVCacheMethod.NONE:
            return "KV Cache: No optimization (full cache)"

        elif method == KVCacheMethod.PAGED_ATTENTION:
            return (
                f"KV Cache: Paged Attention (vLLM-style) with "
                f"block_size={self.kv_config.block_size}. "
                "Memory-efficient through block-level allocation."
            )

        elif method == KVCacheMethod.SLIDING_WINDOW:
            return (
                f"KV Cache: Sliding Window Attention with "
                f"window_size={self.kv_config.window_size}. "
                "Limits context to recent tokens."
            )

        elif method == KVCacheMethod.H2O_EVICTION:
            return (
                f"KV Cache: H2O Eviction with "
                f"heavy_hitter_ratio={self.kv_config.heavy_hitter_ratio}, "
                f"recent_ratio={self.kv_config.recent_ratio}. "
                "Keeps important and recent tokens."
            )

        return f"KV Cache: {method.value}"

    def is_compatible(self, model_config: dict[str, Any]) -> bool:
        """Check compatibility with model architecture."""
        # Most KV cache optimizations work with transformer models
        return True

    def estimate_memory_savings(
        self,
        sequence_length: int,
        batch_size: int,
        hidden_size: int,
        num_layers: int,
        num_kv_heads: int,
    ) -> dict[str, float]:
        """Estimate memory savings from KV cache optimization.

        Args:
            sequence_length: Maximum sequence length
            batch_size: Batch size
            hidden_size: Model hidden dimension
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads

        Returns:
            Dictionary with baseline and optimized memory estimates
        """
        head_dim = hidden_size // num_kv_heads
        # KV cache size: 2 (K+V) * layers * heads * head_dim * seq_len * batch * 2 (fp16)
        baseline_bytes = (
            2 * num_layers * num_kv_heads * head_dim * sequence_length * batch_size * 2
        )

        method = self.kv_config.method

        if method == KVCacheMethod.NONE:
            optimized_bytes = baseline_bytes

        elif method == KVCacheMethod.SLIDING_WINDOW:
            effective_len = min(sequence_length, self.kv_config.window_size)
            optimized_bytes = (
                2 * num_layers * num_kv_heads * head_dim * effective_len * batch_size * 2
            )

        elif method == KVCacheMethod.H2O_EVICTION:
            keep_ratio = self.kv_config.heavy_hitter_ratio + self.kv_config.recent_ratio
            optimized_bytes = int(baseline_bytes * min(1.0, keep_ratio))

        else:  # Paged attention - similar total but better utilization
            optimized_bytes = baseline_bytes  # Memory usage similar, but better managed

        return {
            "baseline_bytes": baseline_bytes,
            "baseline_gb": baseline_bytes / (1024**3),
            "optimized_bytes": optimized_bytes,
            "optimized_gb": optimized_bytes / (1024**3),
            "savings_ratio": 1 - (optimized_bytes / baseline_bytes) if baseline_bytes > 0 else 0,
        }
