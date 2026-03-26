"""Attention Optimization Strategies.

Support for various attention optimizations:
- FlashAttention-2
- Grouped Query Attention (GQA)
- Multi-Query Attention (MQA)
- Ring Attention (for long sequences)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from strategies.base import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)


class AttentionMethod(Enum):
    """Available attention optimization methods."""

    STANDARD = "standard"  # Standard multi-head attention
    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"  # PyTorch scaled_dot_product_attention
    GQA = "gqa"  # Grouped Query Attention
    MQA = "mqa"  # Multi-Query Attention
    RING_ATTENTION = "ring_attention"


@dataclass
class AttentionConfig:
    """Configuration for attention optimization."""

    method: AttentionMethod = AttentionMethod.STANDARD
    # FlashAttention settings
    use_flash_attention_2: bool = False
    flash_attention_causal: bool = True
    # GQA/MQA settings
    num_kv_heads: int | None = None  # None = use model default
    # Ring attention settings
    ring_attention_enabled: bool = False
    ring_size: int = 4  # Number of GPUs in ring
    # SDPA settings
    enable_math: bool = True
    enable_flash: bool = True
    enable_mem_efficient: bool = True


class AttentionStrategy(BaseStrategy):
    """Attention optimization strategy.

    Supports various attention implementations:
    - FlashAttention-2: Memory-efficient exact attention
    - SDPA: PyTorch's native scaled_dot_product_attention
    - GQA/MQA: Reduced KV heads for memory savings
    """

    def __init__(
        self,
        method: str | AttentionMethod = AttentionMethod.FLASH_ATTENTION_2,
        config: StrategyConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize attention strategy.

        Args:
            method: Attention method to use
            config: Base strategy configuration
            **kwargs: Method-specific options
        """
        super().__init__(config)

        if isinstance(method, str):
            method = AttentionMethod(method.lower())

        self.attn_config = AttentionConfig(
            method=method,
            use_flash_attention_2=(method == AttentionMethod.FLASH_ATTENTION_2),
            **{k: v for k, v in kwargs.items() if k in AttentionConfig.__dataclass_fields__},
        )

    @property
    def name(self) -> str:
        return f"attention_{self.attn_config.method.value}"

    def setup(self, model: Any, **kwargs: Any) -> Any:
        """Apply attention optimization to model.

        Args:
            model: Model to optimize
            **kwargs: Additional options

        Returns:
            Optimized model
        """
        method = self.attn_config.method

        if method == AttentionMethod.STANDARD:
            logger.info("Attention: Standard multi-head attention (no optimization)")
            self._is_applied = True
            return model

        elif method == AttentionMethod.FLASH_ATTENTION_2:
            return self._setup_flash_attention(model, **kwargs)

        elif method == AttentionMethod.SDPA:
            return self._setup_sdpa(model, **kwargs)

        elif method in [AttentionMethod.GQA, AttentionMethod.MQA]:
            return self._setup_grouped_attention(model, **kwargs)

        elif method == AttentionMethod.RING_ATTENTION:
            return self._setup_ring_attention(model, **kwargs)

        else:
            logger.warning(f"Unknown attention method: {method}")
            return model

    def _setup_flash_attention(self, model: Any, **kwargs: Any) -> Any:
        """Enable FlashAttention-2."""
        logger.info("Enabling FlashAttention-2")

        try:
            # Check if flash-attn is available
            import flash_attn  # noqa: F401

            # Enable flash attention in model config
            if hasattr(model, "config"):
                model.config._attn_implementation = "flash_attention_2"
                model.config.attention_config = {
                    "method": "flash_attention_2",
                    "causal": self.attn_config.flash_attention_causal,
                }

            # For transformers models, we might need to reload with flash attention
            if isinstance(model, str):
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype="auto",
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )

            self._is_applied = True
            return model

        except ImportError:
            logger.warning(
                "flash-attn not installed. Falling back to SDPA. "
                "Install with: pip install flash-attn"
            )
            return self._setup_sdpa(model, **kwargs)

    def _setup_sdpa(self, model: Any, **kwargs: Any) -> Any:
        """Enable PyTorch SDPA (Scaled Dot Product Attention)."""
        logger.info("Enabling PyTorch SDPA")

        import torch

        # Configure SDPA backends
        torch.backends.cuda.enable_flash_sdp(self.attn_config.enable_flash)
        torch.backends.cuda.enable_math_sdp(self.attn_config.enable_math)
        torch.backends.cuda.enable_mem_efficient_sdp(self.attn_config.enable_mem_efficient)

        if hasattr(model, "config"):
            model.config._attn_implementation = "sdpa"
            model.config.attention_config = {
                "method": "sdpa",
                "enable_flash": self.attn_config.enable_flash,
                "enable_math": self.attn_config.enable_math,
                "enable_mem_efficient": self.attn_config.enable_mem_efficient,
            }

        self._is_applied = True
        return model

    def _setup_grouped_attention(self, model: Any, **kwargs: Any) -> Any:
        """Configure GQA or MQA.

        Note: GQA/MQA is typically a model architecture choice, not a runtime
        optimization. This method verifies and configures accordingly.
        """
        method = self.attn_config.method
        logger.info(f"Configuring {method.value.upper()}")

        if hasattr(model, "config"):
            num_kv_heads = self.attn_config.num_kv_heads
            if num_kv_heads is not None:
                # Check if model supports configurable KV heads
                if hasattr(model.config, "num_key_value_heads"):
                    model.config.num_key_value_heads = num_kv_heads
                    logger.info(f"Set num_key_value_heads to {num_kv_heads}")
                else:
                    logger.warning(
                        "Model doesn't support configurable KV heads. "
                        "GQA/MQA requires architectural support."
                    )

            model.config.attention_config = {
                "method": method.value,
                "num_kv_heads": num_kv_heads,
            }

        self._is_applied = True
        return model

    def _setup_ring_attention(self, model: Any, **kwargs: Any) -> Any:
        """Configure ring attention for long sequences.

        Ring attention distributes sequence across GPUs in a ring pattern
        for processing very long sequences.
        """
        logger.info(
            f"Configuring ring attention with ring_size={self.attn_config.ring_size}"
        )

        if hasattr(model, "config"):
            model.config.attention_config = {
                "method": "ring_attention",
                "ring_size": self.attn_config.ring_size,
            }

        self._is_applied = True
        return model

    def describe(self) -> str:
        method = self.attn_config.method

        if method == AttentionMethod.STANDARD:
            return "Attention: Standard multi-head attention. No memory optimization."

        elif method == AttentionMethod.FLASH_ATTENTION_2:
            return (
                "Attention: FlashAttention-2. IO-aware exact attention algorithm. "
                "Up to 2-4x speedup and significant memory savings."
            )

        elif method == AttentionMethod.SDPA:
            return (
                "Attention: PyTorch SDPA. Native scaled_dot_product_attention with "
                "automatic backend selection (Flash/Memory-efficient/Math)."
            )

        elif method == AttentionMethod.GQA:
            kv_heads = self.attn_config.num_kv_heads or "model default"
            return (
                f"Attention: Grouped Query Attention with {kv_heads} KV heads. "
                "Reduces KV cache memory through head sharing."
            )

        elif method == AttentionMethod.MQA:
            return (
                "Attention: Multi-Query Attention with single KV head. "
                "Maximum KV cache reduction but may affect quality."
            )

        elif method == AttentionMethod.RING_ATTENTION:
            return (
                f"Attention: Ring Attention with ring_size={self.attn_config.ring_size}. "
                "Enables very long sequences across multiple GPUs."
            )

        return f"Attention: {method.value}"

    def is_compatible(self, model_config: dict[str, Any]) -> bool:
        """Check if attention method is compatible with model."""
        method = self.attn_config.method

        # FlashAttention requires GPU with compute capability >= 7.5
        if method == AttentionMethod.FLASH_ATTENTION_2:
            import torch

            if not torch.cuda.is_available():
                return False
            # Check compute capability
            props = torch.cuda.get_device_properties(0)
            if props.major < 7 or (props.major == 7 and props.minor < 5):
                logger.warning(
                    f"FlashAttention requires compute capability >= 7.5, "
                    f"got {props.major}.{props.minor}"
                )
                return False

        # GQA/MQA requires model architectural support
        if method in [AttentionMethod.GQA, AttentionMethod.MQA]:
            attention_type = model_config.get("attention_type", "mha").lower()
            if method == AttentionMethod.GQA and attention_type not in ["gqa", "mha"]:
                return False
            if method == AttentionMethod.MQA and attention_type not in ["mqa", "mha"]:
                return False

        return True

    def benchmark_sequence_lengths(self) -> list[int]:
        """Return sequence lengths to benchmark for this attention method."""
        method = self.attn_config.method

        if method == AttentionMethod.RING_ATTENTION:
            # Ring attention is for very long sequences
            return [8192, 16384, 32768, 65536, 131072]
        else:
            # Standard benchmarking lengths
            return [512, 1024, 2048, 4096, 8192, 16384]

    def estimate_memory_savings(
        self,
        sequence_length: int,
        batch_size: int,
        hidden_size: int,
        num_heads: int,
    ) -> dict[str, float]:
        """Estimate memory savings from attention optimization.

        Args:
            sequence_length: Sequence length
            batch_size: Batch size
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads

        Returns:
            Dictionary with memory estimates
        """
        method = self.attn_config.method

        # Standard attention memory: O(seq_len^2 * batch_size * num_heads)
        standard_memory = sequence_length**2 * batch_size * num_heads * 4  # FP32

        if method == AttentionMethod.FLASH_ATTENTION_2:
            # FlashAttention: O(seq_len * batch_size * hidden_size)
            optimized_memory = sequence_length * batch_size * hidden_size * 4
        elif method == AttentionMethod.SDPA:
            # SDPA with memory efficient: similar to FlashAttention
            optimized_memory = sequence_length * batch_size * hidden_size * 4
        elif method in [AttentionMethod.GQA, AttentionMethod.MQA]:
            # Reduced KV cache
            num_kv_heads = self.attn_config.num_kv_heads or num_heads
            reduction = num_kv_heads / num_heads
            optimized_memory = standard_memory * reduction
        else:
            optimized_memory = standard_memory

        return {
            "standard_memory_bytes": standard_memory,
            "optimized_memory_bytes": optimized_memory,
            "savings_ratio": 1 - (optimized_memory / standard_memory),
            "standard_memory_gb": standard_memory / (1024**3),
            "optimized_memory_gb": optimized_memory / (1024**3),
        }
