"""Quantization Strategies.

Support for various quantization methods:
- FP16/BF16 (baseline)
- INT8 (bitsandbytes)
- INT4 / NF4 (bitsandbytes)
- GPTQ (4-bit)
- AWQ (4-bit)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from strategies.base import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    """Available quantization methods."""

    NONE = "none"  # FP16/BF16 baseline
    INT8 = "int8"  # bitsandbytes 8-bit
    INT4 = "int4"  # bitsandbytes 4-bit
    NF4 = "nf4"  # bitsandbytes NormalFloat4
    GPTQ = "gptq"  # GPTQ 4-bit
    AWQ = "awq"  # AWQ 4-bit


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    method: QuantizationMethod = QuantizationMethod.NONE
    bits: int = 16
    # bitsandbytes settings
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # GPTQ settings
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_desc_act: bool = False
    # AWQ settings
    awq_bits: int = 4
    awq_group_size: int = 128
    awq_zero_point: bool = True


class QuantizationStrategy(BaseStrategy):
    """Quantization strategy for model compression.

    Supports INT8, INT4/NF4 (bitsandbytes), GPTQ, and AWQ quantization methods.
    """

    def __init__(
        self,
        method: str | QuantizationMethod = QuantizationMethod.NONE,
        bits: int | None = None,
        config: StrategyConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize quantization strategy.

        Args:
            method: Quantization method to use
            bits: Number of bits (auto-determined if None)
            config: Base strategy configuration
            **kwargs: Method-specific options
        """
        super().__init__(config)

        if isinstance(method, str):
            method = QuantizationMethod(method.lower())

        # Determine bits from method if not specified
        if bits is None:
            bits = self._default_bits(method)

        self.quant_config = QuantizationConfig(
            method=method,
            bits=bits,
            load_in_8bit=(method == QuantizationMethod.INT8),
            load_in_4bit=(method in [QuantizationMethod.INT4, QuantizationMethod.NF4]),
            bnb_4bit_quant_type="nf4" if method == QuantizationMethod.NF4 else "fp4",
            **{k: v for k, v in kwargs.items() if k in QuantizationConfig.__dataclass_fields__},
        )

    def _default_bits(self, method: QuantizationMethod) -> int:
        """Get default bits for a quantization method."""
        mapping = {
            QuantizationMethod.NONE: 16,
            QuantizationMethod.INT8: 8,
            QuantizationMethod.INT4: 4,
            QuantizationMethod.NF4: 4,
            QuantizationMethod.GPTQ: 4,
            QuantizationMethod.AWQ: 4,
        }
        return mapping.get(method, 16)

    @property
    def name(self) -> str:
        method = self.quant_config.method
        if method == QuantizationMethod.NONE:
            return "baseline_fp16"
        return f"quantization_{method.value}_{self.quant_config.bits}bit"

    def setup(self, model: Any, **kwargs: Any) -> Any:
        """Apply quantization to model.

        Args:
            model: Model to quantize (or model name for loading)
            **kwargs: Additional options

        Returns:
            Quantized model
        """
        method = self.quant_config.method

        if method == QuantizationMethod.NONE:
            logger.info("Quantization: disabled (FP16/BF16 baseline)")
            self._is_applied = True
            return model

        elif method == QuantizationMethod.INT8:
            return self._apply_int8(model, **kwargs)

        elif method in [QuantizationMethod.INT4, QuantizationMethod.NF4]:
            return self._apply_int4(model, **kwargs)

        elif method == QuantizationMethod.GPTQ:
            return self._apply_gptq(model, **kwargs)

        elif method == QuantizationMethod.AWQ:
            return self._apply_awq(model, **kwargs)

        else:
            logger.warning(f"Unknown quantization method: {method}")
            return model

    def _apply_int8(self, model: Any, **kwargs: Any) -> Any:
        """Apply INT8 quantization using bitsandbytes."""
        logger.info("Applying INT8 quantization (bitsandbytes)")

        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # If model is a string (model name), load with quantization
            if isinstance(model, str):
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                # Model already loaded - this is more complex
                # In practice, quantization should be applied during loading
                logger.warning(
                    "INT8 quantization works best when applied during model loading. "
                    "Consider passing model name instead of loaded model."
                )

            self._is_applied = True
            return model

        except ImportError as e:
            logger.error(f"bitsandbytes not available: {e}")
            raise

    def _apply_int4(self, model: Any, **kwargs: Any) -> Any:
        """Apply INT4/NF4 quantization using bitsandbytes."""
        quant_type = self.quant_config.bnb_4bit_quant_type
        logger.info(f"Applying 4-bit quantization (bitsandbytes, type={quant_type})")

        try:
            import torch
            from transformers import BitsAndBytesConfig

            compute_dtype = getattr(torch, self.quant_config.bnb_4bit_compute_dtype)

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=self.quant_config.bnb_4bit_use_double_quant,
            )

            if isinstance(model, str):
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                logger.warning(
                    "4-bit quantization works best when applied during model loading."
                )

            self._is_applied = True
            return model

        except ImportError as e:
            logger.error(f"bitsandbytes not available: {e}")
            raise

    def _apply_gptq(self, model: Any, **kwargs: Any) -> Any:
        """Apply GPTQ quantization."""
        logger.info(
            f"Applying GPTQ quantization (bits={self.quant_config.gptq_bits}, "
            f"group_size={self.quant_config.gptq_group_size})"
        )

        try:
            from transformers import GPTQConfig

            quantization_config = GPTQConfig(
                bits=self.quant_config.gptq_bits,
                group_size=self.quant_config.gptq_group_size,
                desc_act=self.quant_config.gptq_desc_act,
            )

            if isinstance(model, str):
                from transformers import AutoModelForCausalLM

                # Try loading pre-quantized model first
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model,
                        quantization_config=quantization_config,
                        device_map="auto",
                    )
                except Exception:
                    logger.info("Pre-quantized model not found, quantizing on-the-fly")
                    # Would need calibration data for on-the-fly quantization
                    raise NotImplementedError(
                        "On-the-fly GPTQ quantization requires calibration data"
                    )

            self._is_applied = True
            return model

        except ImportError as e:
            logger.error(f"auto-gptq not available: {e}")
            raise

    def _apply_awq(self, model: Any, **kwargs: Any) -> Any:
        """Apply AWQ quantization."""
        logger.info(
            f"Applying AWQ quantization (bits={self.quant_config.awq_bits}, "
            f"group_size={self.quant_config.awq_group_size})"
        )

        try:
            from transformers import AwqConfig

            quantization_config = AwqConfig(
                bits=self.quant_config.awq_bits,
                group_size=self.quant_config.awq_group_size,
                zero_point=self.quant_config.awq_zero_point,
            )

            if isinstance(model, str):
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    quantization_config=quantization_config,
                    device_map="auto",
                )

            self._is_applied = True
            return model

        except ImportError as e:
            logger.error(f"autoawq not available: {e}")
            raise

    def describe(self) -> str:
        method = self.quant_config.method
        bits = self.quant_config.bits

        if method == QuantizationMethod.NONE:
            return "Quantization: None (FP16/BF16 baseline)"

        elif method == QuantizationMethod.INT8:
            return "Quantization: INT8 (bitsandbytes). ~2x memory reduction."

        elif method == QuantizationMethod.INT4:
            return (
                "Quantization: INT4 (bitsandbytes). ~4x memory reduction. "
                "Uses FP4 quantization."
            )

        elif method == QuantizationMethod.NF4:
            return (
                "Quantization: NF4 (bitsandbytes NormalFloat4). ~4x memory reduction. "
                "Better quality than FP4."
            )

        elif method == QuantizationMethod.GPTQ:
            return (
                f"Quantization: GPTQ {bits}-bit with group_size="
                f"{self.quant_config.gptq_group_size}. High quality post-training quantization."
            )

        elif method == QuantizationMethod.AWQ:
            return (
                f"Quantization: AWQ {bits}-bit with group_size="
                f"{self.quant_config.awq_group_size}. Activation-aware weight quantization."
            )

        return f"Quantization: {method.value} {bits}-bit"

    def estimate_memory_reduction(self, original_size_gb: float) -> dict[str, float]:
        """Estimate memory reduction from quantization.

        Args:
            original_size_gb: Original model size in GB (FP16)

        Returns:
            Dictionary with estimated sizes
        """
        bits = self.quant_config.bits
        baseline_bits = 16

        # Simple estimation based on bit reduction
        reduction_ratio = bits / baseline_bits
        quantized_size_gb = original_size_gb * reduction_ratio

        return {
            "original_size_gb": original_size_gb,
            "quantized_size_gb": quantized_size_gb,
            "reduction_ratio": reduction_ratio,
            "memory_saved_gb": original_size_gb - quantized_size_gb,
        }

    def quantize_model(self, model: Any, method: str, bits: int) -> Any:
        """Quantize a model with specified method and bits.

        Convenience method for quick quantization.

        Args:
            model: Model or model name to quantize
            method: Quantization method name
            bits: Number of bits

        Returns:
            Quantized model
        """
        self.quant_config.method = QuantizationMethod(method.lower())
        self.quant_config.bits = bits
        return self.setup(model)
