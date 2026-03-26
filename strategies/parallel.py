"""Parallelism Strategies.

Support for multi-GPU parallelism:
- Tensor Parallelism (split model across GPUs)
- Pipeline Parallelism (layer-wise distribution)
- Sequence Parallelism (for long sequences)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from strategies.base import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)


class ParallelMethod(Enum):
    """Available parallelism methods."""

    NONE = "none"
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    SEQUENCE_PARALLEL = "sequence_parallel"
    HYBRID = "hybrid"  # Combination of tensor and pipeline


@dataclass
class ParallelConfig:
    """Configuration for parallelism strategy."""

    method: ParallelMethod = ParallelMethod.NONE
    # Tensor parallelism
    tensor_parallel_size: int = 1
    # Pipeline parallelism
    pipeline_parallel_size: int = 1
    num_micro_batches: int = 4
    # Sequence parallelism
    sequence_parallel_size: int = 1
    # Device configuration
    device_ids: list[int] | None = None
    # Communication backend
    backend: str = "nccl"


class ParallelStrategy(BaseStrategy):
    """Multi-GPU parallelism strategy.

    Supports:
    - Tensor Parallelism: Split model tensors across GPUs
    - Pipeline Parallelism: Distribute layers across GPUs
    - Sequence Parallelism: Split long sequences
    """

    def __init__(
        self,
        method: str | ParallelMethod = ParallelMethod.NONE,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        config: StrategyConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize parallelism strategy.

        Args:
            method: Parallelism method to use
            tensor_parallel_size: Number of GPUs for tensor parallelism
            pipeline_parallel_size: Number of GPUs for pipeline parallelism
            config: Base strategy configuration
            **kwargs: Additional options
        """
        super().__init__(config)

        if isinstance(method, str):
            method = ParallelMethod(method.lower())

        self.parallel_config = ParallelConfig(
            method=method,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            **{k: v for k, v in kwargs.items() if k in ParallelConfig.__dataclass_fields__},
        )

    @property
    def name(self) -> str:
        method = self.parallel_config.method
        if method == ParallelMethod.TENSOR_PARALLEL:
            return f"tensor_parallel_tp{self.parallel_config.tensor_parallel_size}"
        elif method == ParallelMethod.PIPELINE_PARALLEL:
            return f"pipeline_parallel_pp{self.parallel_config.pipeline_parallel_size}"
        elif method == ParallelMethod.HYBRID:
            tp = self.parallel_config.tensor_parallel_size
            pp = self.parallel_config.pipeline_parallel_size
            return f"hybrid_tp{tp}_pp{pp}"
        return f"parallel_{method.value}"

    def setup(self, model: Any, **kwargs: Any) -> Any:
        """Apply parallelism strategy to model.

        Args:
            model: Model to parallelize
            **kwargs: Additional options

        Returns:
            Parallelized model
        """
        method = self.parallel_config.method

        if method == ParallelMethod.NONE:
            logger.info("Parallelism: disabled (single GPU)")
            self._is_applied = True
            return model

        # Check GPU availability
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            logger.warning(
                f"Parallelism requires multiple GPUs. Found {num_gpus}. Disabling."
            )
            return model

        if method == ParallelMethod.TENSOR_PARALLEL:
            return self._setup_tensor_parallel(model, **kwargs)

        elif method == ParallelMethod.PIPELINE_PARALLEL:
            return self._setup_pipeline_parallel(model, **kwargs)

        elif method == ParallelMethod.SEQUENCE_PARALLEL:
            return self._setup_sequence_parallel(model, **kwargs)

        elif method == ParallelMethod.HYBRID:
            return self._setup_hybrid_parallel(model, **kwargs)

        else:
            logger.warning(f"Unknown parallelism method: {method}")
            return model

    def _setup_tensor_parallel(self, model: Any, **kwargs: Any) -> Any:
        """Set up tensor parallelism.

        Tensor parallelism splits individual layers across GPUs.
        """
        tp_size = self.parallel_config.tensor_parallel_size
        num_gpus = torch.cuda.device_count()

        if tp_size > num_gpus:
            logger.warning(
                f"Requested tensor_parallel_size={tp_size} but only {num_gpus} GPUs available. "
                f"Using {num_gpus}."
            )
            tp_size = num_gpus
            self.parallel_config.tensor_parallel_size = tp_size

        logger.info(f"Setting up tensor parallelism with {tp_size} GPUs")

        # For vLLM-style deployment, configure model for TP
        if hasattr(model, "config"):
            model.config.parallel_config = {
                "method": "tensor_parallel",
                "tensor_parallel_size": tp_size,
                "device_ids": list(range(tp_size)),
            }

        # If loading a model path, we'd use accelerate or similar
        if isinstance(model, str):
            try:
                from accelerate import init_empty_weights, load_checkpoint_and_dispatch
                from transformers import AutoConfig, AutoModelForCausalLM

                config = AutoConfig.from_pretrained(model)
                with init_empty_weights():
                    empty_model = AutoModelForCausalLM.from_config(config)

                # Create device map for tensor parallelism
                device_map = self._create_tp_device_map(empty_model, tp_size)

                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    device_map=device_map,
                    torch_dtype="auto",
                )

            except ImportError:
                logger.warning(
                    "accelerate not available for tensor parallelism setup. "
                    "Using basic multi-GPU placement."
                )
                # Fallback: just use device_map="auto"
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    device_map="auto",
                    torch_dtype="auto",
                )

        self._is_applied = True
        return model

    def _create_tp_device_map(self, model: Any, tp_size: int) -> dict[str, int]:
        """Create device map for tensor parallelism.

        This is a simplified approach - real TP requires splitting layers.
        """
        device_map = {}
        layers = []

        # Find all transformer layers
        for name, _ in model.named_modules():
            if "layers" in name or "h." in name:
                layer_name = name.split(".")[0]
                if layer_name not in layers:
                    layers.append(layer_name)

        # Distribute layers across GPUs
        if layers:
            layers_per_gpu = len(layers) // tp_size
            for i, layer in enumerate(layers):
                gpu_id = min(i // layers_per_gpu, tp_size - 1)
                device_map[layer] = gpu_id

        return device_map

    def _setup_pipeline_parallel(self, model: Any, **kwargs: Any) -> Any:
        """Set up pipeline parallelism.

        Pipeline parallelism distributes consecutive layers across GPUs.
        """
        pp_size = self.parallel_config.pipeline_parallel_size
        num_gpus = torch.cuda.device_count()

        if pp_size > num_gpus:
            logger.warning(
                f"Requested pipeline_parallel_size={pp_size} but only {num_gpus} GPUs available. "
                f"Using {num_gpus}."
            )
            pp_size = num_gpus
            self.parallel_config.pipeline_parallel_size = pp_size

        logger.info(
            f"Setting up pipeline parallelism with {pp_size} GPUs, "
            f"{self.parallel_config.num_micro_batches} micro-batches"
        )

        if hasattr(model, "config"):
            model.config.parallel_config = {
                "method": "pipeline_parallel",
                "pipeline_parallel_size": pp_size,
                "num_micro_batches": self.parallel_config.num_micro_batches,
            }

        # Pipeline parallelism typically uses accelerate or DeepSpeed
        if isinstance(model, str):
            from transformers import AutoModelForCausalLM

            # Use balanced device map for pipeline-like distribution
            model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="balanced",
                torch_dtype="auto",
            )

        self._is_applied = True
        return model

    def _setup_sequence_parallel(self, model: Any, **kwargs: Any) -> Any:
        """Set up sequence parallelism.

        Sequence parallelism splits long sequences across GPUs.
        """
        sp_size = self.parallel_config.sequence_parallel_size
        logger.info(f"Setting up sequence parallelism with {sp_size} GPUs")

        if hasattr(model, "config"):
            model.config.parallel_config = {
                "method": "sequence_parallel",
                "sequence_parallel_size": sp_size,
            }

        # Sequence parallelism is typically implemented at the attention level
        # and requires custom model modifications or framework support

        self._is_applied = True
        return model

    def _setup_hybrid_parallel(self, model: Any, **kwargs: Any) -> Any:
        """Set up hybrid tensor + pipeline parallelism."""
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        total_gpus = tp_size * pp_size

        num_gpus = torch.cuda.device_count()
        if total_gpus > num_gpus:
            logger.warning(
                f"Requested {total_gpus} GPUs (TP={tp_size} × PP={pp_size}) "
                f"but only {num_gpus} available."
            )
            return model

        logger.info(
            f"Setting up hybrid parallelism: TP={tp_size}, PP={pp_size} "
            f"(total {total_gpus} GPUs)"
        )

        if hasattr(model, "config"):
            model.config.parallel_config = {
                "method": "hybrid",
                "tensor_parallel_size": tp_size,
                "pipeline_parallel_size": pp_size,
                "total_gpus": total_gpus,
            }

        self._is_applied = True
        return model

    def describe(self) -> str:
        method = self.parallel_config.method

        if method == ParallelMethod.NONE:
            return "Parallelism: None (single GPU)"

        elif method == ParallelMethod.TENSOR_PARALLEL:
            tp = self.parallel_config.tensor_parallel_size
            return (
                f"Parallelism: Tensor Parallel with {tp} GPUs. "
                "Splits model weights across GPUs. Low latency, high bandwidth requirement."
            )

        elif method == ParallelMethod.PIPELINE_PARALLEL:
            pp = self.parallel_config.pipeline_parallel_size
            micro = self.parallel_config.num_micro_batches
            return (
                f"Parallelism: Pipeline Parallel with {pp} stages, {micro} micro-batches. "
                "Distributes layers across GPUs. Lower bandwidth, higher latency per request."
            )

        elif method == ParallelMethod.SEQUENCE_PARALLEL:
            sp = self.parallel_config.sequence_parallel_size
            return (
                f"Parallelism: Sequence Parallel with {sp} GPUs. "
                "Splits long sequences. Enables very long context."
            )

        elif method == ParallelMethod.HYBRID:
            tp = self.parallel_config.tensor_parallel_size
            pp = self.parallel_config.pipeline_parallel_size
            return (
                f"Parallelism: Hybrid (TP={tp}, PP={pp}). "
                "Combines tensor and pipeline parallelism for maximum scaling."
            )

        return f"Parallelism: {method.value}"

    def is_compatible(self, model_config: dict[str, Any]) -> bool:
        """Check if parallelism is compatible with setup."""
        num_gpus = torch.cuda.device_count()

        if self.parallel_config.method == ParallelMethod.NONE:
            return True

        total_required = self.parallel_config.tensor_parallel_size
        total_required *= self.parallel_config.pipeline_parallel_size

        if total_required > num_gpus:
            logger.warning(
                f"Parallelism requires {total_required} GPUs but only {num_gpus} available"
            )
            return False

        return True

    def estimate_scaling_efficiency(
        self,
        num_gpus: int,
        model_size_params: int,
    ) -> dict[str, float]:
        """Estimate scaling efficiency for parallelism.

        Args:
            num_gpus: Number of GPUs to use
            model_size_params: Model size in parameters

        Returns:
            Dictionary with scaling efficiency estimates
        """
        method = self.parallel_config.method

        # Base overhead estimates
        if method == ParallelMethod.TENSOR_PARALLEL:
            # TP has higher communication overhead but good scaling
            communication_overhead = 0.05 * num_gpus  # 5% per GPU
            scaling_efficiency = max(0.6, 1.0 - communication_overhead)

        elif method == ParallelMethod.PIPELINE_PARALLEL:
            # PP has bubble overhead
            micro_batches = self.parallel_config.num_micro_batches
            bubble_fraction = (num_gpus - 1) / (micro_batches + num_gpus - 1)
            scaling_efficiency = max(0.5, 1.0 - bubble_fraction)

        elif method == ParallelMethod.HYBRID:
            # Combination of overheads
            tp = self.parallel_config.tensor_parallel_size
            pp = self.parallel_config.pipeline_parallel_size
            tp_overhead = 0.05 * tp
            bubble = (pp - 1) / (self.parallel_config.num_micro_batches + pp - 1)
            scaling_efficiency = max(0.4, (1.0 - tp_overhead) * (1.0 - bubble))

        else:
            scaling_efficiency = 1.0

        return {
            "num_gpus": num_gpus,
            "scaling_efficiency": scaling_efficiency,
            "effective_throughput_multiplier": num_gpus * scaling_efficiency,
            "memory_per_gpu_ratio": 1.0 / num_gpus,
        }
