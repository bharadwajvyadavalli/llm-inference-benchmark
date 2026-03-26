"""vLLM Backend: High-throughput serving with PagedAttention."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from serving.base import BaseServingBackend, GenerationConfig, GenerationOutput

logger = logging.getLogger(__name__)


class VLLMBackend(BaseServingBackend):
    """vLLM serving backend.

    Uses vLLM's optimized inference engine with:
    - PagedAttention for efficient KV cache management
    - Continuous batching
    - CUDA graph optimization
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ) -> None:
        """Initialize vLLM backend.

        Args:
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (None for model default)
        """
        super().__init__()
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None

    @property
    def name(self) -> str:
        return "vllm"

    def load_model(
        self,
        model_name: str,
        strategy_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Load model using vLLM.

        Args:
            model_name: HuggingFace model path
            strategy_config: Strategy configuration
            **kwargs: Additional vLLM options
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )

        logger.info(f"Loading model {model_name} with vLLM backend")

        # Build vLLM configuration
        vllm_kwargs = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
        }

        if self.max_model_len is not None:
            vllm_kwargs["max_model_len"] = self.max_model_len

        # Apply strategy config
        if strategy_config:
            self._apply_strategy_config(vllm_kwargs, strategy_config)

        # Override with any explicit kwargs
        vllm_kwargs.update(kwargs)

        # Create vLLM engine
        self._llm = LLM(model_name, **vllm_kwargs)
        self._model_name = model_name
        self._is_loaded = True

        logger.info(f"Model loaded successfully with vLLM (TP={self.tensor_parallel_size})")

    def _apply_strategy_config(
        self,
        vllm_kwargs: dict[str, Any],
        strategy_config: dict[str, Any],
    ) -> None:
        """Apply strategy configuration to vLLM kwargs."""
        config = strategy_config.get("config", {})
        params = config.get("params", {})
        name = strategy_config.get("name", "")

        # Quantization
        if "quantization" in name:
            method = params.get("method", "")
            if method == "awq":
                vllm_kwargs["quantization"] = "awq"
            elif method == "gptq":
                vllm_kwargs["quantization"] = "gptq"
            elif method in ["int8", "int4"]:
                # bitsandbytes quantization
                vllm_kwargs["quantization"] = "bitsandbytes"
                vllm_kwargs["load_format"] = "bitsandbytes"

        # Speculative decoding
        if "speculative" in name:
            draft_model = params.get("draft_model_name")
            if draft_model:
                vllm_kwargs["speculative_model"] = draft_model
                vllm_kwargs["num_speculative_tokens"] = params.get("draft_length", 5)

        # Attention implementation
        if "attention" in name:
            method = params.get("method", "")
            if method == "flash_attention_2":
                # vLLM uses FlashAttention by default when available
                pass

        # KV cache configuration
        if "kv_cache" in name:
            block_size = params.get("block_size")
            if block_size:
                vllm_kwargs["block_size"] = block_size

    def generate(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[GenerationOutput]:
        """Generate text using vLLM.

        Args:
            prompts: List of input prompts
            generation_config: Generation configuration
            **kwargs: Additional options

        Returns:
            List of GenerationOutput
        """
        if not self._is_loaded or self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed")

        if generation_config is None:
            generation_config = GenerationConfig()

        # Create sampling params
        sampling_params = SamplingParams(
            max_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            repetition_penalty=generation_config.repetition_penalty,
            stop=generation_config.stop_sequences or None,
        )

        # Generate
        start_time = time.perf_counter()
        outputs = self._llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()

        total_latency_ms = (end_time - start_time) * 1000

        # Convert to GenerationOutput
        results = []
        for output in outputs:
            # Get the first completion
            completion = output.outputs[0]

            results.append(
                GenerationOutput(
                    text=completion.text,
                    tokens=list(completion.token_ids),
                    num_tokens=len(completion.token_ids),
                    finish_reason=completion.finish_reason,
                    latency_ms=total_latency_ms / len(prompts),  # Average
                    ttft_ms=None,  # Would need metrics tracking
                )
            )

        return results

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"available": False}

        memory_info = {
            "available": True,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        }

        # Try to get vLLM-specific memory info
        if self._llm is not None:
            try:
                # vLLM may expose cache statistics
                cache_config = getattr(self._llm, "cache_config", None)
                if cache_config:
                    memory_info["vllm_block_size"] = getattr(cache_config, "block_size", None)
                    memory_info["vllm_num_gpu_blocks"] = getattr(cache_config, "num_gpu_blocks", None)
            except Exception:
                pass

        return memory_info

    def unload_model(self) -> None:
        """Unload the vLLM model."""
        if self._llm is not None:
            del self._llm
            self._llm = None

        super().unload_model()

    def get_model_config(self) -> dict[str, Any]:
        """Get the loaded model's configuration."""
        if self._llm is None:
            return {}

        try:
            config = self._llm.llm_engine.model_config
            return {
                "model": config.model,
                "max_model_len": config.max_model_len,
                "dtype": str(config.dtype),
                "tensor_parallel_size": self.tensor_parallel_size,
            }
        except Exception:
            return {"model": self._model_name}
