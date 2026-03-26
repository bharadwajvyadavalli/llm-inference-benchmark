"""HuggingFace Backend: Raw transformers generate() baseline."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from serving.base import BaseServingBackend, GenerationConfig, GenerationOutput

logger = logging.getLogger(__name__)


class HuggingFaceBackend(BaseServingBackend):
    """HuggingFace transformers backend.

    Uses raw model.generate() for inference. This is the unoptimized
    baseline for comparison with optimized serving frameworks.
    """

    def __init__(
        self,
        device: str = "cuda",
        torch_dtype: str = "auto",
    ) -> None:
        """Initialize HuggingFace backend.

        Args:
            device: Device to use ("cuda", "cpu", or specific GPU)
            torch_dtype: Torch dtype ("auto", "float16", "bfloat16", "float32")
        """
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype

    @property
    def name(self) -> str:
        return "huggingface"

    def load_model(
        self,
        model_name: str,
        strategy_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Load model using HuggingFace transformers.

        Args:
            model_name: HuggingFace model path
            strategy_config: Strategy configuration (for quantization, etc.)
            **kwargs: Additional options
        """
        logger.info(f"Loading model {model_name} with HuggingFace backend")

        # Determine torch dtype
        if self.torch_dtype == "auto":
            dtype = "auto"
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Check for quantization config
        quantization_config = None
        if strategy_config:
            quant_method = strategy_config.get("config", {}).get("params", {}).get("method")
            if quant_method:
                quantization_config = self._create_quantization_config(strategy_config)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        # Check for attention implementation
        if strategy_config:
            attn_impl = strategy_config.get("config", {}).get("params", {}).get("attn_implementation")
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        # Move to device if not using device_map
        if self.device != "cuda" and hasattr(self._model, "to"):
            self._model = self._model.to(self.device)

        self._model.eval()
        self._model_name = model_name
        self._is_loaded = True

        logger.info(f"Model loaded successfully on {self.device}")

    def _create_quantization_config(
        self, strategy_config: dict[str, Any]
    ) -> Any:
        """Create quantization config from strategy config."""
        params = strategy_config.get("config", {}).get("params", {})
        method = params.get("method", "")

        if method in ["int8"]:
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(load_in_8bit=True)

        elif method in ["int4", "nf4"]:
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4" if method == "nf4" else "fp4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        elif method == "gptq":
            from transformers import GPTQConfig

            return GPTQConfig(
                bits=params.get("bits", 4),
                group_size=params.get("group_size", 128),
            )

        elif method == "awq":
            from transformers import AwqConfig

            return AwqConfig(
                bits=params.get("bits", 4),
                group_size=params.get("group_size", 128),
            )

        return None

    def generate(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[GenerationOutput]:
        """Generate text using model.generate().

        Args:
            prompts: List of input prompts
            generation_config: Generation configuration
            **kwargs: Additional options

        Returns:
            List of GenerationOutput
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if generation_config is None:
            generation_config = GenerationConfig()

        results = []

        # Process each prompt
        for prompt in prompts:
            start_time = time.perf_counter()

            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._model.device)

            input_length = inputs["input_ids"].shape[1]

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=generation_config.max_new_tokens,
                    temperature=generation_config.temperature,
                    top_p=generation_config.top_p,
                    top_k=generation_config.top_k,
                    do_sample=generation_config.do_sample,
                    repetition_penalty=generation_config.repetition_penalty,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Decode output
            generated_tokens = outputs[0][input_length:]
            generated_text = self._tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            # Determine finish reason
            if len(generated_tokens) >= generation_config.max_new_tokens:
                finish_reason = "length"
            else:
                finish_reason = "stop"

            results.append(
                GenerationOutput(
                    text=generated_text,
                    tokens=generated_tokens.tolist(),
                    num_tokens=len(generated_tokens),
                    finish_reason=finish_reason,
                    latency_ms=latency_ms,
                    ttft_ms=None,  # Not tracked in basic generate()
                )
            )

        return results

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        }

    def generate_streaming(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> Any:
        """Generate text with streaming output.

        Args:
            prompt: Input prompt
            generation_config: Generation configuration

        Yields:
            Generated tokens one at a time
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if generation_config is None:
            generation_config = GenerationConfig()

        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "do_sample": generation_config.do_sample,
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            yield text

        thread.join()
