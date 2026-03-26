"""TGI Backend: HuggingFace Text Generation Inference client."""

from __future__ import annotations

import logging
import time
from typing import Any

from serving.base import BaseServingBackend, GenerationConfig, GenerationOutput

logger = logging.getLogger(__name__)


class TGIBackend(BaseServingBackend):
    """HuggingFace Text Generation Inference (TGI) backend.

    This backend connects to a running TGI server. The server must be
    started separately with the model loaded.

    TGI features:
    - Flash decoding
    - Continuous batching
    - Quantization support
    - Multi-GPU tensor parallelism
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        timeout: float = 60.0,
    ) -> None:
        """Initialize TGI backend.

        Args:
            server_url: URL of the TGI server
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._client = None

    @property
    def name(self) -> str:
        return "tgi"

    def load_model(
        self,
        model_name: str,
        strategy_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Connect to TGI server.

        Note: The model is loaded by the TGI server, not this client.
        This method verifies connectivity and stores model info.

        Args:
            model_name: Model name (for reference only, server has its own model)
            strategy_config: Strategy configuration (informational)
            **kwargs: Additional options
        """
        try:
            from text_generation import Client
        except ImportError:
            raise ImportError(
                "text-generation not installed. Install with: pip install text-generation"
            )

        logger.info(f"Connecting to TGI server at {self.server_url}")

        # Create client
        self._client = Client(self.server_url, timeout=self.timeout)

        # Test connection
        try:
            info = self._client.info()
            self._model_name = info.model_id
            logger.info(f"Connected to TGI server with model: {self._model_name}")
        except Exception as e:
            logger.warning(f"Could not get server info: {e}")
            self._model_name = model_name

        self._is_loaded = True

    def generate(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[GenerationOutput]:
        """Generate text using TGI server.

        Args:
            prompts: List of input prompts
            generation_config: Generation configuration
            **kwargs: Additional options

        Returns:
            List of GenerationOutput
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("Not connected to TGI server. Call load_model() first.")

        if generation_config is None:
            generation_config = GenerationConfig()

        results = []

        for prompt in prompts:
            start_time = time.perf_counter()

            try:
                response = self._client.generate(
                    prompt,
                    max_new_tokens=generation_config.max_new_tokens,
                    temperature=generation_config.temperature,
                    top_p=generation_config.top_p,
                    top_k=generation_config.top_k,
                    repetition_penalty=generation_config.repetition_penalty,
                    do_sample=generation_config.do_sample,
                )

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                # Extract details from response
                generated_text = response.generated_text

                # TGI provides detailed token info
                details = response.details
                tokens = []
                if details and hasattr(details, "tokens"):
                    tokens = [t.id for t in details.tokens]

                finish_reason = "stop"
                if details:
                    finish_reason = details.finish_reason.value if hasattr(details.finish_reason, "value") else str(details.finish_reason)

                results.append(
                    GenerationOutput(
                        text=generated_text,
                        tokens=tokens,
                        num_tokens=len(tokens) if tokens else len(generated_text.split()),
                        finish_reason=finish_reason,
                        latency_ms=latency_ms,
                        ttft_ms=None,
                    )
                )

            except Exception as e:
                logger.error(f"TGI generation failed: {e}")
                results.append(
                    GenerationOutput(
                        text="",
                        tokens=[],
                        num_tokens=0,
                        finish_reason="error",
                        latency_ms=0,
                    )
                )

        return results

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
            Generated text chunks
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("Not connected to TGI server.")

        if generation_config is None:
            generation_config = GenerationConfig()

        for response in self._client.generate_stream(
            prompt,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            do_sample=generation_config.do_sample,
        ):
            if response.token:
                yield response.token.text

    def get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage from TGI server.

        Note: TGI doesn't expose detailed memory info via API.
        Returns server info instead.
        """
        if self._client is None:
            return {"available": False}

        try:
            info = self._client.info()
            return {
                "available": True,
                "model_id": info.model_id,
                "model_dtype": info.model_dtype,
                "max_input_length": info.max_input_length,
                "max_total_tokens": info.max_total_tokens,
                "max_batch_total_tokens": getattr(info, "max_batch_total_tokens", None),
            }
        except Exception as e:
            logger.warning(f"Could not get TGI server info: {e}")
            return {"available": False, "error": str(e)}

    def get_server_info(self) -> dict[str, Any]:
        """Get detailed TGI server information."""
        if self._client is None:
            return {}

        try:
            info = self._client.info()
            return {
                "model_id": info.model_id,
                "model_dtype": info.model_dtype,
                "model_device_type": info.model_device_type,
                "model_pipeline_tag": info.model_pipeline_tag,
                "max_best_of": info.max_best_of,
                "max_stop_sequences": info.max_stop_sequences,
                "max_input_length": info.max_input_length,
                "max_total_tokens": info.max_total_tokens,
            }
        except Exception as e:
            return {"error": str(e)}

    def health_check(self) -> bool:
        """Check if the TGI server is healthy.

        Returns:
            True if server is responsive
        """
        if self._client is None:
            return False

        try:
            self._client.info()
            return True
        except Exception:
            return False

    def unload_model(self) -> None:
        """Close connection to TGI server."""
        self._client = None
        super().unload_model()
