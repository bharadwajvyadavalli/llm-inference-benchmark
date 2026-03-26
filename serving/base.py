"""BaseServingBackend: Abstract interface for LLM serving backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "stop_sequences": self.stop_sequences,
        }


@dataclass
class GenerationOutput:
    """Output from text generation."""

    text: str
    tokens: list[int]
    num_tokens: int
    finish_reason: str
    latency_ms: float
    ttft_ms: float | None = None
    token_timestamps: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "tokens": self.tokens,
            "num_tokens": self.num_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "token_timestamps": self.token_timestamps,
        }


class BaseServingBackend(ABC):
    """Abstract base class for LLM serving backends.

    All backends must implement:
    - name: Property returning the backend name
    - load_model: Load a model with optional strategy config
    - generate: Run inference on prompts
    - get_memory_usage: Report current GPU memory state
    """

    def __init__(self) -> None:
        """Initialize the backend."""
        self._model = None
        self._tokenizer = None
        self._model_name: str | None = None
        self._is_loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name."""
        pass

    @abstractmethod
    def load_model(
        self,
        model_name: str,
        strategy_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Load a model with optional strategy configuration.

        Args:
            model_name: HuggingFace model path or local path
            strategy_config: Strategy configuration dictionary
            **kwargs: Additional backend-specific options
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[GenerationOutput]:
        """Generate text for the given prompts.

        Args:
            prompts: List of input prompts
            generation_config: Generation configuration
            **kwargs: Additional options

        Returns:
            List of GenerationOutput for each prompt
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> dict[str, Any]:
        """Get current GPU memory usage.

        Returns:
            Dictionary with memory statistics
        """
        pass

    def unload_model(self) -> None:
        """Unload the current model and free memory."""
        self._model = None
        self._tokenizer = None
        self._model_name = None
        self._is_loaded = False

        # Clear GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._is_loaded

    @property
    def model_name(self) -> str | None:
        """Get the name of the loaded model."""
        return self._model_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self._model_name!r})"
