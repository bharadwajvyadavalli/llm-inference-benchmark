"""Speculative Decoding Strategy.

Speculative decoding uses a smaller draft model to generate candidate tokens,
which are then verified by the larger target model in parallel. This can
significantly reduce latency when the draft model's predictions align well
with the target model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from strategies.base import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeDecodingConfig:
    """Configuration for speculative decoding."""

    draft_model_name: str | None = None
    draft_length: int = 5  # Number of tokens to speculatively generate
    acceptance_threshold: float = 0.9  # Minimum probability to accept draft token
    max_draft_tokens: int = 10
    temperature: float = 1.0


class SpeculativeDecodingStrategy(BaseStrategy):
    """Speculative decoding strategy using a draft model.

    This strategy trades extra compute (running two models) for lower latency
    by accepting multiple tokens per forward pass when the draft model's
    predictions align with what the target model would have generated.
    """

    # Default draft models for common target models
    DEFAULT_DRAFT_MODELS = {
        "llama-2-7b": "EleutherAI/pythia-160m",
        "llama-2-13b": "EleutherAI/pythia-410m",
        "mistral-7b": "EleutherAI/pythia-160m",
        "pythia-1b": "EleutherAI/pythia-70m",
        "pythia-410m": "EleutherAI/pythia-70m",
    }

    def __init__(
        self,
        draft_model_name: str | None = None,
        draft_length: int = 5,
        acceptance_threshold: float = 0.9,
        config: StrategyConfig | None = None,
    ) -> None:
        """Initialize speculative decoding strategy.

        Args:
            draft_model_name: HuggingFace path to draft model
            draft_length: Number of tokens to generate speculatively
            acceptance_threshold: Minimum probability to accept draft tokens
            config: Base strategy configuration
        """
        super().__init__(config)
        self.spec_config = SpeculativeDecodingConfig(
            draft_model_name=draft_model_name,
            draft_length=draft_length,
            acceptance_threshold=acceptance_threshold,
        )
        self._draft_model = None
        self._target_model = None

    @property
    def name(self) -> str:
        return "speculative_decoding"

    def setup(self, model: Any, **kwargs: Any) -> Any:
        """Set up speculative decoding with draft model.

        Args:
            model: Target model to apply speculative decoding to
            **kwargs: Additional options including:
                - draft_model: Pre-loaded draft model
                - model_name: Name of target model (for default draft selection)

        Returns:
            Configured model (may wrap original)
        """
        self._target_model = model

        # Get draft model name
        draft_model_name = self.spec_config.draft_model_name
        if draft_model_name is None:
            model_name = kwargs.get("model_name", "")
            draft_model_name = self._get_default_draft(model_name)

        if draft_model_name is None:
            logger.warning(
                "No draft model specified and no default available. "
                "Speculative decoding disabled."
            )
            self._is_applied = False
            return model

        # Load draft model if not provided
        draft_model = kwargs.get("draft_model")
        if draft_model is None:
            draft_model = self._load_draft_model(draft_model_name)

        self._draft_model = draft_model
        self._is_applied = True

        logger.info(
            f"Speculative decoding enabled with draft model: {draft_model_name}, "
            f"draft_length={self.spec_config.draft_length}"
        )

        # Return a wrapper or configured model
        # In practice, this would integrate with the serving backend
        return self._create_speculative_wrapper(model, draft_model)

    def _get_default_draft(self, model_name: str) -> str | None:
        """Get default draft model for a given target model."""
        model_name_lower = model_name.lower()
        for key, draft in self.DEFAULT_DRAFT_MODELS.items():
            if key in model_name_lower:
                return draft
        return None

    def _load_draft_model(self, model_name: str) -> Any:
        """Load draft model from HuggingFace.

        Args:
            model_name: HuggingFace model path

        Returns:
            Loaded model
        """
        try:
            from transformers import AutoModelForCausalLM

            logger.info(f"Loading draft model: {model_name}")
            draft_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            return draft_model
        except Exception as e:
            logger.error(f"Failed to load draft model {model_name}: {e}")
            raise

    def _create_speculative_wrapper(
        self, target_model: Any, draft_model: Any
    ) -> Any:
        """Create a wrapper that implements speculative decoding.

        This is a simplified implementation. In practice, you would use
        a serving framework's native speculative decoding support.
        """
        # Store configuration on the model for later use
        if hasattr(target_model, "config"):
            target_model.config.speculative_config = {
                "draft_model": draft_model,
                "draft_length": self.spec_config.draft_length,
                "acceptance_threshold": self.spec_config.acceptance_threshold,
            }

        return target_model

    def describe(self) -> str:
        draft = self.spec_config.draft_model_name or "auto-selected"
        return (
            f"Speculative Decoding: Draft model generates {self.spec_config.draft_length} "
            f"candidate tokens, verified by target model. "
            f"Draft model: {draft}, acceptance threshold: {self.spec_config.acceptance_threshold}"
        )

    def is_compatible(self, model_config: dict[str, Any]) -> bool:
        """Check if speculative decoding is compatible with model."""
        # Speculative decoding works best with autoregressive models
        architecture = model_config.get("architecture", "").lower()
        incompatible = ["encoder", "seq2seq", "t5"]
        return not any(inc in architecture for inc in incompatible)

    def teardown(self, model: Any) -> Any:
        """Clean up speculative decoding setup."""
        if self._draft_model is not None:
            # Free draft model memory
            del self._draft_model
            self._draft_model = None

        if hasattr(model, "config") and hasattr(model.config, "speculative_config"):
            delattr(model.config, "speculative_config")

        self._is_applied = False
        return model

    def get_metrics(self) -> dict[str, Any]:
        """Return speculative decoding specific metrics.

        Returns:
            Dictionary with acceptance rate, speedup, etc.
        """
        return {
            "draft_length": self.spec_config.draft_length,
            "acceptance_threshold": self.spec_config.acceptance_threshold,
            "draft_model": self.spec_config.draft_model_name,
            # These would be populated during inference
            "acceptance_rate": None,
            "speedup_ratio": None,
        }
