"""Batching Strategies.

Support for various batching approaches:
- Static batching (fixed size, padded)
- Dynamic batching (group similar lengths)
- Continuous batching (vLLM-style iteration-level scheduling)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from strategies.base import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)


class BatchingMethod(Enum):
    """Available batching methods."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    CONTINUOUS = "continuous"


@dataclass
class BatchingConfig:
    """Configuration for batching strategy."""

    method: BatchingMethod = BatchingMethod.STATIC
    # Static batching
    batch_size: int = 8
    pad_to_max_length: bool = True
    max_length: int = 2048
    # Dynamic batching
    max_batch_size: int = 32
    max_wait_time_ms: int = 100
    length_bucket_step: int = 64
    # Continuous batching
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    iteration_level_scheduling: bool = True


class BatchingStrategy(BaseStrategy):
    """Batching strategy for inference optimization.

    Supports:
    - Static: Fixed batch size with padding
    - Dynamic: Group similar-length sequences
    - Continuous: vLLM-style iteration-level scheduling
    """

    def __init__(
        self,
        method: str | BatchingMethod = BatchingMethod.STATIC,
        batch_size: int = 8,
        max_batch_size: int = 32,
        config: StrategyConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize batching strategy.

        Args:
            method: Batching method to use
            batch_size: Batch size for static batching
            max_batch_size: Maximum batch size for dynamic/continuous
            config: Base strategy configuration
            **kwargs: Method-specific options
        """
        super().__init__(config)

        if isinstance(method, str):
            method = BatchingMethod(method.lower())

        self.batch_config = BatchingConfig(
            method=method,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            **{k: v for k, v in kwargs.items() if k in BatchingConfig.__dataclass_fields__},
        )

    @property
    def name(self) -> str:
        return f"batching_{self.batch_config.method.value}"

    def setup(self, model: Any, **kwargs: Any) -> Any:
        """Configure batching strategy.

        Args:
            model: Model to configure
            **kwargs: Additional options

        Returns:
            Configured model
        """
        method = self.batch_config.method

        if method == BatchingMethod.STATIC:
            return self._setup_static(model, **kwargs)
        elif method == BatchingMethod.DYNAMIC:
            return self._setup_dynamic(model, **kwargs)
        elif method == BatchingMethod.CONTINUOUS:
            return self._setup_continuous(model, **kwargs)
        else:
            logger.warning(f"Unknown batching method: {method}")
            return model

    def _setup_static(self, model: Any, **kwargs: Any) -> Any:
        """Configure static batching."""
        logger.info(
            f"Configuring static batching with batch_size={self.batch_config.batch_size}"
        )

        if hasattr(model, "config"):
            model.config.batching_config = {
                "method": "static",
                "batch_size": self.batch_config.batch_size,
                "pad_to_max_length": self.batch_config.pad_to_max_length,
                "max_length": self.batch_config.max_length,
            }

        self._is_applied = True
        return model

    def _setup_dynamic(self, model: Any, **kwargs: Any) -> Any:
        """Configure dynamic batching."""
        logger.info(
            f"Configuring dynamic batching with max_batch_size="
            f"{self.batch_config.max_batch_size}, "
            f"max_wait_time={self.batch_config.max_wait_time_ms}ms"
        )

        if hasattr(model, "config"):
            model.config.batching_config = {
                "method": "dynamic",
                "max_batch_size": self.batch_config.max_batch_size,
                "max_wait_time_ms": self.batch_config.max_wait_time_ms,
                "length_bucket_step": self.batch_config.length_bucket_step,
            }

        self._is_applied = True
        return model

    def _setup_continuous(self, model: Any, **kwargs: Any) -> Any:
        """Configure continuous batching (vLLM-style)."""
        logger.info(
            f"Configuring continuous batching with max_num_seqs="
            f"{self.batch_config.max_num_seqs}, "
            f"max_batched_tokens={self.batch_config.max_num_batched_tokens}"
        )

        if hasattr(model, "config"):
            model.config.batching_config = {
                "method": "continuous",
                "max_num_seqs": self.batch_config.max_num_seqs,
                "max_num_batched_tokens": self.batch_config.max_num_batched_tokens,
                "iteration_level_scheduling": self.batch_config.iteration_level_scheduling,
            }

        self._is_applied = True
        return model

    def describe(self) -> str:
        method = self.batch_config.method

        if method == BatchingMethod.STATIC:
            return (
                f"Batching: Static with batch_size={self.batch_config.batch_size}. "
                "Fixed batch size with padding. Simple but potentially inefficient."
            )

        elif method == BatchingMethod.DYNAMIC:
            return (
                f"Batching: Dynamic with max_batch_size={self.batch_config.max_batch_size}, "
                f"max_wait={self.batch_config.max_wait_time_ms}ms. "
                "Groups similar-length sequences to minimize padding."
            )

        elif method == BatchingMethod.CONTINUOUS:
            return (
                f"Batching: Continuous (vLLM-style) with max_seqs="
                f"{self.batch_config.max_num_seqs}. "
                "Iteration-level scheduling for maximum throughput."
            )

        return f"Batching: {method.value}"

    def create_batches(
        self,
        prompts: list[str],
        tokenizer: Any,
    ) -> list[list[str]]:
        """Create batches from prompts according to strategy.

        Args:
            prompts: List of prompt strings
            tokenizer: Tokenizer for length calculation

        Returns:
            List of batches (each batch is a list of prompts)
        """
        method = self.batch_config.method

        if method == BatchingMethod.STATIC:
            return self._create_static_batches(prompts)

        elif method == BatchingMethod.DYNAMIC:
            return self._create_dynamic_batches(prompts, tokenizer)

        else:
            # Continuous batching is handled by serving framework
            return self._create_static_batches(prompts)

    def _create_static_batches(self, prompts: list[str]) -> list[list[str]]:
        """Create fixed-size batches."""
        batch_size = self.batch_config.batch_size
        batches = []

        for i in range(0, len(prompts), batch_size):
            batches.append(prompts[i : i + batch_size])

        return batches

    def _create_dynamic_batches(
        self,
        prompts: list[str],
        tokenizer: Any,
    ) -> list[list[str]]:
        """Create batches grouping similar-length sequences."""
        # Tokenize to get lengths
        lengths = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            lengths.append(len(tokens))

        # Sort by length
        sorted_indices = sorted(range(len(prompts)), key=lambda i: lengths[i])

        # Create batches respecting max_batch_size
        batches = []
        current_batch = []

        for idx in sorted_indices:
            current_batch.append(prompts[idx])

            if len(current_batch) >= self.batch_config.max_batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        return batches

    def estimate_throughput_scaling(
        self,
        baseline_throughput: float,
        batch_sizes: list[int],
    ) -> dict[int, dict[str, float]]:
        """Estimate throughput scaling across batch sizes.

        Args:
            baseline_throughput: Throughput at batch_size=1
            batch_sizes: List of batch sizes to estimate

        Returns:
            Dictionary mapping batch_size to estimated metrics
        """
        results = {}

        for batch_size in batch_sizes:
            # Simple scaling model (actual scaling depends on many factors)
            # Assume sublinear scaling due to memory bandwidth limits
            scaling_factor = batch_size ** 0.7  # Sublinear scaling
            estimated_throughput = baseline_throughput * scaling_factor

            # Efficiency decreases with batch size due to padding overhead
            if self.batch_config.method == BatchingMethod.STATIC:
                efficiency = max(0.5, 1.0 - (batch_size * 0.01))
            else:
                efficiency = max(0.7, 1.0 - (batch_size * 0.005))

            results[batch_size] = {
                "estimated_throughput": estimated_throughput * efficiency,
                "scaling_factor": scaling_factor,
                "efficiency": efficiency,
                "ideal_throughput": baseline_throughput * batch_size,
            }

        return results
