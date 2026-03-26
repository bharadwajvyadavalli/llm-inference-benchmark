"""BaseStrategy: Abstract base class for optimization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""

    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "params": self.params,
        }


class BaseStrategy(ABC):
    """Abstract base class for inference optimization strategies.

    All strategies must implement:
    - name: Property returning the strategy name
    - setup: Apply the optimization to a model
    - describe: Human-readable description
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        """Initialize the strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config or StrategyConfig()
        self._is_applied = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass

    @abstractmethod
    def setup(self, model: Any, **kwargs: Any) -> Any:
        """Apply the optimization strategy to the model.

        Args:
            model: The model to optimize
            **kwargs: Additional configuration options

        Returns:
            The optimized model (may be the same object or a new one)
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """Return human-readable description of the strategy and its config."""
        pass

    def get_config(self) -> dict[str, Any]:
        """Return the strategy configuration as a dictionary."""
        return {
            "name": self.name,
            "config": self.config.to_dict(),
        }

    def is_compatible(self, model_config: dict[str, Any]) -> bool:
        """Check if this strategy is compatible with the given model.

        Args:
            model_config: Model configuration dictionary

        Returns:
            True if compatible, False otherwise
        """
        # Default: all strategies are compatible
        return True

    def teardown(self, model: Any) -> Any:
        """Remove the optimization from the model (if possible).

        Args:
            model: The optimized model

        Returns:
            The original model (may not always be possible)
        """
        self._is_applied = False
        return model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BaselineStrategy(BaseStrategy):
    """Baseline strategy with no optimizations applied.

    Used as a reference point for measuring the impact of other strategies.
    """

    @property
    def name(self) -> str:
        return "baseline"

    def setup(self, model: Any, **kwargs: Any) -> Any:
        """No-op setup for baseline."""
        self._is_applied = True
        return model

    def describe(self) -> str:
        return "Baseline: No optimizations applied. FP16/BF16 precision as loaded."


class CompositeStrategy(BaseStrategy):
    """Combine multiple strategies together.

    Applies strategies in order, allowing combinations like
    quantization + attention optimization.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        config: StrategyConfig | None = None,
    ) -> None:
        """Initialize composite strategy.

        Args:
            strategies: List of strategies to combine
            config: Configuration for the composite
        """
        super().__init__(config)
        self.strategies = strategies

    @property
    def name(self) -> str:
        strategy_names = "+".join(s.name for s in self.strategies)
        return f"composite({strategy_names})"

    def setup(self, model: Any, **kwargs: Any) -> Any:
        """Apply all strategies in sequence."""
        for strategy in self.strategies:
            model = strategy.setup(model, **kwargs)
        self._is_applied = True
        return model

    def describe(self) -> str:
        descriptions = [s.describe() for s in self.strategies]
        return "Composite strategy:\n" + "\n".join(f"  - {d}" for d in descriptions)

    def is_compatible(self, model_config: dict[str, Any]) -> bool:
        """Check if all strategies are compatible."""
        return all(s.is_compatible(model_config) for s in self.strategies)

    def teardown(self, model: Any) -> Any:
        """Teardown all strategies in reverse order."""
        for strategy in reversed(self.strategies):
            model = strategy.teardown(model)
        self._is_applied = False
        return model
