"""WorkloadGenerator: Generate reproducible benchmark workloads."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Prompt:
    """A single prompt for benchmarking."""

    text: str
    expected_output_tokens: int
    category: str
    metadata: dict[str, Any] | None = None


class BaseWorkload(ABC):
    """Base class for workload generators."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize workload generator with seed for reproducibility."""
        self.seed = seed
        self.rng = random.Random(seed)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return workload name."""
        pass

    @abstractmethod
    def generate(self) -> list[Prompt]:
        """Generate the workload prompts."""
        pass

    def describe(self) -> str:
        """Return human-readable description of the workload."""
        return f"{self.name} workload"


class ShortPromptWorkload(BaseWorkload):
    """Short prompt workload simulating chatbot interactions."""

    # Sample short prompts covering different categories
    PROMPT_TEMPLATES = {
        "qa": [
            "What is the capital of France?",
            "How many planets are in our solar system?",
            "What year did World War II end?",
            "Who wrote Romeo and Juliet?",
            "What is the largest mammal on Earth?",
            "What is the speed of light?",
            "Who invented the telephone?",
            "What is the chemical symbol for gold?",
            "How many continents are there?",
            "What is the tallest mountain in the world?",
        ],
        "instruction": [
            "Explain how photosynthesis works in simple terms.",
            "List three benefits of regular exercise.",
            "Describe the process of making coffee.",
            "Explain what machine learning is to a beginner.",
            "Give me tips for better sleep.",
            "Explain the difference between weather and climate.",
            "Describe how a car engine works.",
            "List the steps to change a tire.",
            "Explain how vaccines work.",
            "Describe the water cycle.",
        ],
        "chat": [
            "Hello! How are you today?",
            "What do you think about artificial intelligence?",
            "Can you help me plan a trip?",
            "Tell me something interesting.",
            "What's your favorite book recommendation?",
            "How can I learn a new language effectively?",
            "What should I cook for dinner tonight?",
            "Can you explain a complex topic simply?",
            "What are some good productivity tips?",
            "How do I stay motivated?",
        ],
    }

    def __init__(
        self,
        n: int = 100,
        min_output_tokens: int = 50,
        max_output_tokens: int = 100,
        seed: int = 42,
    ) -> None:
        """Initialize short prompt workload.

        Args:
            n: Number of prompts to generate
            min_output_tokens: Minimum expected output tokens
            max_output_tokens: Maximum expected output tokens
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.n = n
        self.min_output_tokens = min_output_tokens
        self.max_output_tokens = max_output_tokens

    @property
    def name(self) -> str:
        return "short_prompt"

    def generate(self) -> list[Prompt]:
        """Generate short prompt workload."""
        prompts = []
        categories = list(self.PROMPT_TEMPLATES.keys())

        for i in range(self.n):
            category = self.rng.choice(categories)
            template = self.rng.choice(self.PROMPT_TEMPLATES[category])

            expected_tokens = self.rng.randint(
                self.min_output_tokens, self.max_output_tokens
            )

            prompts.append(
                Prompt(
                    text=template,
                    expected_output_tokens=expected_tokens,
                    category=category,
                    metadata={"index": i},
                )
            )

        return prompts

    def describe(self) -> str:
        return (
            f"Short prompt workload: {self.n} prompts, "
            f"{self.min_output_tokens}-{self.max_output_tokens} expected output tokens"
        )


class LongContextWorkload(BaseWorkload):
    """Long context workload for document QA scenarios."""

    # Sample document snippets to build context
    DOCUMENT_SNIPPETS = [
        """Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience without being explicitly
        programmed. It focuses on developing algorithms that can access data
        and use it to learn for themselves.""",
        """The history of computing dates back to ancient civilizations with
        devices like the abacus. Modern computing began with Charles Babbage's
        analytical engine in the 19th century, followed by electronic computers
        in the mid-20th century.""",
        """Climate change refers to long-term shifts in global temperatures and
        weather patterns. While natural factors have contributed to climate
        changes throughout Earth's history, human activities have been the main
        driver of climate change since the 1800s.""",
        """Quantum computing harnesses quantum mechanical phenomena such as
        superposition and entanglement to process information. Unlike classical
        computers that use bits, quantum computers use quantum bits or qubits,
        which can exist in multiple states simultaneously.""",
        """The human brain contains approximately 86 billion neurons, each
        connected to thousands of other neurons through synapses. This vast
        network enables complex cognitive functions including memory, reasoning,
        and consciousness.""",
    ]

    QUESTIONS = [
        "What is the main topic discussed in this document?",
        "Summarize the key points from the text above.",
        "What are the implications of the information presented?",
        "How does this relate to current technological trends?",
        "What conclusions can be drawn from this passage?",
    ]

    def __init__(
        self,
        n: int = 50,
        min_context_length: int = 4096,
        max_context_length: int = 16384,
        min_output_tokens: int = 200,
        max_output_tokens: int = 500,
        seed: int = 42,
    ) -> None:
        """Initialize long context workload.

        Args:
            n: Number of prompts to generate
            min_context_length: Minimum context length in characters
            max_context_length: Maximum context length in characters
            min_output_tokens: Minimum expected output tokens
            max_output_tokens: Maximum expected output tokens
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.n = n
        self.min_context_length = min_context_length
        self.max_context_length = max_context_length
        self.min_output_tokens = min_output_tokens
        self.max_output_tokens = max_output_tokens

    @property
    def name(self) -> str:
        return "long_context"

    def _build_context(self, target_length: int) -> str:
        """Build a context of approximately target_length characters."""
        context_parts = []
        current_length = 0

        while current_length < target_length:
            snippet = self.rng.choice(self.DOCUMENT_SNIPPETS)
            context_parts.append(snippet)
            current_length += len(snippet)

            # Add some padding text
            padding = f"\n\nSection {len(context_parts)}:\n"
            context_parts.append(padding)
            current_length += len(padding)

        return "".join(context_parts)[:target_length]

    def generate(self) -> list[Prompt]:
        """Generate long context workload."""
        prompts = []

        for i in range(self.n):
            target_length = self.rng.randint(
                self.min_context_length, self.max_context_length
            )
            context = self._build_context(target_length)
            question = self.rng.choice(self.QUESTIONS)

            prompt_text = f"Document:\n{context}\n\nQuestion: {question}"

            expected_tokens = self.rng.randint(
                self.min_output_tokens, self.max_output_tokens
            )

            prompts.append(
                Prompt(
                    text=prompt_text,
                    expected_output_tokens=expected_tokens,
                    category="document_qa",
                    metadata={
                        "index": i,
                        "context_length": len(context),
                    },
                )
            )

        return prompts

    def describe(self) -> str:
        return (
            f"Long context workload: {self.n} prompts, "
            f"{self.min_context_length}-{self.max_context_length} context chars, "
            f"{self.min_output_tokens}-{self.max_output_tokens} expected output tokens"
        )


class BatchWorkload(BaseWorkload):
    """Batch workload for testing throughput scaling."""

    def __init__(
        self,
        batch_sizes: list[int] | None = None,
        prompts_per_batch: int = 10,
        seed: int = 42,
    ) -> None:
        """Initialize batch workload.

        Args:
            batch_sizes: List of batch sizes to test
            prompts_per_batch: Number of unique prompts per batch configuration
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.batch_sizes = batch_sizes or [1, 4, 8, 16, 32, 64]
        self.prompts_per_batch = prompts_per_batch
        self._base_workload = ShortPromptWorkload(n=prompts_per_batch, seed=seed)

    @property
    def name(self) -> str:
        return "batch"

    def generate(self) -> list[Prompt]:
        """Generate batch workload."""
        return self._base_workload.generate()

    def get_batch_sizes(self) -> list[int]:
        """Return the batch sizes to test."""
        return self.batch_sizes

    def describe(self) -> str:
        return (
            f"Batch workload: batch sizes {self.batch_sizes}, "
            f"{self.prompts_per_batch} prompts per batch"
        )


class MultiTurnWorkload(BaseWorkload):
    """Multi-turn conversation workload for KV cache pressure testing."""

    CONVERSATION_STARTERS = [
        "Let's discuss the future of technology.",
        "I'd like to learn about history.",
        "Can we talk about science?",
        "Tell me about different cultures.",
        "Let's explore philosophy together.",
    ]

    FOLLOW_UP_TEMPLATES = [
        "That's interesting. Can you elaborate on that?",
        "What are the implications of this?",
        "How does this compare to other perspectives?",
        "Can you give me a specific example?",
        "What would happen if we applied this differently?",
    ]

    def __init__(
        self,
        n: int = 30,
        turns_per_conversation: int = 5,
        seed: int = 42,
    ) -> None:
        """Initialize multi-turn workload.

        Args:
            n: Number of conversations to generate
            turns_per_conversation: Number of turns per conversation
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.n = n
        self.turns_per_conversation = turns_per_conversation

    @property
    def name(self) -> str:
        return "multi_turn"

    def generate(self) -> list[Prompt]:
        """Generate multi-turn conversation prompts.

        Returns prompts representing full conversations with growing context.
        """
        prompts = []

        for conv_idx in range(self.n):
            conversation_history = []
            starter = self.rng.choice(self.CONVERSATION_STARTERS)
            conversation_history.append(f"User: {starter}")

            for turn in range(self.turns_per_conversation):
                # Build the full conversation prompt
                full_prompt = "\n".join(conversation_history)
                full_prompt += "\nAssistant:"

                prompts.append(
                    Prompt(
                        text=full_prompt,
                        expected_output_tokens=100,
                        category="conversation",
                        metadata={
                            "conversation_index": conv_idx,
                            "turn": turn,
                            "context_turns": len(conversation_history),
                        },
                    )
                )

                # Simulate assistant response for next turn
                assistant_response = f"Assistant: [Response to turn {turn + 1}]"
                conversation_history.append(assistant_response)

                # Add follow-up user message
                if turn < self.turns_per_conversation - 1:
                    follow_up = self.rng.choice(self.FOLLOW_UP_TEMPLATES)
                    conversation_history.append(f"User: {follow_up}")

        return prompts

    def describe(self) -> str:
        return (
            f"Multi-turn workload: {self.n} conversations, "
            f"{self.turns_per_conversation} turns each"
        )


class WorkloadGenerator:
    """Factory for creating workload instances."""

    WORKLOAD_CLASSES = {
        "short": ShortPromptWorkload,
        "short_prompt": ShortPromptWorkload,
        "long": LongContextWorkload,
        "long_context": LongContextWorkload,
        "batch": BatchWorkload,
        "multi_turn": MultiTurnWorkload,
    }

    def __init__(self, seed: int = 42) -> None:
        """Initialize workload generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed

    def create(
        self,
        workload_type: str,
        **kwargs: Any,
    ) -> BaseWorkload:
        """Create a workload instance.

        Args:
            workload_type: Type of workload to create
            **kwargs: Additional arguments for workload constructor

        Returns:
            Workload instance

        Raises:
            ValueError: If workload_type is unknown
        """
        workload_type = workload_type.lower()
        if workload_type not in self.WORKLOAD_CLASSES:
            raise ValueError(
                f"Unknown workload type: {workload_type}. "
                f"Available: {list(self.WORKLOAD_CLASSES.keys())}"
            )

        cls = self.WORKLOAD_CLASSES[workload_type]
        return cls(seed=self.seed, **kwargs)

    def create_all(self) -> list[BaseWorkload]:
        """Create instances of all workload types."""
        return [
            ShortPromptWorkload(seed=self.seed),
            LongContextWorkload(seed=self.seed),
            BatchWorkload(seed=self.seed),
            MultiTurnWorkload(seed=self.seed),
        ]

    @classmethod
    def list_workloads(cls) -> list[str]:
        """Return list of available workload types."""
        return list(cls.WORKLOAD_CLASSES.keys())
