"""Quality Metrics: Perplexity, task accuracy, quality degradation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Container for quality measurement results."""

    perplexity: float
    perplexity_delta: float  # Change from baseline
    task_accuracy: float
    task_accuracy_delta: float
    quality_score: float  # Normalized 0-100
    degradation_score: float  # Quality loss per speedup

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "perplexity": self.perplexity,
            "perplexity_delta": self.perplexity_delta,
            "task_accuracy": self.task_accuracy,
            "task_accuracy_delta": self.task_accuracy_delta,
            "quality_score": self.quality_score,
            "degradation_score": self.degradation_score,
        }


class QualityMetrics:
    """Compute quality metrics for model outputs.

    Metrics computed:
    - Perplexity on held-out eval set
    - Task accuracy (MMLU subset, HellaSwag subset)
    - Quality degradation score
    """

    def __init__(self, baseline_perplexity: float | None = None) -> None:
        """Initialize quality metrics collector.

        Args:
            baseline_perplexity: Baseline model perplexity for delta calculation
        """
        self.baseline_perplexity = baseline_perplexity
        self.baseline_accuracy: float | None = None
        self._perplexity_samples: list[float] = []

    def compute_perplexity(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
        max_length: int = 512,
    ) -> dict[str, float]:
        """Compute perplexity on a set of texts.

        Args:
            model: Language model
            tokenizer: Tokenizer for the model
            texts: List of text samples
            max_length: Maximum sequence length

        Returns:
            Dictionary with perplexity metrics
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                try:
                    # Tokenize
                    encodings = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    )

                    input_ids = encodings["input_ids"].to(model.device)

                    # Get loss
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss

                    # Accumulate
                    num_tokens = input_ids.size(1)
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens

                except Exception as e:
                    logger.warning(f"Error computing perplexity for sample: {e}")
                    continue

        if total_tokens == 0:
            return {"perplexity": float("inf"), "perplexity_delta": 0.0}

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Calculate delta from baseline
        delta = 0.0
        if self.baseline_perplexity is not None:
            delta = perplexity - self.baseline_perplexity

        self._perplexity_samples.append(perplexity)

        return {
            "perplexity": perplexity,
            "perplexity_delta": delta,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
        }

    def compute_task_accuracy(
        self,
        model: Any,
        tokenizer: Any,
        task: str = "mmlu",
        num_samples: int = 100,
    ) -> dict[str, float]:
        """Compute accuracy on a benchmark task.

        Args:
            model: Language model
            tokenizer: Tokenizer
            task: Task name ("mmlu" or "hellaswag")
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary with accuracy metrics
        """
        try:
            if task.lower() == "mmlu":
                return self._eval_mmlu(model, tokenizer, num_samples)
            elif task.lower() == "hellaswag":
                return self._eval_hellaswag(model, tokenizer, num_samples)
            else:
                logger.warning(f"Unknown task: {task}")
                return {"accuracy": 0.0}
        except Exception as e:
            logger.error(f"Error evaluating task {task}: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def _eval_mmlu(
        self,
        model: Any,
        tokenizer: Any,
        num_samples: int,
    ) -> dict[str, float]:
        """Evaluate on MMLU subset.

        This is a simplified evaluation. Full MMLU has many subjects.
        """
        try:
            from datasets import load_dataset

            # Load a subset of MMLU
            dataset = load_dataset(
                "cais/mmlu", "abstract_algebra", split=f"test[:{num_samples}]"
            )

            correct = 0
            total = 0

            model.eval()
            with torch.no_grad():
                for sample in dataset:
                    question = sample["question"]
                    choices = sample["choices"]
                    answer_idx = sample["answer"]

                    # Format as multiple choice
                    prompt = f"Question: {question}\n"
                    for i, choice in enumerate(choices):
                        prompt += f"{chr(65+i)}. {choice}\n"
                    prompt += "Answer:"

                    # Get model prediction
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                    )
                    response = tokenizer.decode(outputs[0][-1:])

                    # Check if correct
                    predicted = ord(response.strip().upper()) - 65
                    if predicted == answer_idx:
                        correct += 1
                    total += 1

            accuracy = correct / total if total > 0 else 0.0

            delta = 0.0
            if self.baseline_accuracy is not None:
                delta = accuracy - self.baseline_accuracy

            return {
                "accuracy": accuracy,
                "accuracy_delta": delta,
                "correct": correct,
                "total": total,
                "task": "mmlu",
            }

        except Exception as e:
            logger.warning(f"Could not load MMLU dataset: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def _eval_hellaswag(
        self,
        model: Any,
        tokenizer: Any,
        num_samples: int,
    ) -> dict[str, float]:
        """Evaluate on HellaSwag subset."""
        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "hellaswag", split=f"validation[:{num_samples}]"
            )

            correct = 0
            total = 0

            model.eval()
            with torch.no_grad():
                for sample in dataset:
                    context = sample["ctx"]
                    endings = sample["endings"]
                    label = int(sample["label"])

                    # Score each ending
                    scores = []
                    for ending in endings:
                        text = context + " " + ending
                        inputs = tokenizer(text, return_tensors="pt").to(model.device)
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        scores.append(-outputs.loss.item())

                    # Predict the one with highest score (lowest loss)
                    predicted = scores.index(max(scores))
                    if predicted == label:
                        correct += 1
                    total += 1

            accuracy = correct / total if total > 0 else 0.0

            delta = 0.0
            if self.baseline_accuracy is not None:
                delta = accuracy - self.baseline_accuracy

            return {
                "accuracy": accuracy,
                "accuracy_delta": delta,
                "correct": correct,
                "total": total,
                "task": "hellaswag",
            }

        except Exception as e:
            logger.warning(f"Could not load HellaSwag dataset: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def compute_quality_degradation(
        self,
        baseline_perplexity: float,
        optimized_perplexity: float,
        speedup: float,
    ) -> dict[str, float]:
        """Compute quality degradation score.

        Quality degradation = (perplexity increase) / speedup
        Lower is better - means less quality loss per unit speedup.

        Args:
            baseline_perplexity: Baseline model perplexity
            optimized_perplexity: Optimized model perplexity
            speedup: Speedup factor from optimization

        Returns:
            Dictionary with degradation metrics
        """
        perplexity_increase = optimized_perplexity - baseline_perplexity
        perplexity_increase_pct = (
            (perplexity_increase / baseline_perplexity) * 100
            if baseline_perplexity > 0
            else 0
        )

        # Degradation score: quality loss per speedup
        degradation = perplexity_increase_pct / speedup if speedup > 0 else 0

        # Quality score: 100 - degradation (capped at 0-100)
        quality_score = max(0, min(100, 100 - abs(degradation)))

        return {
            "perplexity_increase": perplexity_increase,
            "perplexity_increase_pct": perplexity_increase_pct,
            "speedup": speedup,
            "degradation_score": degradation,
            "quality_score": quality_score,
        }

    def set_baseline(
        self,
        perplexity: float | None = None,
        accuracy: float | None = None,
    ) -> None:
        """Set baseline metrics for delta calculations.

        Args:
            perplexity: Baseline perplexity
            accuracy: Baseline accuracy
        """
        if perplexity is not None:
            self.baseline_perplexity = perplexity
        if accuracy is not None:
            self.baseline_accuracy = accuracy

    def get_summary(self) -> dict[str, Any]:
        """Get summary of quality measurements.

        Returns:
            Dictionary with quality statistics
        """
        if not self._perplexity_samples:
            return {}

        return {
            "avg_perplexity": sum(self._perplexity_samples) / len(self._perplexity_samples),
            "min_perplexity": min(self._perplexity_samples),
            "max_perplexity": max(self._perplexity_samples),
            "num_samples": len(self._perplexity_samples),
            "baseline_perplexity": self.baseline_perplexity,
        }

    def reset(self) -> None:
        """Reset accumulated samples."""
        self._perplexity_samples = []
