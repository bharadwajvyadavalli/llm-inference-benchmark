"""Tests for workload generators."""

import pytest


class TestWorkloadGenerator:
    """Tests for WorkloadGenerator factory."""

    def test_create_short_workload(self):
        """Test creating short prompt workload."""
        from benchmark.workloads import WorkloadGenerator

        gen = WorkloadGenerator(seed=42)
        workload = gen.create("short")

        assert workload.name == "short_prompt"
        prompts = workload.generate()
        assert len(prompts) > 0

    def test_create_long_workload(self):
        """Test creating long context workload."""
        from benchmark.workloads import WorkloadGenerator

        gen = WorkloadGenerator(seed=42)
        workload = gen.create("long")

        assert workload.name == "long_context"

    def test_create_all_workloads(self):
        """Test creating all workload types."""
        from benchmark.workloads import WorkloadGenerator

        gen = WorkloadGenerator(seed=42)
        workloads = gen.create_all()

        assert len(workloads) == 4
        names = [w.name for w in workloads]
        assert "short_prompt" in names
        assert "long_context" in names
        assert "batch" in names
        assert "multi_turn" in names

    def test_unknown_workload(self):
        """Test error on unknown workload type."""
        from benchmark.workloads import WorkloadGenerator

        gen = WorkloadGenerator()

        with pytest.raises(ValueError):
            gen.create("unknown_type")

    def test_list_workloads(self):
        """Test listing available workloads."""
        from benchmark.workloads import WorkloadGenerator

        workloads = WorkloadGenerator.list_workloads()

        assert "short" in workloads
        assert "long" in workloads
        assert "batch" in workloads


class TestShortPromptWorkload:
    """Tests for ShortPromptWorkload."""

    def test_generate_prompts(self):
        """Test generating short prompts."""
        from benchmark.workloads import ShortPromptWorkload

        workload = ShortPromptWorkload(n=10, seed=42)
        prompts = workload.generate()

        assert len(prompts) == 10
        for prompt in prompts:
            assert len(prompt.text) > 0
            assert prompt.expected_output_tokens > 0

    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        from benchmark.workloads import ShortPromptWorkload

        workload1 = ShortPromptWorkload(n=5, seed=42)
        workload2 = ShortPromptWorkload(n=5, seed=42)

        prompts1 = workload1.generate()
        prompts2 = workload2.generate()

        for p1, p2 in zip(prompts1, prompts2):
            assert p1.text == p2.text
            assert p1.category == p2.category

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        from benchmark.workloads import ShortPromptWorkload

        workload1 = ShortPromptWorkload(n=5, seed=42)
        workload2 = ShortPromptWorkload(n=5, seed=123)

        prompts1 = workload1.generate()
        prompts2 = workload2.generate()

        # At least some prompts should differ
        differences = sum(1 for p1, p2 in zip(prompts1, prompts2) if p1.text != p2.text)
        assert differences > 0


class TestLongContextWorkload:
    """Tests for LongContextWorkload."""

    def test_generate_long_prompts(self):
        """Test generating long context prompts."""
        from benchmark.workloads import LongContextWorkload

        workload = LongContextWorkload(
            n=5,
            min_context_length=1000,
            max_context_length=2000,
            seed=42,
        )
        prompts = workload.generate()

        assert len(prompts) == 5
        for prompt in prompts:
            # Check that context is within range
            assert len(prompt.text) >= 1000

    def test_context_length_metadata(self):
        """Test that context length is stored in metadata."""
        from benchmark.workloads import LongContextWorkload

        workload = LongContextWorkload(n=3, seed=42)
        prompts = workload.generate()

        for prompt in prompts:
            assert "context_length" in prompt.metadata


class TestBatchWorkload:
    """Tests for BatchWorkload."""

    def test_batch_sizes(self):
        """Test batch size configuration."""
        from benchmark.workloads import BatchWorkload

        workload = BatchWorkload(
            batch_sizes=[1, 2, 4, 8],
            seed=42,
        )

        assert workload.get_batch_sizes() == [1, 2, 4, 8]

    def test_generate_base_prompts(self):
        """Test generating base prompts for batching."""
        from benchmark.workloads import BatchWorkload

        workload = BatchWorkload(prompts_per_batch=5, seed=42)
        prompts = workload.generate()

        assert len(prompts) == 5


class TestMultiTurnWorkload:
    """Tests for MultiTurnWorkload."""

    def test_generate_conversations(self):
        """Test generating multi-turn conversations."""
        from benchmark.workloads import MultiTurnWorkload

        workload = MultiTurnWorkload(
            n=3,
            turns_per_conversation=4,
            seed=42,
        )
        prompts = workload.generate()

        # Should have n * turns_per_conversation prompts
        assert len(prompts) == 3 * 4

    def test_conversation_metadata(self):
        """Test that conversation metadata is correct."""
        from benchmark.workloads import MultiTurnWorkload

        workload = MultiTurnWorkload(n=2, turns_per_conversation=3, seed=42)
        prompts = workload.generate()

        # Check that turns are numbered correctly
        conv_indices = [p.metadata["conversation_index"] for p in prompts]
        assert 0 in conv_indices
        assert 1 in conv_indices

        turns = [p.metadata["turn"] for p in prompts]
        assert 0 in turns
        assert 1 in turns
        assert 2 in turns
