# Prompt 1: llm-inference-benchmark

Create a GitHub repo called `llm-inference-benchmark` — a comprehensive benchmarking suite that stress-tests LLM serving and inference optimization strategies across model sizes, hardware, and providers. This is NOT a tutorial or wrapper — it is an original benchmarking tool that produces actionable performance data.

## REPO STRUCTURE

```
llm-inference-benchmark/
├── README.md
├── LICENSE (MIT)
├── pyproject.toml
├── .gitignore (Python)
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── benchmark_config.yaml       # Global benchmark settings
│   ├── models.yaml                 # Model registry (sizes, families, HF paths)
│   └── hardware_profiles.yaml      # GPU specs for normalization (A100, H100, RTX 4090, etc.)
├── benchmark/
│   ├── __init__.py
│   ├── runner.py                   # BenchmarkRunner: orchestrates all benchmark suites
│   ├── profiler.py                 # GPU profiler: memory, utilization, power draw, FLOPS
│   └── workloads.py                # Workload generators: short prompts, long context, batch, chat
├── strategies/
│   ├── __init__.py
│   ├── base.py                     # BaseStrategy interface
│   ├── speculative_decoding.py     # Speculative decoding with draft models
│   ├── kv_cache.py                 # KV cache strategies: paged attention, sliding window, eviction policies
│   ├── quantization.py             # Quantization: GPTQ, AWQ, bitsandbytes (4/8-bit), GGUF
│   ├── batching.py                 # Batching strategies: static, dynamic, continuous batching
│   ├── attention.py                # Attention optimizations: FlashAttention-2, GQA, MQA, ring attention
│   └── parallel.py                 # Parallelism: tensor parallel, pipeline parallel, sequence parallel
├── metrics/
│   ├── __init__.py
│   ├── latency.py                  # TTFT, per-token latency, end-to-end latency, P50/P95/P99
│   ├── throughput.py               # Tokens/sec (generation + prefill), requests/sec, batch throughput
│   ├── memory.py                   # Peak GPU memory, KV cache memory, memory efficiency ratio
│   ├── quality.py                  # Perplexity degradation, accuracy on benchmarks pre/post optimization
│   └── cost.py                     # $/1M tokens, tokens per GPU-hour, cost-quality Pareto frontier
├── analysis/
│   ├── __init__.py
│   ├── report.py                   # Generate markdown + HTML benchmark reports
│   ├── plots.py                    # Visualization: latency curves, throughput scaling, memory breakdown, Pareto plots
│   ├── comparisons.py              # Head-to-head strategy comparisons with statistical significance
│   └── pareto.py                   # Pareto frontier analysis: quality vs. latency vs. cost
├── serving/
│   ├── __init__.py
│   ├── vllm_backend.py             # vLLM serving backend wrapper
│   ├── tgi_backend.py              # HuggingFace TGI backend wrapper
│   ├── hf_backend.py               # Raw HuggingFace transformers generate() baseline
│   └── base.py                     # BaseServingBackend interface
├── scripts/
│   ├── run_benchmark.py            # Main CLI: run full or selective benchmark suite
│   ├── run_single_strategy.py      # Benchmark a single strategy in isolation
│   ├── compare_strategies.py       # Head-to-head comparison between two strategies
│   └── generate_report.py          # Generate report from saved benchmark results
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_speculative_decoding_analysis.ipynb
│   ├── 03_quantization_tradeoffs.ipynb
│   ├── 04_kv_cache_deep_dive.ipynb
│   └── 05_cost_optimization.ipynb
└── tests/
    ├── test_profiler.py
    ├── test_metrics.py
    ├── test_strategies.py
    └── test_workloads.py
```

## README.md REQUIREMENTS

- Title: "llm-inference-benchmark"
- Subtitle: "Stress-testing LLM inference: speculative decoding, KV cache optimization, quantization, batching, and attention strategies — measured on real hardware with real workloads"
- Architecture diagram (mermaid) showing:
  ```
  Workload Generator → Serving Backend (vLLM / TGI / HF)
       ↓                        ↓
  Strategy Layer          GPU Profiler
  (speculative, quant,        ↓
   KV cache, batching)   Metrics Collector
       ↓                  (latency, throughput,
  Benchmark Runner         memory, quality, cost)
       ↓                        ↓
  Analysis Engine → Report Generator → Plots + Tables + Pareto Frontiers
  ```
- Section: **Strategies Benchmarked** — one paragraph per strategy explaining what it optimizes and the tradeoff:
  1. Speculative Decoding — draft model generates candidates, target model verifies in parallel. Trades extra compute for lower latency.
  2. KV Cache Optimization — paged attention (vLLM-style), sliding window, token eviction policies. Trades memory management overhead for lower memory footprint.
  3. Quantization — GPTQ, AWQ, bitsandbytes 4/8-bit, GGUF. Trades precision for memory savings and throughput gains. Key question: at what point does quality degrade?
  4. Batching — static, dynamic, continuous batching. Trades per-request latency for aggregate throughput.
  5. Attention Optimizations — FlashAttention-2, GQA, MQA. Trades implementation complexity for memory and compute savings.
  6. Parallelism — tensor parallel, pipeline parallel. Trades communication overhead for multi-GPU scaling.
- Section: **Metrics** — describe each metric category and why it matters
- Section: **Workloads** — describe test workload types: short prompts (chatbot), long context (document QA), batch processing (offline), multi-turn conversation
- Section: **Supported Models** — Llama-2-7B/13B, Mistral-7B, Pythia (70M-1B for scaling), Phi-2
- Section: **Quick Start** — CLI command to run a basic benchmark, 5 lines
- Section: **Results** — placeholder tables showing strategy × metric matrix
- Tech stack: PyTorch, vLLM, HuggingFace Transformers, TGI, bitsandbytes, auto-gptq, autoawq, Flash Attention, matplotlib, plotly

## IMPLEMENTATION DETAILS

### `benchmark/runner.py` — BenchmarkRunner
- `run_suite(strategies, models, workloads, backends)`: Cartesian product of all combinations. For each: warm up (3 runs), then measure (10 runs). Collect all metrics. Save raw results to JSON.
- `run_single(strategy, model, workload, backend)`: Single benchmark run with full profiling.
- Support `--strategy`, `--model`, `--workload`, `--backend` CLI filters to run subsets.
- Automatic CUDA synchronization for accurate timing. Proper warmup handling.
- Save intermediate results so long benchmark suites can be resumed after crash.

### `benchmark/profiler.py` — GPUProfiler
- `start()` / `stop()` / `snapshot()`: Track GPU memory (allocated, reserved, peak), GPU utilization %, power draw (via nvidia-smi or pynvml), compute FLOPS estimate.
- Context manager interface: `with GPUProfiler() as p: ...`
- `memory_breakdown()`: Distinguish model weights, KV cache, activations, optimizer states.
- Works with single and multi-GPU setups.

### `benchmark/workloads.py` — WorkloadGenerator
- `short_prompt_workload(n=100)`: Generate 100 short prompts (1-3 sentences). Mix of QA, instruction, chat. Target: 50-100 output tokens.
- `long_context_workload(n=50)`: Generate prompts with 4K-16K context windows (document + question). Target: 200-500 output tokens.
- `batch_workload(batch_sizes=[1, 4, 8, 16, 32, 64])`: Same prompts at different batch sizes to measure throughput scaling.
- `multi_turn_workload(n=30, turns=5)`: Multi-turn conversations with growing context. Measures KV cache pressure.
- All workloads are deterministic (seeded) for reproducibility.

### `strategies/` — Strategy Implementations
Each strategy implements `BaseStrategy`:
- `name` property
- `setup(model, config)`: Apply the optimization
- `describe()`: Human-readable description of what this strategy does and its config

#### `speculative_decoding.py`
- Configure draft model (smaller model from same family, e.g., Pythia-70M drafting for Pythia-1B).
- Configurable: draft length (number of speculative tokens), acceptance threshold, draft model name.
- Benchmark: measure acceptance rate, speedup ratio, quality preservation.

#### `kv_cache.py`
- Strategies: no optimization (baseline), paged attention (vLLM-style), sliding window (configurable window size), H2O-style heavy hitter eviction.
- Measure: memory usage across sequence lengths, latency impact, quality degradation at long contexts.

#### `quantization.py`
- Support: FP16 (baseline), INT8 (bitsandbytes), INT4 (bitsandbytes NF4), GPTQ (4-bit), AWQ (4-bit).
- For each: measure model size reduction, inference speedup, perplexity change on a held-out eval set.
- `quantize_model(model, method, bits)`: Apply quantization and return quantized model.

#### `batching.py`
- Static batching: fixed batch size, padded.
- Dynamic batching: group similar-length sequences, pad minimally.
- Continuous batching: vLLM-style iteration-level scheduling.
- Measure throughput scaling curves: tokens/sec vs. batch size for each strategy.

#### `attention.py`
- FlashAttention-2: enable/disable and measure speedup + memory savings.
- GQA vs MQA vs MHA: compare architectures where models support it.
- Benchmark across sequence lengths: 512, 1K, 2K, 4K, 8K, 16K.

### `metrics/` — Metric Collectors
Each metric module implements collection + aggregation:

#### `latency.py`
- Time to First Token (TTFT): time from request to first generated token.
- Per-token latency: inter-token interval during generation.
- End-to-end latency: total request time.
- Report: mean, P50, P95, P99, std deviation for each.

#### `throughput.py`
- Generation tokens/sec: output tokens per second.
- Prefill tokens/sec: input processing speed.
- Requests/sec: at various batch sizes.
- Throughput scaling efficiency: throughput at batch N / (N × throughput at batch 1).

#### `memory.py`
- Peak GPU memory (torch.cuda.max_memory_allocated).
- KV cache memory footprint: estimate from model config × sequence length × batch size.
- Memory efficiency: useful compute memory / total allocated memory.
- Memory vs. sequence length curves.

#### `quality.py`
- Perplexity on a held-out eval set (WikiText-2 or C4 subset): measure before and after each optimization.
- Task accuracy: run a small benchmark (MMLU 5-shot subset, HellaSwag subset) pre/post optimization.
- Quality degradation score: normalized metric showing how much quality was lost for a given speedup.

#### `cost.py`
- $/1M tokens: estimate based on GPU rental costs (configurable $/GPU-hour) and measured throughput.
- Tokens per GPU-hour.
- Pareto efficiency: identify strategies that are not dominated on the quality-cost-latency frontier.

### `analysis/` — Report Generation

#### `report.py`
- `generate_markdown(results)`: Full markdown report with tables, key findings, and recommendations.
- `generate_html(results)`: Interactive HTML report with embedded Plotly charts.

#### `plots.py`
- `latency_comparison_chart(results)`: Grouped bar chart of TTFT and per-token latency across strategies.
- `throughput_scaling_chart(results)`: Line chart of throughput vs. batch size per strategy.
- `memory_breakdown_chart(results)`: Stacked bar chart showing weights/KV cache/activations/overhead.
- `quality_vs_speedup_scatter(results)`: Scatter plot with strategy labels, quality on Y, speedup on X.
- `pareto_frontier_plot(results)`: 2D Pareto frontier (quality vs. cost or quality vs. latency) with dominated strategies grayed out.
- All plots: save as PNG + interactive Plotly HTML.

#### `pareto.py`
- `compute_pareto_frontier(results, objectives)`: Given multi-objective results, compute the Pareto-optimal set.
- `recommend_strategy(results, priority)`: Given user priority ("latency", "cost", "quality"), recommend the best strategy from the Pareto set.

### `serving/` — Backend Wrappers
Each backend wraps a serving framework with a common interface:
- `load_model(model_name, strategy_config)`: Load model with specified optimizations.
- `generate(prompts, generation_config)`: Run inference, return outputs + timing data.
- `get_memory_usage()`: Current GPU memory state.

- `vllm_backend.py`: Wrap vLLM's LLM class. Support paged attention, continuous batching natively.
- `tgi_backend.py`: Wrap TGI client (assumes TGI server is running). Support Flash decoding, quantization.
- `hf_backend.py`: Raw HuggingFace `model.generate()`. The unoptimized baseline.

## pyproject.toml

Dependencies: torch, transformers, vllm, bitsandbytes, auto-gptq, autoawq, accelerate, datasets, pynvml, matplotlib, plotly, seaborn, pandas, numpy, pyyaml, pytest, tqdm.
Optional: `[dev]` (pytest, ruff, black), `[tgi]` (text-generation), `[notebooks]` (jupyter).

## CONFIG FILES

- `benchmark_config.yaml`: warmup_runs, measurement_runs, output_dir, save_intermediate, random_seed, gpu_cost_per_hour (for cost estimates)
- `models.yaml`: list of models with name, hf_path, family, size_params, context_length, default_generation_config
- `hardware_profiles.yaml`: GPU specs — name, memory_gb, memory_bandwidth_gbps, fp16_tflops, tensor_cores (used for normalizing results across hardware)

## SCRIPTS

- `run_benchmark.py`: Main CLI with argparse. `--strategies all|speculative,quantization` `--models all|llama-7b,mistral-7b` `--workloads all|short,long` `--backends all|vllm,hf` `--output results/`
- `run_single_strategy.py`: Quick single-strategy benchmark for development/debugging.
- `compare_strategies.py`: Head-to-head between exactly two strategies with statistical significance testing (paired t-test on latency distributions).
- `generate_report.py`: Load saved results JSON, generate full markdown + HTML report.

## TESTS

- Test profiler: mock GPU stats, verify memory tracking.
- Test metrics: verify latency aggregation math (P50/P95/P99 calculations).
- Test workloads: verify deterministic generation, correct token length ranges.
- Test strategies: verify quantization config application, speculative decoding setup.

## GENERAL REQUIREMENTS

- Type hints everywhere.
- All benchmarks must be reproducible: fixed seeds, deterministic workloads, logged environment info (GPU model, driver version, CUDA version, library versions).
- Results saved as structured JSON with full metadata for later analysis.
- Support single-GPU and multi-GPU benchmarking.
- Graceful fallback: if a strategy isn't compatible with a model/backend combo, log a warning and skip (don't crash the full suite).
