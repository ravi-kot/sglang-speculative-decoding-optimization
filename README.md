# SGLang Inference Experiments

Reproducible GPU inference experiments built around SGLang, with technical write-ups, source changes, profiler evidence, benchmark data, and reproduction scripts.

## Experiments

| Experiment | Focus | Published blog | GitHub source |
| --- | --- | --- | --- |
| Small-M RMSNorm | One-row-per-block mapping for fused-add Gemma RMSNorm | [Open blog](https://ravi-kot.github.io/sglang-speculative-decoding-optimization/experiments/rmsnorm/) | [README.md](https://github.com/ravi-kot/sglang-speculative-decoding-optimization/blob/main/experiments/rmsnorm/README.md) · [All files](https://github.com/ravi-kot/sglang-speculative-decoding-optimization/tree/main/experiments/rmsnorm) |
| Speculative decoding | Improving single-request speculative decoding on an RTX 5070 Laptop GPU | [Open blog](https://ravi-kot.github.io/sglang-speculative-decoding-optimization/experiments/speculative-decoding/) | [article.md](https://github.com/ravi-kot/sglang-speculative-decoding-optimization/blob/main/experiments/speculative-decoding/article.md) · [All files](https://github.com/ravi-kot/sglang-speculative-decoding-optimization/tree/main/experiments/speculative-decoding) |
| Optimizing Qwen3.6 speculative decoding | 335 to 975 tok/s with cost-aware verification and exact block-24 fixes | [Open blog](https://ravi-kot.github.io/sglang-speculative-decoding-optimization/experiments/qwen36-speculative-decoding/) | [article.md](https://github.com/ravi-kot/sglang-speculative-decoding-optimization/blob/main/experiments/qwen36-speculative-decoding/article.md) · [All files](https://github.com/ravi-kot/sglang-speculative-decoding-optimization/tree/main/experiments/qwen36-speculative-decoding) |

Each experiment is self-contained under `experiments/` so future studies can be added without mixing their code or results.

