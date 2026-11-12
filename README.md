# SGLang Inference Experiments

Reproducible GPU inference experiments built around SGLang, with technical write-ups, source changes, profiler evidence, benchmark data, and reproduction scripts.

## Experiments

| Experiment | Focus | Article | Evidence and code |
| --- | --- | --- | --- |
| Small-M RMSNorm | One-row-per-block mapping for fused-add Gemma RMSNorm | [Read the article](experiments/rmsnorm/) | [Experiment files](experiments/rmsnorm/README.md) |
| Speculative decoding | Improving single-request speculative decoding on an RTX 5070 Laptop GPU | [Read the article](experiments/speculative-decoding/) | [Experiment files](experiments/speculative-decoding/README.md) |

Each experiment is self-contained under `experiments/` so future studies can be added without mixing their code or results.

