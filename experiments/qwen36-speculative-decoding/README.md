# 975 tok/s on One GPU: Optimizing Qwen3.6 Speculative Decoding

This experiment follows a batch-size-1 SGLang speculative-decoding stack from stock MTP-4 to a block-diffusion drafter with adaptive verification and exact kernel fixes. The target remains Qwen3.6-35B-A3B-FP8.

| Configuration | Thinking off | Thinking on | Speedup over stock |
| --- | ---: | ---: | ---: |
| Stock SGLang MTP-4 | 335 tok/s | 319 tok/s | 1.00x / 1.00x |
| Block 24 + AVL + L-skip + exact fixes | **975 tok/s** | **717 tok/s** | **2.91x / 2.25x** |

Results are reweighted medians from 512 independent held-out MLCommons agentic-coding turns. Comparisons use the same turns and the same boot; intervals and per-context results are in [`data/bench/bench_tables.md`](data/bench/bench_tables.md).

## Read and reproduce

- [`index.html`](index.html): rendered article
- [`article.md`](article.md): article source
- [`data/`](data/): curated benchmark, engineering, correctness, profiler and kernel-gate tables
- [`code/q36/`](code/q36/): hooks, kernels, measurement tools, training and orchestration used in the experiment
- [`code/g7e/bole_repro2.py`](code/g7e/bole_repro2.py): Bole reproduction helper

Rebuild the figures and page from this directory:

```powershell
python make_charts.py
python build_site.py
```

Large checkpoints, feature tensors, raw Nsight reports, raw server logs and credentials are intentionally excluded.

