# Optimizing Qwen3.6 Speculative Decoding: 975 tok/s on One RTX PRO 6000

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

## BEST3 baseline artifacts

BEST3 is the named block-8 serving baseline used between stock SGLang and the longer-block experiments. It is a serving configuration built around Coral's released DFlash2 drafter, not a separately trained checkpoint.

- [`code/q36/run17.sh`](code/q36/run17.sh): defines the BEST3 flags and the final `bb8` benchmark configuration.
- [`code/q36/q37_nvtx.py`](code/q36/q37_nvtx.py): implements the serving hooks for the hot vocabulary, narrow FP8 paths, exact small kernels, planning, and state handling used by the configuration.
- [`code/q36/s5_kernels/`](code/q36/s5_kernels/): CUDA sources used by the exact serving paths.
- [`data/bench/bench.csv`](data/bench/bench.csv): final 512-turn `bb8` throughput, confidence intervals, acceptance, step time, and context-bin results.
- [`data/correctness.csv`](data/correctness.csv): BEST3 comparisons with target-only greedy decoding.
- [`data/nsys_kernels.csv`](data/nsys_kernels.csv): the `n8` BEST3 critical-path profile.

## Run 2 and Run 6 artifacts

Run 2 and Run 6 name trained drafter checkpoints used in the study. The repository includes their implementation, recorded recipes, launch orchestration, evaluation results, and server configurations. The checkpoint weights and feature tensors are excluded because they are large generated artifacts.

| Run | Published artifacts |
| --- | --- |
| Run 2, block 16 | [`ft2_train.py`](code/q36/ft2_train.py), the training implementation; [`drafter_offline.csv`](data/drafter_offline.csv), recording block 16, teacher KL 2.0, learning rate 3e-5, selected step 1,000 and held-out results; [`run17.sh`](code/q36/run17.sh), containing the Run 2 evaluation and serving configurations |
| Run 6, block 24 | [`next_boot_s15.sh`](code/q36/next_boot_s15.sh), the exact launch recipe and offline gate; [`ft2_train.py`](code/q36/ft2_train.py), the training implementation; [`drafter_offline.csv`](data/drafter_offline.csv), recording selected step 1,244 and held-out results; [`run17.sh`](code/q36/run17.sh), containing the block-24 server and profiler configurations |

The final 512-turn measurements for both checkpoints are in [`data/bench/bench.csv`](data/bench/bench.csv). Same-boot engineering comparisons are in [`data/hold128_experiments.csv`](data/hold128_experiments.csv), and the chronological experiment record is in [`data/scoreboard16.md`](data/scoreboard16.md).

