"""Isolated SM120 tuning of the split-KV verify attention (two Triton launches: prefix split-KV + combine).

  SGLANG_SPLITKV_COMBINE_WARPS=<w> python profiling/splitkv_bench.py [--ctx 1024 4096] [--lext 4 6 7]

Real Qwen3.5-2B full-attention shape (8 query heads, 2 KV heads, head_dim 256), batch 1, prefix KV
scattered through a large pool (random kv_indices, as in serving). Times one call (both kernels) by
CUDA-graph replay for each (n_splits, BLOCK_N, num_warps) and checks the output against the current
default configuration. Run once per combine-warps value (the env var is read at import).
"""

import argparse
import itertools
import os
import statistics
import sys

import torch

sys.path.insert(0, "<sglang-overlay>")
from sglang.kernels.ops.attention import verify_splitkv as vs  # noqa: E402

H_Q, H_KV, D = 8, 2, 256


def run(cfg, ctx, lext, inputs, trials=200):
    n_splits, block_n, warps = cfg
    vk = vs.VerifySplitKV(1, H_Q, H_KV, D, D, lext, device="cuda", n_splits=n_splits, block_n=block_n,
                          num_warps=warps)
    q, ke, ve, kb, vb, qo, kvp, kvi = inputs
    out = torch.empty(lext, H_Q, D, device="cuda", dtype=torch.bfloat16)
    f = lambda: vk(q, ke, ve, kb, vb, qo, kvp, kvi, D ** -0.5, o_out=out)
    for _ in range(3):
        f()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    e0, e1, ts = torch.cuda.Event(True), torch.cuda.Event(True), []
    for _ in range(trials):
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts), out.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--lext", type=int, nargs="+", default=[4, 6, 7])
    a = ap.parse_args()
    torch.manual_seed(0)
    pool = 16384
    kb = torch.randn(pool, H_KV, D, device="cuda", dtype=torch.bfloat16)
    vb = torch.randn(pool, H_KV, D, device="cuda", dtype=torch.bfloat16)
    print(f"combine_warps={vs.COMBINE_NUM_WARPS}")
    for ctx, lext in itertools.product(a.ctx, a.lext):
        q = torch.randn(lext, H_Q, D, device="cuda", dtype=torch.bfloat16)
        ke = torch.randn(lext, H_KV, D, device="cuda", dtype=torch.bfloat16)
        ve = torch.randn(lext, H_KV, D, device="cuda", dtype=torch.bfloat16)
        qo = torch.tensor([0, lext], device="cuda", dtype=torch.int32)
        kvp = torch.tensor([0, ctx], device="cuda", dtype=torch.int32)
        kvi = torch.randperm(pool, device="cuda")[:ctx].to(torch.int32)
        inputs = (q, ke, ve, kb, vb, qo, kvp, kvi)
        default = (vs.choose_n_splits(ctx), *vs.block_config(D))
        t0, ref = run(default, ctx, lext, inputs)
        best = (t0, default)
        rows = []
        for cfg in itertools.product((4, 8, 16, 32), (16, 32, 64), (2, 4, 8)):
            try:
                t, o = run(cfg, ctx, lext, inputs)
            except Exception as e:  # e.g. shared-memory limits for a tile
                rows.append((cfg, None, type(e).__name__))
                continue
            err = (o.float() - ref.float()).abs().max().item()
            rows.append((cfg, t, f"{err:.1e}"))
            if t < best[0] and err < 2e-2:
                best = (t, cfg)
        print(f"ctx={ctx} lext={lext}: default {default} {t0:.1f} us | best {best[1]} {best[0]:.1f} us "
              f"({t0 / best[0]:.2f}x)")
        for cfg, t, err in sorted((r for r in rows if r[1]), key=lambda r: r[1])[:3]:
            print(f"    {cfg}  {t:.1f} us  maxerr {err}")


if __name__ == "__main__":
    main()
