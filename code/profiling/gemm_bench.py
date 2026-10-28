"""Target-verify GEMM roofline check: real Qwen3.5 text-body linear weights at the verify row counts.

  python profiling/gemm_bench.py configs/local.yaml [--M 1 3 4 5 6] [--blas cublas,cublaslt]

Loads every 2-D text-body linear weight (plus the tied embedding used as the output head) from the
checkpoint, groups by (N, K), and times y = x @ W.T for each group at batch M as one CUDA-graph
replay over all real weights of that shape (distinct tensors, so no L2 reuse across layers). Reports
per-group time, achieved GB/s and the gap to the DRAM floor (bytes / --bw), summed per verify pass.
The floor uses the measured streaming bandwidth (head_bench GEMV/GEMM runs: ~368 GB/s).
"""

import argparse
import collections
import json
import statistics
import struct
from pathlib import Path

import torch
import yaml


def text_linear_weights(model_dir):
    out = []
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with open(f, "rb") as h:
            hdr = json.loads(h.read(struct.unpack("<Q", h.read(8))[0]))
        for k, v in hdr.items():
            if k == "__metadata__" or len(v["shape"]) != 2 or "visual" in k or "mtp" in k:
                continue
            if k.endswith(".weight") and (("proj" in k) or "embed_tokens" in k or "lm_head" in k):
                out.append((f, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--M", type=int, nargs="+", default=[1, 3, 4, 5, 6])
    ap.add_argument("--blas", default="cublas,cublaslt")
    ap.add_argument("--bw", type=float, default=368e9)
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    from safetensors import safe_open

    # Merge exactly as SGLang's runtime does (one GEMM per merged projection).
    MERGE = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gate_up", "up_proj": "gate_up",
             "in_proj_qkvz": "in_proj", "in_proj_ba": "in_proj", "in_proj_qkv": "in_proj", "in_proj_z": "in_proj",
             "in_proj_b": "in_proj", "in_proj_a": "in_proj"}
    merged = collections.defaultdict(list)
    for f, k in text_linear_weights(cfg["model"]):
        mod, name = k.rsplit(".", 2)[0], k.rsplit(".", 2)[1]
        merged[(mod, MERGE.get(name, name))].append((f, k))
    groups = collections.defaultdict(list)
    for (mod, name), parts in merged.items():
        ts = []
        for f, k in parts:
            with safe_open(str(f), "pt", device="cuda") as st:
                ts.append(st.get_tensor(k).to(torch.bfloat16))
        w = torch.cat(ts, 0) if len(ts) > 1 else ts[0]
        groups[tuple(w.shape)].append(w)
    print("merged GEMM shapes (N x K : count):", {f"{n}x{k}": len(v) for (n, k), v in groups.items()})
    tot_bytes = sum(w.numel() * 2 for ws in groups.values() for w in ws)
    print(f"{sum(len(v) for v in groups.values())} weights, {len(groups)} shapes, {tot_bytes / 1e9:.3f} GB "
          f"-> floor {tot_bytes / a.bw * 1e3:.2f} ms per verify at {a.bw / 1e9:.0f} GB/s")
    res = {}
    for blas in a.blas.split(","):
        torch.backends.cuda.preferred_blas_library(blas)
        for M in a.M:
            total = 0.0
            rows = []
            for (n, k), ws in sorted(groups.items(), key=lambda kv: -kv[0][0] * kv[0][1] * len(kv[1])):
                x = torch.randn(M, k, device="cuda", dtype=torch.bfloat16)
                outs = [torch.empty(M, n, device="cuda", dtype=torch.bfloat16) for _ in ws]
                for w, o in zip(ws, outs):  # warm up / select algorithm outside capture
                    torch.matmul(x, w.t(), out=o)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for w, o in zip(ws, outs):
                        torch.matmul(x, w.t(), out=o)
                for _ in range(3):
                    g.replay()
                e0, e1, ts = torch.cuda.Event(True), torch.cuda.Event(True), []
                for _ in range(a.trials):
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1))
                ms = statistics.median(ts)
                b = n * k * 2 * len(ws)
                total += ms
                rows.append((n, k, len(ws), ms, b / (ms * 1e-3) / 1e9, b / a.bw * 1e3))
            res[(blas, M)] = total
            print(f"[{blas} M={M}] verify GEMMs {total:.2f} ms (floor {tot_bytes / a.bw * 1e3:.2f}, "
                  f"eff {tot_bytes / a.bw * 1e3 / total:.0%})")
            if M in (1, 3) or M == max(a.M):
                for n, k, c, ms, gbs, fl in rows[:6]:
                    print(f"     {n:>6}x{k:<5} x{c:<3} {ms:6.3f} ms  {gbs:5.0f} GB/s  floor {fl:6.3f}  gap {ms - fl:+.3f}")


if __name__ == "__main__":
    main()
