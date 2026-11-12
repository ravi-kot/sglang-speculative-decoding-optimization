"""Feasibility of a certified-exact quantized target LM head (offline, real hidden states).

  python profiling/lmhead_screen.py <model_dir> <hidden.pt> [--groups 32 128] [--rows 2048]

Verify reads the full BF16 tied head [248320, 2048] (1.02 GB) every round only to take an
argmax. Screen instead: approx = h . Wq (INT4 g32 Marlin, 0.25 GB) and a per-row error bound
b_i = sum_g ||(W - Wq)_ig|| * ||h_g|| (Cauchy-Schwarz per group) + kernel rounding margin.
Every token whose upper bound approx_i + b_i reaches the best lower bound max_j(approx_j - b_j)
is a candidate; the exact BF16 logits of the candidates alone give the exact argmax.
This prints the candidate-count distribution (what top-K cap a fixed-shape kernel needs) and
checks the exact argmax is always a candidate. hidden.pt: lm_head inputs dumped by the overlay
(SGLANG_LMHEAD_DUMP, variant ADUMP).
"""

import argparse
import glob
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "<sglang-overlay>")


def load_embed(model_dir):
    for f in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(f, "pt", device="cuda") as st:
            for k in st.keys():
                if k.endswith("embed_tokens.weight") and "mtp" not in k:
                    return st.get_tensor(k)
    raise SystemExit("embed_tokens.weight not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("hidden")
    ap.add_argument("--groups", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--m", type=int, default=6, help="verify rows (draft tokens) for the kernel check")
    a = ap.parse_args()
    from sgl_kernel.scalar_type import scalar_types
    from sglang.srt.speculative.draft_lm_head_quant import Int4MarlinDraftHead
    from sglang.test.test_marlin_utils import marlin_quantize

    W = load_embed(a.model_dir)  # [V, K] bf16
    V, K = W.shape
    H = torch.load(a.hidden).to(torch.bfloat16)
    H = H[H.abs().sum(1) > 0]
    if H.shape[0] > a.rows:
        H = H[torch.randperm(H.shape[0], generator=torch.Generator().manual_seed(0))[: a.rows]]
    H = H.cuda()
    T = H.shape[0]
    print(f"W {tuple(W.shape)} {W.dtype} | hidden rows {T}")

    # per-row, per-group norms of the INT4 g32 quantization error (dequantized reference)
    En = {g: torch.empty(V, K // g, device="cuda") for g in a.groups}
    Wq = torch.empty_like(W)
    for i in range(0, V, 15520):
        wq = marlin_quantize(W[i : i + 15520].t().contiguous(), scalar_types.uint4b8, 32, False)[0].t()
        Wq[i : i + 15520] = wq
        e = W[i : i + 15520].float() - wq.float()
        for g in a.groups:
            En[g][i : i + 15520] = e.view(-1, K // g, g).norm(dim=2)
    del e
    head = Int4MarlinDraftHead(W)
    torch.cuda.empty_cache()

    stats = {g: [] for g in a.groups}
    kerr, gaps = 0.0, []
    for r in range(0, T, 128):
        h = H[r : r + 128]
        exact = (h @ W.T).float()  # cuBLAS BF16, as stock verify
        argmax = exact.argmax(1)
        t2 = exact.topk(2, 1).values
        gaps.append(t2[:, 0] - t2[:, 1])
        del exact
        approx = head.quant_method.apply(head, h).float()  # deployed kernel output (BF16)
        if r == 0:
            ref = torch.cat([h.float() @ Wq[i : i + 31040].float().T for i in range(0, V, 31040)], 1)
            kerr = ((approx - ref).abs() / (ref.abs() + 1)).max().item()
            del ref
        # BF16 rounding of approx and of the stock exact logit (2 x 2^-8 rel) + fp32 accumulation
        margin = approx.abs() * 2.0 ** -6 + 5e-2
        for g in a.groups:
            hn = h.float().view(-1, K // g, g).norm(dim=2)
            ub = approx + margin + hn @ En[g].T
            lo = (2 * approx - ub).max(1, keepdim=True).values
            n = (ub >= lo).sum(1)
            inside = ub.gather(1, argmax[:, None]) >= lo
            cert = {kc: (ub.topk(kc + 1, 1).values[:, -1:] < lo).squeeze(1) for kc in (64, 256)}
            stats[g].append(torch.stack([n.float(), inside.squeeze(1).float(), cert[64].float(),
                                         cert[256].float()], 1).cpu())
            del ub
    gap = torch.cat(gaps)
    print(f"Marlin vs fp32 h.Wq: max |err|/(|ref|+1) = {kerr:.2e} (margin uses 1.6e-02 rel + 0.05)")
    print(f"exact top1-top2 gap: p10 {gap.quantile(.1):.3f} p50 {gap.median():.3f}")
    report(stats, gap, kerr, a, V, K)
    del head, Wq
    torch.cuda.empty_cache()
    kernel_check(W, H, a.m)


def kernel_check(W, H, M):
    """Deployed CertifiedLMHead: argmax == stock cuBLAS BF16 head on every row; graph-replay time."""
    from sglang.srt.layers.certified_lm_head import CertifiedLMHead

    ch = CertifiedLMHead(W)
    mism, nact = 0, []
    for r in range(0, H.shape[0] - M + 1, M):
        h = H[r : r + M]
        stock = (h @ W.T).float().argmax(1)
        out = ch(h)
        mism += (out.argmax(1) != stock).sum().item()
        nact.append((out > float("-inf")).sum(1).float())
    n = torch.cat(nact)
    print(f"CertifiedLMHead M={M}: argmax mismatches vs stock {mism}/{n.numel()} | candidates per row "
          f"p50 {n.median():.0f} max {n.max():.0f}")

    def timed(f, trials=200):
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
            ts.append(e0.elapsed_time(e1))
        return sorted(ts)[len(ts) // 2]

    h = H[:M].contiguous()
    t_stock = timed(lambda: (h @ W.T).float())
    t_cert = timed(lambda: ch(h))
    t_q = timed(lambda: ch.qhead.quant_method.apply(ch.qhead, h))
    print(f"time M={M}: stock BF16 head {t_stock:.3f} ms | certified {t_cert:.3f} ms (INT4 screen alone "
          f"{t_q:.3f} ms) | saves {t_stock - t_cert:.3f} ms/verify")


def report(stats, gap, kerr, a, V, K):
    for g in a.groups:
        s = torch.cat(stats[g])
        n = s[:, 0]
        q = [n.quantile(p).item() for p in (0.5, 0.9, 0.99)]
        print(f"bound groups {g:4d}: candidates p50 {q[0]:.0f} p90 {q[1]:.0f} p99 {q[2]:.0f} max {n.max():.0f}"
              f" | argmax inside {s[:, 1].mean():.4f} | top-64 certified {s[:, 2].mean():.4f}"
              f" top-256 certified {s[:, 3].mean():.4f} | bound table {V * K // g * 2 / 2**20:.0f} MiB fp16")


if __name__ == "__main__":
    main()
