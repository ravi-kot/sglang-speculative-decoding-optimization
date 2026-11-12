"""Session 9 attention gate (no server): our planner's current KV split (stable_chunk, padded grid as in the best-stack trace) vs the
one-wave split (Q37_ATTN_ONEWAVE: tiles x kv heads <= 188 SMs), same FlashInfer fa2 paged kernel, real shapes:
  target verify: 8 query rows, 16 q heads, 2 kv heads, dim 256, FP8 KV; contexts 18K / 48K / 101K / 165K (one per MLCommons bin)
  drafter:       8 query rows, 32 q heads, 8 kv heads, dim 128, FP8 KV, 8K window
L2 is evicted before every call (256 MB fill, its own time subtracted). Prints per case and the round-weighted saving
(10 target layers, bins weighted equally; 6 drafter layers). Last line S9_ATTN saves_us_per_target_layer <x> round_ms <y>."""
import os, sys
import torch, flashinfer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q37_nvtx import stable_chunk, SLACK

NSM = 188
work = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
ev = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def timed(fn, n=30):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(3): ev.fill_(1); fn()
    torch.cuda.synchronize(); s.record()
    for _ in range(n): ev.fill_(1); fn()
    e.record(); torch.cuda.synchronize(); t_all = s.elapsed_time(e)
    s.record()
    for _ in range(n): ev.fill_(1)
    e.record(); torch.cuda.synchronize(); t_ev = s.elapsed_time(e)
    return (t_all - t_ev) / n * 1e3


def case(L, hq, hkv, d, padded):
    k = (torch.randn(L, 1, hkv, d, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    v = (torch.randn(L, 1, hkv, d, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    q = torch.randn(8, hq, d, device="cuda", dtype=torch.bfloat16) * 0.1
    args = (torch.tensor([0, 8], device="cuda", dtype=torch.int32), torch.tensor([0, L], device="cuda", dtype=torch.int32),
            torch.arange(L, device="cuda", dtype=torch.int32), torch.tensor([1], device="cuda", dtype=torch.int32), hq, hkv, d, 1)
    kw = dict(causal=True, sm_scale=d ** -0.5, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn, o_data_type=torch.bfloat16)
    w0 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(work, "NHD", backend="fa2"); w0.plan(*args, **kw)
    cta_q = int(w0._plan_info[3]) if hasattr(w0, "_plan_info") else 64
    q_tiles = (8 * hq // hkv + cta_q - 1) // cta_q
    c_cur = stable_chunk([L], SLACK, q_tiles, padded) or 128
    c_one = stable_chunk([L], SLACK, q_tiles, padded, target=NSM // hkv) or c_cur
    res = {}
    for tag, c in (("cur", c_cur), ("one", c_one)):
        w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(work, "NHD", backend="fa2")
        w.plan(*args, fixed_split_size=c, **kw)
        o = w.run(q, (k, v)); torch.cuda.synchronize()
        res[tag] = (c, (L + c - 1) // c, timed(lambda: w.run(q, (k, v))), o.float())
    d_out = (res["cur"][3] - res["one"][3]).abs().max().item()
    print(f"ATTN L={L} hq={hq} hkv={hkv} d={d}: current chunk {res['cur'][0]} ({res['cur'][1]} chunks x {hkv} heads) {res['cur'][2]:.1f} us | "
          f"one-wave chunk {res['one'][0]} ({res['one'][1]} chunks) {res['one'][2]:.1f} us | saving {res['cur'][2] - res['one'][2]:+.1f} us | max|diff| {d_out:.2e}", flush=True)
    del k, v
    return res["cur"][2] - res["one"][2]




def interleave(w):
    """Reorder the plan's work list from (q tile, kv chunk) to (kv chunk, q tile), so the q tiles that read the same KV chunk run
    side by side (second read can hit L2). The kernel takes (request, q tile, kv chunk) from the index arrays; triples stay intact."""
    pi = [int(x) for x in w._plan_info]; n = pi[0]; buf = w._int_workspace_buffer
    v = lambda off: buf[off: off + 4 * n].view(torch.int32)
    rq, qt, kt = v(pi[4]), v(pi[5]), v(pi[6])
    perm = torch.argsort(rq.long() * (1 << 40) + kt.long() * (1 << 20) + qt.long(), stable=True)
    for t in (rq, qt, kt):
        t.copy_(t[perm].clone())
    return pi


def case_b(B, L, hq=16, hkv=2, d=256, padded=188, perm=False, timing=True):
    k = (torch.randn(L, 1, hkv, d, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    v = (torch.randn(L, 1, hkv, d, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    q = torch.randn(B, hq, d, device="cuda", dtype=torch.bfloat16) * 0.1
    args = (torch.tensor([0, B], device="cuda", dtype=torch.int32), torch.tensor([0, L], device="cuda", dtype=torch.int32),
            torch.arange(L, device="cuda", dtype=torch.int32), torch.tensor([1], device="cuda", dtype=torch.int32), hq, hkv, d, 1)
    kw = dict(causal=True, sm_scale=d ** -0.5, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn, o_data_type=torch.bfloat16)
    w0 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(work, "NHD", backend="fa2"); w0.plan(*args, **kw)
    cta_q = int(w0._plan_info[3]); q_tiles = (B * hq // hkv + cta_q - 1) // cta_q
    c = stable_chunk([L], SLACK, q_tiles, padded) or 128     # the server's split (Q37_SYNCFREE planner)
    w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(work, "NHD", backend="fa2"); w.plan(*args, fixed_split_size=c, **kw)
    o0 = w.run(q, (k, v)).float(); t0 = timed(lambda: w.run(q, (k, v))) if timing else 0
    pi = interleave(w) if perm else [int(x) for x in w._plan_info]
    o1 = w.run(q, (k, v)).float(); t1 = timed(lambda: w.run(q, (k, v))) if timing else 0
    kvb = 2 * L * hkv * d   # fp8 K + V bytes, one read
    print(f"ATTNB B={B} L={L}: cta_q {cta_q} q_tiles {q_tiles} chunk {c} ({(L + c - 1) // c} chunks) work {pi[0]} split {pi[-1]} | "
          f"tile-major {t0:.1f} us{'' if not perm else f' | chunk-interleaved {t1:.1f} us ({t1 - t0:+.1f})'} | one-read KV {kvb / 1e6:.1f} MB "
          f"= {kvb / max(t0, 1e-9) / 1e3:.0f} GB/s at tile-major | bitwise equal {bool((o0 == o1).all())}", flush=True)
    del k, v
    return t0, t1



def case_a128(L, B=16, hq=16, hkv=2, d=256, cfgs=None):
    """Our 128-row verify kernel (attn128.py) vs FlashInfer fa2 at the server's split, same cold-L2 timing. Correctness against an
    fp32 torch reference on rows < Lr (Lr = AVL's kept rows); FlashInfer's own error is the bar. Returns {(Lr, cfg): us} for correct runs."""
    from attn128 import verify_attn
    k = (torch.randn(L, 1, hkv, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(L, 1, hkv, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    q = torch.randn(B, hq, d, device="cuda", dtype=torch.bfloat16)
    ip, ix, lp = (torch.tensor([0, L], device="cuda", dtype=torch.int32), torch.arange(L, device="cuda", dtype=torch.int32),
                  torch.tensor([1], device="cuda", dtype=torch.int32))
    args = (torch.tensor([0, B], device="cuda", dtype=torch.int32), ip, ix, lp, hq, hkv, d, 1)
    kw = dict(causal=True, sm_scale=d ** -0.5, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn, o_data_type=torch.bfloat16)
    w0 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(work, "NHD", backend="fa2"); w0.plan(*args, **kw)
    q_tiles = (B * hq // hkv + int(w0._plan_info[3]) - 1) // int(w0._plan_info[3])
    w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(work, "NHD", backend="fa2")
    w.plan(*args, fixed_split_size=stable_chunk([L], SLACK, q_tiles, 188) or 128, **kw)
    o_fi = w.run(q, (k, v)).float(); t_fi = timed(lambda: w.run(q, (k, v)))
    # fp32 reference, one KV head at a time (scores [8*B, L])
    ref = torch.empty(B, hq, d, device="cuda")
    pos = torch.arange(L, device="cuda"); lim = L - B + torch.arange(B, device="cuda")
    for h in range(hkv):
        kk, vv = k[:, 0, h].float(), v[:, 0, h].float()
        qq = q[:, h * (hq // hkv):(h + 1) * (hq // hkv)].float()                      # [B, G, d]
        sc = torch.einsum("bgd,ld->bgl", qq, kk) * d ** -0.5
        sc = sc.masked_fill(pos[None, None, :] > lim[:, None, None], float("-inf"))
        ref[:, h * (hq // hkv):(h + 1) * (hq // hkv)] = torch.einsum("bgl,ld->bgd", sc.softmax(-1), vv)
    e_fi = (o_fi - ref).abs().max().item()
    best = {}
    for Lr in (B, 12, 8, 4):
        Lt = torch.tensor([Lr], device="cuda", dtype=torch.int32)
        for c in cfgs:
            cfg = dict(zip(("BLOCK_N", "num_warps", "num_stages", "MIN_CHUNK"), c))
            try:
                o = verify_attn(q, k, v, ip, ix, lp, d ** -0.5, L=Lt, cfg=cfg).float(); torch.cuda.synchronize()
            except Exception as ex:   # resource limits (shared memory / registers) for this config
                print(f"A128 L={L} Lr={Lr} cfg {c} FAILED {type(ex).__name__}: {str(ex)[:120]}", flush=True); continue
            e = (o[:Lr] - ref[:Lr]).abs().max().item(); z = o[Lr:].abs().max().item() if Lr < B else 0.0
            t = timed(lambda: verify_attn(q, k, v, ip, ix, lp, d ** -0.5, L=Lt, cfg=cfg))
            ok = e <= 2 * e_fi + 2e-3 and z == 0.0
            print(f"A128 L={L} Lr={Lr} cfg {c}: ours {t:.1f} us | fa2 B{B} {t_fi:.1f} us | err ours {e:.2e} fa2 {e_fi:.2e} | rows>=Lr {z:.0e} "
                  f"| {'OK' if ok else 'BAD'}", flush=True)
            if ok:
                best[(Lr, c)] = t
    del k, v
    return t_fi, best



def case_lskip(L, B, hq=16, hkv=2, d=256):
    """Q37_ATTN_LSKIP gate: the full split plan vs its "first t q tiles" variants (lskip_images), same FlashInfer kernel."""
    from q37_nvtx import lskip_images
    k = (torch.randn(L, 1, hkv, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(L, 1, hkv, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    q = torch.randn(B, hq, d, device="cuda", dtype=torch.bfloat16)
    args = (torch.tensor([0, B], device="cuda", dtype=torch.int32), torch.tensor([0, L], device="cuda", dtype=torch.int32),
            torch.arange(L, device="cuda", dtype=torch.int32), torch.tensor([1], device="cuda", dtype=torch.int32), hq, hkv, d, 1)
    kw = dict(causal=True, sm_scale=d ** -0.5, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn, o_data_type=torch.bfloat16)
    w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(work, "NHD", backend="fa2"); w.plan(*args, **kw)
    tiles = (B * hq // hkv + int(w._plan_info[3]) - 1) // int(w._plan_info[3])
    w.plan(*args, fixed_split_size=stable_chunk([L], 6 * B, tiles, 188) or 128, **kw)
    run = lambda: w.run(q, (k, v))
    oA = run().float(); tA = timed(run)
    img = lskip_images(w, [L], B, hq // hkv, slack=6 * B)
    res = {}
    if img is None:
        print(f"LSKIP L={L} B={B}: plan not split", flush=True); return res
    dev = img.cuda(); ws = w._int_workspace_buffer; N = img.shape[1]
    for t in range(1, tiles):
        ws[:N].copy_(dev[t - 1]); r = t * int(w._plan_info[3]) * hkv // hq
        o = run().float(); e = (o[:r] - oA[:r]).abs().max().item(); z = o[r:].abs().max().item()
        tt = timed(run); ok = e <= 2e-3 and z == 0.0
        print(f"LSKIP L={L} B={B} tiles {t}/{tiles} (tokens < {r}): {tt:.1f} us vs full {tA:.1f} us | max|diff| rows<{r} {e:.1e} | rows>={r} {z:.0e} | {'OK' if ok else 'BAD'}", flush=True)
        res[t] = (tA - tt) if ok else None
    ws[:N].copy_(dev[tiles - 1]); same = bool((run().float() == oA).all())
    print(f"LSKIP L={L} B={B} restore full plan bitwise {same}", flush=True)
    if not same:
        res = {t: None for t in res}
    del k, v
    return res


if len(sys.argv) > 1 and sys.argv[1] == "lskip":   # python s9_attn_gate.py lskip [B]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    tot = {}
    for L in (18000, 48000, 101000, 165000):
        for t, x in case_lskip(L, B).items():
            tot.setdefault(t, []).append(x)
    ok = bool(tot) and all(x is not None for v in tot.values() for x in v)
    print(f"S9_LSKIP B={B} {'PASS' if ok else 'FAIL'} " + " ".join(f"t{t} saves_us {sum(v) / len(v):.1f}" for t, v in tot.items() if None not in v), flush=True)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "a128":   # python s9_attn_gate.py a128 [B]: Q37_ATTN128 gate + config sweep
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    cfgs = [(32, 8, 1, 128), (32, 16, 1, 128), (16, 8, 2, 128), (16, 16, 2, 128), (16, 16, 3, 128), (32, 8, 1, 256)]   # fit 99 KB smem
    tot = {}
    for L in (18000, 48000, 101000, 165000):
        t_fi, best = case_a128(L, B, cfgs=cfgs)
        for (Lr, c), t in best.items():
            tot.setdefault(c, {}).setdefault(Lr, []).append(t_fi - t)
    # one config for the server: the best mean saving at full length (Lr = B) that is correct at every length and every Lr
    full = {c: sum(v[B]) / len(v[B]) for c, v in tot.items() if len(v.get(B, [])) == 4 and all(len(x) == 4 for x in v.values())}
    if full:
        c = max(full, key=full.get)
        print(f"S9_A128 cfg {','.join(map(str, c))} saves_us_per_layer {full[c]:.1f} at Lr {B}; "
              + " ".join(f"Lr{Lr} {sum(x) / len(x):.1f}" for Lr, x in sorted(tot[c].items())), flush=True)
    else:
        print("S9_A128 none correct at every length", flush=True)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "blocks":   # block-size study: python s9_attn_gate.py blocks [B ...]
    Bs = [int(x) for x in sys.argv[2:]] or [8, 16]
    for L in (18000, 101000, 165000):
        for B in Bs:
            case_b(B, L, perm=True)
elif len(sys.argv) > 1 and sys.argv[1] == "one":    # ncu: python s9_attn_gate.py one B L perm(0|1); 4 untimed calls
    B, L, p = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
    case_b(B, L, perm=p, timing=False)
else:
    tgt = [case(L, 16, 2, 256, 188) for L in (18000, 48000, 101000, 165000)]
    drf = case(8192 + 8, 32, 8, 128, 47)
    per_layer = sum(tgt) / len(tgt)
    print(f"S9_ATTN saves_us_per_target_layer {per_layer:.1f} drafter_layer {drf:.1f} round_ms {(10 * per_layer + 6 * drf) / 1e3:.3f}", flush=True)
