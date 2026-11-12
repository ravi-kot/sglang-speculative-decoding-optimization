"""Q37_ATTN128 (ours): target-verify attention for block-16/24 speculative decoding at BS1 on Qwen3.6 full-attention layers
(16 q heads, 2 KV heads, head_dim 256, FP8 paged KV).
- 128-row CTA: one CTA holds all 128 packed query rows of a KV head (16 tokens x 8 GQA heads), so each KV tile is read,
  converted FP8->BF16 and staged once. FlashInfer fa2 caps head_dim 256 at 64-row tiles, so block 16 runs two CTAs per KV chunk
  (measured: 129.5 us vs 95.7 us at block 8, 101K context, L2-hit second read: the cost is per-CTA latency, not DRAM).
- Verify-length aware: with Q37_AVL the kernel reads the round's L from device memory and computes only the 32/64/128 rows
  that cover tokens < L; rows of tokens >= L are written as zeros (never used: drafts >= L are masked before acceptance, and
  causal attention keeps rows < L independent of them).
- Split-KV (flash-decoding) with a fixed grid (NSPLIT x KV heads x row tiles), lengths read from the device indptr, so one
  CUDA graph serves every context length; a second kernel merges the splits (log2-domain LSE weights).
Prior art: FlashAttention-2 online softmax (Dao 2023), Flash-Decoding split-KV (Dao, Haziza, Massa, Sizov 2023), FlashInfer GQA
row packing (Ye et al. 2025). Track S: summation order differs from FlashInfer (outputs close, not bitwise)."""
import os
import torch
import triton
import triton.language as tl

NSM = int(os.environ.get("Q37_NSM", "188"))
TILE = 128


@triton.jit
def _body(Q, K, V, KVI, Op, Lp, s, hk, row0, p0, kv_len, start, end, qk_scale,
          sq_t, sq_h, sk_t, sk_h, sv_t, sv_h, so_s, so_t, so_h,
          BM: tl.constexpr, QLEN: tl.constexpr, GRP: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
          PAGE: tl.constexpr, BLOCK_N: tl.constexpr):
    r = row0 + tl.arange(0, BM)
    rm = r < QLEN * GRP
    t = r // GRP
    hq = hk * GRP + r % GRP
    offs_d = tl.arange(0, D)
    q = tl.load(Q + t[:, None] * sq_t + hq[:, None] * sq_h + offs_d[None, :], mask=rm[:, None], other=0.0)
    lim = kv_len - QLEN + t                      # causal: last KV position row may see
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nv = offs_n < end
        if PAGE == 1:
            tok = tl.load(KVI + p0 + offs_n, mask=nv, other=0)
        else:
            tok = tl.load(KVI + p0 + offs_n // PAGE, mask=nv, other=0) * PAGE + offs_n % PAGE
        tok = tok.to(tl.int64)
        # masked lanes read token 0 (a valid slot) and are zeroed after conversion: a stale FP8 NaN never reaches the dots
        k = tl.where(nv[:, None], tl.load(K + tok[:, None] * sk_t + hk * sk_h + offs_d[None, :]).to(tl.bfloat16), 0.0)
        v = tl.where(nv[:, None], tl.load(V + tok[:, None] * sv_t + hk * sv_h + offs_d[None, :]).to(tl.bfloat16), 0.0)
        sc = tl.dot(q, tl.trans(k)) * qk_scale
        ok = nv[None, :] & (offs_n[None, :] <= lim[:, None])
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(sc - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    l_s = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_s[:, None]
    lse = tl.where(l_i == 0.0, float("-inf"), m_i + tl.log2(l_s))
    tl.store(Op + s * so_s + t[:, None] * so_t + hq[:, None] * so_h + offs_d[None, :], o, mask=rm[:, None])
    tl.store(Lp + s * (QLEN * HQ) + t * HQ + hq, lse, mask=rm)


@triton.jit
def _chunk(kv_len, NSPLIT: tl.constexpr, MIN_CHUNK: tl.constexpr, BLOCK_N: tl.constexpr):
    c = tl.maximum(tl.cdiv(kv_len, NSPLIT), MIN_CHUNK)
    return tl.cdiv(c, BLOCK_N) * BLOCK_N


@triton.jit
def _fwd(Q, K, V, KVI, KVP, LPL, LV, Op, Lp, qk_scale,
         sq_t, sq_h, sk_t, sk_h, sv_t, sv_h, so_s, so_t, so_h,
         QLEN: tl.constexpr, GRP: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
         NSPLIT: tl.constexpr, BLOCK_N: tl.constexpr, MIN_CHUNK: tl.constexpr, LAWARE: tl.constexpr, TILE_M: tl.constexpr):
    s = tl.program_id(0)
    hk = tl.program_id(1)
    row0 = tl.program_id(2) * TILE_M
    p0 = tl.load(KVP)
    kv_len = (tl.load(KVP + 1) - p0 - 1) * PAGE + tl.load(LPL)
    chunk = _chunk(kv_len, NSPLIT, MIN_CHUNK, BLOCK_N)
    start = s * chunk
    end = tl.minimum(start + chunk, kv_len)
    if LAWARE:
        need = tl.minimum(tl.load(LV), QLEN) * GRP - row0
    else:
        need = QLEN * GRP - row0
    if start < kv_len:
        if need > 64:
            _body(Q, K, V, KVI, Op, Lp, s, hk, row0, p0, kv_len, start, end, qk_scale, sq_t, sq_h, sk_t, sk_h, sv_t, sv_h,
                  so_s, so_t, so_h, TILE_M, QLEN, GRP, HQ, D, PAGE, BLOCK_N)
        elif need > 32:
            _body(Q, K, V, KVI, Op, Lp, s, hk, row0, p0, kv_len, start, end, qk_scale, sq_t, sq_h, sk_t, sk_h, sv_t, sv_h,
                  so_s, so_t, so_h, 64, QLEN, GRP, HQ, D, PAGE, BLOCK_N)
        elif need > 0:
            _body(Q, K, V, KVI, Op, Lp, s, hk, row0, p0, kv_len, start, end, qk_scale, sq_t, sq_h, sk_t, sk_h, sv_t, sv_h,
                  so_s, so_t, so_h, 32, QLEN, GRP, HQ, D, PAGE, BLOCK_N)


@triton.jit
def _merge(Op, Lp, O, KVP, LPL, LV, v_scale, so_s, so_t, so_h, sO_t, sO_h,
           QLEN: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, NSPLIT: tl.constexpr,
           BLOCK_N: tl.constexpr, MIN_CHUNK: tl.constexpr, LAWARE: tl.constexpr, NS2: tl.constexpr):
    r = tl.program_id(0)
    t = r // HQ
    hq = r % HQ
    offs_d = tl.arange(0, D)
    if LAWARE:
        lr = tl.load(LV)
    else:
        lr = QLEN
    acc = tl.zeros([D], tl.float32)
    if t < lr:
        p0 = tl.load(KVP)
        kv_len = (tl.load(KVP + 1) - p0 - 1) * PAGE + tl.load(LPL)
        ns = tl.cdiv(kv_len, _chunk(kv_len, NSPLIT, MIN_CHUNK, BLOCK_N))
        offs_s = tl.arange(0, NS2)
        lse = tl.load(Lp + offs_s * (QLEN * HQ) + r, mask=offs_s < ns, other=float("-inf"))
        mx = tl.max(lse, 0)
        den = tl.sum(tl.exp2(lse - mx), 0)
        for i in range(0, ns):
            w = tl.exp2(tl.load(Lp + i * (QLEN * HQ) + r) - mx)
            acc += w * tl.load(Op + i * so_s + t * so_t + hq * so_h + offs_d)
        acc = acc * (v_scale / den)
    tl.store(O + t * sO_t + hq * sO_h + offs_d, acc.to(O.dtype.element_ty))


def _cfg():
    c = os.environ.get("Q37_ATTN128_CFG", "32,8,1,128")
    bn, w, st, mc = (int(x) for x in c.split(","))
    return dict(BLOCK_N=bn, num_warps=w, num_stages=st, MIN_CHUNK=mc)


def verify_attn(q, k_cache, v_cache, kv_indptr, kv_indices, last_page_len, sm_scale, k_scale=1.0, v_scale=1.0, L=None,
                out=None, cfg=None, nsplit=None):
    """q [T, HQ, D] bf16 (one request, T verify tokens, causal against the cache that already holds them); k/v_cache NHD paged
    [pages, page, HKV, D] or [tokens, HKV, D] FP8; kv_indptr/kv_indices/last_page_len: FlashInfer paged layout (device).
    L: optional int32 device tensor, L[0] = verify rows kept this round (tokens >= L get zeros). Graph-capturable."""
    T, HQ, D = q.shape
    HKV = k_cache.shape[-2]
    page = k_cache.shape[1] if k_cache.dim() == 4 else 1
    kf, vf = k_cache.view(-1, HKV, D), v_cache.view(-1, HKV, D)
    c = dict(cfg or _cfg())
    bn, mc = c.pop("BLOCK_N"), c.pop("MIN_CHUNK")
    ns = nsplit or max(1, NSM // HKV)
    nt = triton.cdiv(T * (HQ // HKV), TILE)
    op = torch.empty(ns, T, HQ, D, device=q.device, dtype=torch.float32)
    lp = torch.empty(ns, T, HQ, device=q.device, dtype=torch.float32)
    if out is None:
        out = torch.empty(T, HQ, D, device=q.device, dtype=q.dtype)
    lv = L if L is not None else kv_indptr
    qk = float(sm_scale) * float(k_scale) * 1.4426950408889634
    _fwd[(ns, HKV, nt)](q, kf, vf, kv_indices, kv_indptr, last_page_len, lv, op, lp, qk,
                        q.stride(0), q.stride(1), kf.stride(0), kf.stride(1), vf.stride(0), vf.stride(1),
                        op.stride(0), op.stride(1), op.stride(2),
                        QLEN=T, GRP=HQ // HKV, HQ=HQ, D=D, PAGE=page, NSPLIT=ns, BLOCK_N=bn, MIN_CHUNK=mc,
                        LAWARE=L is not None, TILE_M=TILE, **c)
    _merge[(T * HQ,)](op, lp, out, kv_indptr, last_page_len, lv, float(v_scale), op.stride(0), op.stride(1), op.stride(2),
                      out.stride(0), out.stride(1), QLEN=T, HQ=HQ, D=D, PAGE=page, NSPLIT=ns, BLOCK_N=bn, MIN_CHUNK=mc,
                      LAWARE=L is not None, NS2=triton.next_power_of_2(ns), num_warps=4)
    return out
