"""Drafter training v2: block-16 DFlash2 (warm start: Coral's qwen-swe-drafter), full 8K context, DFlash recipe.
Forward = the installed SGLang DFlash2 forward (validated in ft_train.py: 99.4% round agreement with the server when context fits),
generalized to any block size: block = [anchor, mask x (B-1)], non-causal attention over [context window, block], dynamic 2-tap conv
inside the block, final norm, candidates = top-16 of the target lm_head on the deployed 48K hot vocabulary, selector lattice walked
greedily.
Data: per training trace and mode, the union of prompt features over all turns (positions shared by the prefix property; fp8 on disk)
plus each turn's output features, output ids and target top-16 soft labels (Q37_FEAT_DUMP files, run17 g2_train).
Batching (DFlash): per turn, context K/V are computed once for the span [first anchor - W, last anchor + B) and up to --anchors blocks
attend to it through a mask (each block sees context positions [p - W, p) and its own block). Anchors are resampled every epoch.
Options: --hard-frac (post-rejection anchors), --on-weight/--off-weight (per-mode loss), per-mode best checkpoints in <out>_off/_on,
chain eval split by segment (think / tool / text).
Loss: sum_k w_k [(1 - s) CE(hard) + s CE(soft top-16)] + sel * sum_k w_k CE(selector, teacher forced), w_k = exp(-(k-1)/gamma).
Validation (held-out traces, per-turn 8K context, run17 g2_val): greedy-walk acceptance (accepted + 1) at block B and at block 8.
Usage: ft2_train.py --feat <dir> --turns train_turns.jsonl --valfeat <dir> --valturns val_turns.jsonl --out <dir> [options]"""
import argparse, glob, json, math, os, queue, random, threading, time
import torch, torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--feat", required=True); ap.add_argument("--turns", required=True)
ap.add_argument("--valfeat", required=True); ap.add_argument("--valturns", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--drafter", default=os.path.expanduser("~/lab/models/qwen-swe-drafter"))
ap.add_argument("--target", default=os.path.expanduser("~/lab/models/Qwen3.6-35B-A3B-FP8"))
ap.add_argument("--hot", default=os.path.expanduser("~/lab/profiles/q36_s4/hotv/hot_ids_48000.pt"))
ap.add_argument("--block", type=int, default=16)
ap.add_argument("--seed", type=int, default=0, help="data order / anchor sampling (continuations use a new seed)")
ap.add_argument("--window", type=int, default=8192)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=float, default=0.04)
ap.add_argument("--gamma", type=float, default=7.0)
ap.add_argument("--soft", type=float, default=0.5)
ap.add_argument("--sel", type=float, default=1.0)
ap.add_argument("--anchors", type=int, default=48, help="anchor blocks per turn per visit")
ap.add_argument("--accum", type=int, default=8, help="turns per optimizer step")
ap.add_argument("--val-max", type=int, default=4000)
ap.add_argument("--val-every", type=int, default=400)
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--chain-turns", type=int, default=40, help="turns per held-out trace and mode for the chain (server-like) evaluation")
ap.add_argument("--metrics", default="")
ap.add_argument("--teacher-kl", type=float, default=0.0, help="KL to the frozen warm-start drafter run at block --teacher-block on its draft positions (keeps early positions)")
ap.add_argument("--teacher-block", type=int, default=8)
ap.add_argument("--conf-weight", type=float, default=0.0, help="extra loss weight on positions where the target's top-1 probability >= 0.9 (learnable tokens; near-ties are noisy labels)")
ap.add_argument("--err-by-conf", action="store_true", help="eval: drafter miss rate at reached positions, binned by the target's own top-1 probability")
ap.add_argument("--two-pass", type=int, default=0, help="eval: second drafter pass with the first k drafted tokens revealed in the block (iterative unmasking); positions k+1.. re-drafted")
ap.add_argument("--src-repeat", default="", help="comma list aligned with --feat sources: how many times each source's traces enter the sampling pool")
ap.add_argument("--dump-rounds", default="", help="eval: write every chain round (mode, segment, useful tokens, drafter path probabilities) as JSON lines")
ap.add_argument("--hard-frac", type=float, default=0.0, help="share of each turn's anchors replaced by post-rejection anchors: p + accepted(p) under the current model (the server's next-round anchors)")
ap.add_argument("--on-weight", type=float, default=1.0, help="loss weight of thinking-on turns"); ap.add_argument("--off-weight", type=float, default=1.0)
a = ap.parse_args()
dev = os.environ.get("FT2_DEV", "cuda"); torch.manual_seed(a.seed); random.seed(a.seed)
BLK, H, NH, NKV, HD, EPS, THETA, TOPK = a.block, 2048, 32, 8, 128, 1e-6, 1e7, 16
MASK = json.load(open(os.path.join(a.drafter, "config.json")))["dflash_config"]["mask_token_id"]
W_POS = torch.tensor([math.exp(-(k) / a.gamma) for k in range(BLK - 1)], device=dev)          # weight of draft position k+1

# ---------------------------------------------------------------- weights (fc / hidden_norm stay out: features are already projected)
fd = safe_open(os.path.join(a.drafter, "model.safetensors"), "pt")
P = {k: torch.nn.Parameter(fd.get_tensor(k).float().to(dev)) for k in fd.keys() if not k.startswith(("fc.", "hidden_norm."))}
idx = json.load(open(os.path.join(a.target, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(a.target, idx["model.language_model.embed_tokens.weight"]), "pt") as f:
    EMB = f.get_tensor("model.language_model.embed_tokens.weight").to(dev)
hot = torch.load(a.hot).to(dev).long()
P0 = {k: v.detach().to(torch.bfloat16).clone() for k, v in P.items()} if a.teacher_kl > 0 else None
with safe_open(os.path.join(a.target, idx["lm_head.weight"]), "pt") as f:
    HEAD = f.get_tensor("lm_head.weight").to(dev)[hot].contiguous()
hot_pos = torch.full((EMB.shape[0],), -1, dtype=torch.long, device=dev); hot_pos[hot] = torch.arange(len(hot), device=dev)
try:
    from transformers import AutoTokenizer
    _tk = AutoTokenizer.from_pretrained(a.target)
    THINK_END, TOOL_START = _tk.convert_tokens_to_ids("</think>"), _tk.convert_tokens_to_ids("<tool_call>")
except Exception as e:
    THINK_END = TOOL_START = None; print("segment ids unavailable:", e, flush=True)
inv_freq = 1.0 / (THETA ** (torch.arange(0, HD, 2, device=dev).float() / HD))


def rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)) * w


def rope(x, pos):
    fr = pos.float().unsqueeze(-1) * inv_freq
    cos, sin = fr.cos().unsqueeze(-2), fr.sin().unsqueeze(-2)
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def conv(h, base, delta):   # h [N, B, H] (block positions on dim 1), base [taps, H], delta [N, B, taps, G]
    N_, T_, _ = h.shape; G = delta.shape[-1]; gs = H // G
    blocks = h.view(N_, T_, G, gs)
    coef = base.view(1, 1, 2, G, gs) + delta.unsqueeze(-1)
    return (coef[:, :, 0] * blocks + coef[:, :, 1] * F.pad(blocks[:, :-1], (0, 0, 0, 0, 1, 0))).reshape(N_, T_, H)


def prepare(h, l, name):
    c = (h @ P[f"layers.{l}.{name}.kernel_projection.weight"].T).view(*h.shape[:-1], 2, 2, -1)
    return conv(h, P[f"layers.{l}.{name}.base_kernel"][0], c[..., 0, :, :]), c[..., 1, :, :]


def finish(h, l, name, coef):
    return conv(h, P[f"layers.{l}.{name}.base_kernel"][1], coef)


REVEAL = {}
EBC = [[0, 0] for _ in range(10)]
EBP = [[0, 0] for _ in range(64)]   # per block position: [confident (p>=0.9) reached positions, drafter misses]   # per target-top-1-probability decile: [reached positions, drafter misses]


def drafter(ctx_feat, ctx_pos, anchors, p0, P=P, BLK=BLK):
    """One turn: ctx_feat [L, H] (span features), ctx_pos [L], anchors [N] anchor ids, p0 [N] anchor positions.
    Block n sees context positions in [p0 - W, p0) and its own B positions. Returns hidden [N, B, H]."""
    N_ = anchors.shape[0]
    ids = torch.full((N_, BLK), MASK, device=dev, dtype=torch.long); ids[:, 0] = anchors
    if REVEAL.get("tok") is not None:   # iterative unmasking: block positions 1..k carry known (drafted) tokens
        kk = REVEAL["tok"].shape[1]; ids[:, 1: 1 + kk] = REVEAL["tok"]
    x = EMB[ids].float()
    bpos = p0.unsqueeze(1) + torch.arange(BLK, device=dev)                                    # [N, B]
    L = ctx_feat.shape[0]
    cmask = (ctx_pos[None, :] < p0[:, None]) & (ctx_pos[None, :] >= (p0 - a.window)[:, None])   # [N, L]
    own = torch.block_diag(*[torch.ones(BLK, BLK, dtype=torch.bool, device=dev)] * N_)         # [N*B, N*B]
    m = torch.cat([cmask.repeat_interleave(BLK, 0), own], 1)[None, None]                       # [1, 1, N*B, L + N*B]
    residual = None
    for l in range(6):
        pre = f"layers.{l}."
        if residual is None:
            residual = x; h = rms(x, P[pre + "input_layernorm.weight"])
        else:
            residual = x + residual; h = rms(residual, P[pre + "input_layernorm.weight"])
        h, ak = prepare(h, l, "attention_conv")
        q = rope(rms((h @ P[pre + "self_attn.q_proj.weight"].T).view(N_, BLK, NH, HD), P[pre + "self_attn.q_norm.weight"]), bpos)
        k = rope(rms((h @ P[pre + "self_attn.k_proj.weight"].T).view(N_, BLK, NKV, HD), P[pre + "self_attn.k_norm.weight"]), bpos)
        v = (h @ P[pre + "self_attn.v_proj.weight"].T).view(N_, BLK, NKV, HD)
        ck = rope(rms((ctx_feat @ P[pre + "self_attn.k_proj.weight"].T).view(L, NKV, HD), P[pre + "self_attn.k_norm.weight"]), ctx_pos)
        cv = (ctx_feat @ P[pre + "self_attn.v_proj.weight"].T).view(L, NKV, HD)
        K = torch.cat([ck, k.reshape(N_ * BLK, NKV, HD)], 0).transpose(0, 1).repeat_interleave(NH // NKV, 0)   # [NH, L+NB, HD]
        V = torch.cat([cv, v.reshape(N_ * BLK, NKV, HD)], 0).transpose(0, 1).repeat_interleave(NH // NKV, 0)
        Q = q.reshape(N_ * BLK, NH, HD).transpose(0, 1)                                                       # [NH, NB, HD]
        o = F.scaled_dot_product_attention(Q[None], K[None], V[None], attn_mask=m, scale=HD ** -0.5)[0]
        o = o.transpose(0, 1).reshape(N_, BLK, NH * HD) @ P[pre + "self_attn.o_proj.weight"].T
        o = finish(o, l, "attention_conv", ak)
        residual = o + residual; h = rms(residual, P[pre + "post_attention_layernorm.weight"])
        h, mk = prepare(h, l, "mlp_conv")
        h = (F.silu(h @ P[pre + "mlp.gate_proj.weight"].T) * (h @ P[pre + "mlp.up_proj.weight"].T)) @ P[pre + "mlp.down_proj.weight"].T
        x = finish(h, l, "mlp_conv", mk)
    return rms(x + residual, P["norm.weight"])


def lattice(hid, cand, pred_ids):
    hp = hid @ P["candidate_selector.hidden_projection.weight"].T
    A_ = P["candidate_selector.predecessor_codebook"][pred_ids]
    B_ = P["candidate_selector.successor_codebook"][cand]
    return torch.einsum("bepr,becr->bepc", A_ * hp.unsqueeze(2), B_)


# ---------------------------------------------------------------- data
def sessions(turns_file):
    s = {}
    for line in open(turns_file):
        t = json.loads(line); s[t["turn_id"]] = t["session"]
    return s


BADF = set()


def load_trace(files):
    """files: one trace, one mode. Returns prompt features (union, by position) and per-turn output data."""
    pf, pt, turns, bad = {}, {}, [], 0
    for f in sorted(files):
        r = torch.load(f)
        if r["ctx_pos"] is None or r["ver_pos"] is None:
            continue
        cf = r["ctx_feat"]
        for i, p in enumerate(r["ctx_pos"].tolist()):
            if p not in pf:
                pf[p] = i; pt[p] = (f, int(r["ctx_ids"][i]))
            elif pt[p][1] != int(r["ctx_ids"][i]):
                bad += 1
        turns.append((f, r))
    return pf, pt, turns, bad


def turn_arrays(r, prompt_feat_of):
    """Contiguous position-indexed tensors for one turn (prompt positions from the trace union, outputs from the turn)."""
    Pl, out = r["prompt_len"], r["output_ids"]
    feats, toks, soft = {}, {}, {}
    for rd in range(r["ver_pos"].shape[0]):
        c = int(r["ver_commit"][rd])
        if not (0 < c <= r["ver_pos"].shape[1]):
            continue
        for j in range(c):
            p = int(r["ver_pos"][rd, j])
            if Pl <= p < Pl + len(out) and p not in feats:
                feats[p] = r["ver_feat"][rd, j].to(torch.bfloat16)
                if r["top_lp"] is not None:
                    soft[p + 1] = (r["top_lp"][rd, j], r["top_id"][rd, j])
    for i, o in enumerate(out):
        toks[Pl + i] = int(o)
    return Pl, feats, toks, soft


class Trace:
    """One trace, both modes. The prefix cache means each turn file holds only the prompt tokens the server recomputed, so the
    trace's prompt features are a union: files in increasing prompt length, later writes win (a later prompt is the true history of
    every earlier position except the previous turns' generation-prompt tails, which later turns recompute). At sample time the
    turn's own recomputed span is laid over the union, so its own tail is exact. Off and on share the prompt prefix (off runs first,
    on hits its cache), hence one union for both modes."""
    def __init__(self, files):
        rs = []
        for f, mode in files:
            try:
                r = torch.load(f)
            except Exception as ex:   # a truncated file (e.g. written while the disk was full) must not stall the producer thread
                BADF.add(f); print(f"skip unreadable feature file {f}: {type(ex).__name__}", flush=True); continue
            if r["ctx_pos"] is None or r["ver_pos"] is None or len(r["output_ids"]) < BLK + 1:
                continue
            r["_mode"] = mode; rs.append(r)
        rs.sort(key=lambda r: (r["prompt_len"], r["_mode"]))
        self.turns = rs
        maxp = max([int(r["ctx_pos"].max()) for r in rs], default=0)
        dt = rs[0]["ctx_feat"].dtype if rs else torch.bfloat16
        self.dense = torch.zeros(maxp + 1, H, dtype=dt); self.valid = torch.zeros(maxp + 1, dtype=torch.bool)
        self.ids = torch.zeros(maxp + 1, dtype=torch.long)
        for r in rs:
            pos = r["ctx_pos"]
            self.dense[pos] = r["ctx_feat"]; self.ids[pos] = r["ctx_ids"].long(); self.valid[pos] = True

    def sample(self, r, n_anchors, all_anchors=False, stride=1, anchors=None):
        if "_ta" not in r:
            r["_ta"] = turn_arrays(r, None)
        Pl, feats, toks, soft = r["_ta"]
        if not feats:
            return None
        e = Pl
        while e in feats:
            e += 1
        has_prev = lambda p: (p - 1 >= Pl) or (p - 1 < self.valid.shape[0] and bool(self.valid[p - 1]))
        if anchors is not None:   # explicit (chain evaluation): must have contiguous features up to p - 1
            if not all(p - 1 < e and has_prev(p) for p in anchors):
                return None
        else:
            anchors = [p for p in range(Pl, Pl + len(r["output_ids"]) - (BLK - 1)) if p - 1 < e and has_prev(p)]
            if not anchors:
                return None
            anchors = anchors[::stride] if all_anchors else random.sample(anchors, min(n_anchors, len(anchors)))
        lo = max(0, min(anchors) - a.window); hi = max(anchors)
        pp = torch.arange(lo, min(Pl, hi, self.valid.shape[0])); pp = pp[self.valid[pp]] if len(pp) else pp
        pf = self.dense[pp].to(torch.bfloat16)
        own = r["ctx_pos"]; sel_ = (own >= lo) & (own < Pl)
        if sel_.any() and len(pp):   # the turn's own recomputed prompt span (its exact generation-prompt tail)
            op_, of_ = own[sel_], r["ctx_feat"][sel_].to(torch.bfloat16)
            ix = torch.searchsorted(pp, op_); okx = (ix < len(pp)) & (pp[ix.clamp(max=len(pp) - 1)] == op_)
            pf[ix[okx]] = of_[okx]
        op = [p for p in range(max(lo, Pl), hi) if p in feats]
        cf = torch.cat([pf, torch.stack([feats[p] for p in op]) if op else torch.zeros(0, H, dtype=torch.bfloat16)])
        pos = torch.cat([pp, torch.tensor(op, dtype=torch.long)])
        own_ids = dict(zip(own.tolist(), r["ctx_ids"].tolist())) if Pl - 1 in set(own.tolist()) else {}
        anc = [toks[p] if p >= Pl else int(own_ids.get(p, self.ids[p])) for p in anchors]
        y = [[toks.get(p + k, -1) for k in range(1, BLK)] for p in anchors]
        sv = torch.full((len(anchors), BLK - 1, TOPK), -1e4); si = torch.zeros(len(anchors), BLK - 1, TOPK, dtype=torch.long)
        sm = torch.zeros(len(anchors), BLK - 1, dtype=torch.bool)
        for n, p in enumerate(anchors):
            for k in range(1, BLK):
                if (p + k) in soft:
                    v, i = soft[p + k]; sv[n, k - 1] = v.float(); si[n, k - 1] = i.long(); sm[n, k - 1] = True
        cov = float(len(pp)) / max(1, min(Pl, hi) - lo)   # fraction of the prompt window that has features
        return [cf.pin_memory(), pos.pin_memory(), torch.tensor(anc).pin_memory(), torch.tensor(anchors).pin_memory(),
                torch.tensor(y).pin_memory(), sv.pin_memory(), si.pin_memory(), sm.pin_memory()], cov


def to_dev(b):
    return [z.to(dev, non_blocking=True) for z in b]


def loss_on(batch):
    cf, cp, anc, p0, y, sv, si, sm = batch
    hid = drafter(cf.float(), cp, anc, p0)[:, 1:]                                       # [N, B-1, H]
    logits = hid @ HEAD.float().T
    yh = hot_pos[y.clamp(min=0)]; ok = (y >= 0) & (yh >= 0)
    w = W_POS.expand_as(y.float())
    if a.conf_weight:
        w = w * (1 + a.conf_weight * ((sv[..., 0].float().exp() >= 0.9) & sm).float())
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yh.clamp(min=0).reshape(-1), reduction="none").view_as(w)
    hard = (ce * w * ok).sum() / (w * ok).sum().clamp(min=1)
    sh = hot_pos[si]; sok = (sh >= 0) & sm.unsqueeze(-1)
    tgt = torch.where(sok, sv, torch.full_like(sv, -1e4)).softmax(-1)
    lp = logits.log_softmax(-1).gather(-1, sh.clamp(min=0))
    sce = -(tgt * lp * sok).sum(-1); smask = sm & sok.any(-1)
    soft = (sce * w * smask).sum() / (w * smask).sum().clamp(min=1)
    _, cidx = logits.detach().topk(TOPK, dim=-1); cand = hot[cidx]
    pred = torch.cat([anc[:, None], y[:, :-1].clamp(min=0)], 1).unsqueeze(-1)
    score = logits.gather(-1, cidx).unsqueeze(2) + lattice(hid, cand, pred)
    hit = cand == y.unsqueeze(-1); has = hit.any(-1) & (y >= 0)
    sce2 = F.cross_entropy(score[:, :, 0].reshape(-1, TOPK), hit.float().argmax(-1).reshape(-1), reduction="none").view_as(w)
    sel = (sce2 * w * has).sum() / (w * has).sum().clamp(min=1)
    tkl = torch.zeros((), device=logits.device)
    if P0 is not None:   # frozen warm-start drafter at its own block size: keep its first draft positions
        tb = a.teacher_block - 1
        with torch.no_grad():
            tl = (drafter(cf.float(), cp, anc, p0, P=P0, BLK=a.teacher_block)[:, 1:] @ HEAD.float().T).float().log_softmax(-1)
        sl = logits[:, :tb].float().log_softmax(-1)
        kl = (tl.exp() * (tl - sl)).sum(-1)                                                   # [N, tb]
        tkl = (kl * w[:, :tb]).sum() / w[:, :tb].sum()
    return (1 - a.soft) * hard + a.soft * soft + a.sel * sel + a.teacher_kl * tkl, (hard.item(), soft.item(), sel.item(), tkl.item())


@torch.no_grad()
def walk_accept(batch):
    cf, cp, anc, p0, y, *_ = batch
    with torch.autocast(dev, dtype=torch.bfloat16):
        hid = drafter(cf.float(), cp, anc, p0)[:, 1:]
        logits = (hid @ HEAD.float().T).float()
    hid = hid.float()
    _, cidx = logits.topk(TOPK, dim=-1); cand = hot[cidx]; unary = logits.gather(-1, cidx)
    prev = torch.cat([anc[:, None, None].expand(-1, 1, TOPK), cand[:, :-1]], 1)
    score = unary.unsqueeze(2) + lattice(hid, cand, prev)
    s0 = score[:, 0, 0]; i_ = s0.argmax(-1); path = [i_]; conf = [s0.float().softmax(-1).max(-1).values]
    for e in range(1, BLK - 1):
        se = score[:, e].gather(1, i_[:, None, None].expand(-1, 1, TOPK))[:, 0]; i_ = se.argmax(-1); path.append(i_)
        conf.append(se.float().softmax(-1).max(-1).values)
    walk_accept.conf = torch.stack(conf, 1).tolist()   # probability of the chosen candidate per draft position (selector lattice)
    toks = cand.gather(2, torch.stack(path, 1).unsqueeze(-1))[..., 0]
    kp = a.two_pass
    if kp and not REVEAL:   # pass 2: reveal the first kp drafted tokens, walk the lattice from position kp with that fixed predecessor
        REVEAL["tok"] = toks[:, :kp]
        try:
            with torch.autocast(dev, dtype=torch.bfloat16):
                hid2 = drafter(cf.float(), cp, anc, p0)[:, 1:]
                lg2 = (hid2 @ HEAD.float().T).float()
            hid2 = hid2.float()
        finally:
            REVEAL.clear()
        _, ci2 = lg2.topk(TOPK, dim=-1); c2 = hot[ci2]; un2 = lg2.gather(-1, ci2)
        prev2 = torch.cat([anc[:, None, None].expand(-1, 1, TOPK), c2[:, :-1]], 1)
        prev2[:, kp] = toks[:, kp - 1: kp].expand(-1, TOPK)            # fixed predecessor (pass-1 token kp)
        s2 = un2.unsqueeze(2) + lattice(hid2, c2, prev2)
        j_ = s2[:, kp, 0].argmax(-1); p2 = [j_]
        for e in range(kp + 1, BLK - 1):
            j_ = s2[:, e].gather(1, j_[:, None, None].expand(-1, 1, TOPK))[:, 0].argmax(-1); p2.append(j_)
        t2 = c2[:, kp:].gather(2, torch.stack(p2, 1).unsqueeze(-1))[..., 0]
        toks = torch.cat([toks[:, :kp], t2], 1)
    hitp = (toks == y) & (y >= 0)
    if a.err_by_conf and len(batch) > 5:   # reached = all earlier drafts correct; target top-1 prob from the stored soft labels
        sv_, sm_ = batch[5], batch[7]
        reached = torch.cat([torch.ones_like(hitp[:, :1]), hitp[:, :-1].int().cumprod(1).bool()], 1) & (y >= 0) & sm_
        pt = sv_[..., 0].float().exp()
        conf_ = reached & (pt >= 0.9)
        for k_ in range(hitp.shape[1]):   # confident-token misses by block position
            EBP[k_][0] += int(conf_[:, k_].sum()); EBP[k_][1] += int((conf_[:, k_] & ~hitp[:, k_]).sum())
        for b_ in range(10):
            sel_ = reached & (pt >= b_ / 10) & (pt < (b_ + 1) / 10 + (b_ == 9))
            EBC[b_][0] += int(sel_.sum()); EBC[b_][1] += int((sel_ & ~hitp).sum())
    accB = hitp.int().cumprod(1).sum(1) + 1
    acc8 = hitp[:, :7].int().cumprod(1).sum(1) + 1
    return accB.tolist(), acc8.tolist()


def build_traces(featdir, turns_file):
    ss = sessions(turns_file); by = {}
    for f in glob.glob(os.path.join(featdir, "*.pt")):
        b = os.path.basename(f)[:-3]
        if "_" not in b:
            continue
        tid, mode = b.rsplit("_", 1)
        if tid in ss:
            by.setdefault(ss[tid], []).append((f, mode))
    return by


val_by = build_traces(a.valfeat, a.valturns)
train_by = {}; src_rep = {}
reps = [int(x) for x in a.src_repeat.split(",")] if a.src_repeat else []
for si, (fd_, tf_) in enumerate(zip(a.feat.split(","), a.turns.split(","))):   # several (feature dir, turns file) sources, paired in order
    b_ = build_traces(fd_, tf_); train_by.update(b_)
    src_rep.update({k: (reps[si] if si < len(reps) else 1) for k in b_})
print(f"train trace-modes {len(train_by)} ({sum(len(v) for v in train_by.values())} turn files), val trace-modes {len(val_by)} ({sum(len(v) for v in val_by.values())})", flush=True)


VAL_CACHE = {}
DUMP = open(a.dump_rounds, "w") if a.dump_rounds else None


def validate():
    """Equal anchor budget per held-out trace and mode (stride-4 anchors over shuffled turns); results per mode too."""
    accB, acc8, per, covs = [], [], {}, []
    cap = max(1, a.val_max // max(2 * len(val_by), 1))
    for key, fs in sorted(val_by.items()):
        tr = VAL_CACHE.get(key) or VAL_CACHE.setdefault(key, Trace(fs))
        for mode in ("off", "on"):
            kb, k8 = [], []
            rs = [r for r in tr.turns if r["_mode"] == mode]; random.Random(5).shuffle(rs)
            for r in rs:
                out = tr.sample(r, 0, all_anchors=True, stride=max(4, len(r["output_ids"]) // 4))   # ~4 anchors per turn, many turns
                if out is None:
                    continue
                b, cov = out; covs.append(cov)
                for s0 in range(0, b[2].shape[0], 32):
                    sub = [b[0], b[1]] + [z[s0: s0 + 32] for z in b[2:]]
                    x, y8 = walk_accept(to_dev(sub)); kb += x; k8 += y8
                if len(kb) >= cap:
                    break
            accB += kb[:cap]; acc8 += k8[:cap]
            per.setdefault(mode, []).extend(kb[:cap])
    if a.chain_turns:   # server-like rounds: anchor at the prompt end, advance by the accepted length (anchors follow rejections)
        for key, fs in sorted(val_by.items()):
            tr = VAL_CACHE[key]
            for mode in ("off", "on"):
                rs = [r for r in tr.turns if r["_mode"] == mode]; random.Random(7).shuffle(rs)
                for r in rs[: a.chain_turns]:
                    Pl, n = r["prompt_len"], len(r["output_ids"]); p = Pl
                    ol = [int(t) for t in r["output_ids"]]
                    te = ol.index(THINK_END) if mode == "on" and THINK_END in ol else (n if mode == "on" else 0)
                    tc = ol.index(TOOL_START) if TOOL_START in ol else n
                    while p <= Pl + n - 2:
                        out = tr.sample(r, 0, anchors=[p])
                        if out is None:
                            break
                        x, _ = walk_accept(to_dev(out[0])); acc_ = min(x[0], Pl + n - p)
                        per.setdefault("chain_" + mode, []).append(acc_)
                        if DUMP:
                            DUMP.write(json.dumps({"mode": mode, "turn": os.path.basename(str(r.get("_file", ""))), "p": p - Pl, "n": n, "acc": acc_,
                                                   "conf": [round(c, 4) for c in walk_accept.conf[0]]}) + chr(10))
                        seg = "think" if p - Pl < te else ("tool" if p - Pl >= tc else "text")
                        per.setdefault(f"seg_{mode}_{seg}", []).append(acc_); p += acc_
    pm = {m: round(sum(v) / max(len(v), 1), 3) for m, v in per.items()}
    pm.update({"n_" + m: len(v) for m, v in per.items() if m.startswith("seg_")})
    pm["ctx_cov"] = round(sum(covs) / max(len(covs), 1), 4)
    for m, v in [(m, v) for m, v in per.items() if not m.startswith(("chain", "seg_"))]:   # reach: P(useful tokens > k), k = 1..B-1
        pm["reach_" + m] = [round(sum(x > k for x in v) / max(len(v), 1), 3) for k in range(1, BLK)]
    if "chain_off" in pm and "chain_on" in pm:   # model selection: server-like tokens per round, thinking off and on weighted equally
        pm["uniform"] = round(sum(accB) / max(len(accB), 1), 3)
        return (pm["chain_off"] + pm["chain_on"]) / 2, sum(acc8) / max(len(acc8), 1), len(accB), pm
    return sum(accB) / max(len(accB), 1), sum(acc8) / max(len(acc8), 1), len(accB), pm


mfh = open(a.metrics, "a") if a.metrics else None
def log(d):
    if mfh:
        mfh.write(json.dumps(d) + chr(10)); mfh.flush()


t0 = time.time()
vB, v8, nv, pm = validate()
if a.err_by_conf:
    tot = sum(m for _, m in EBC)
    print("ERRPOS confident-token miss rate by draft position: " + " ".join(f"{k + 1}:{m / max(n, 1):.3f}" for k, (n, m) in enumerate(EBP[: BLK - 1])), flush=True)
    print("ERRCONF " + " | ".join(f"p{b / 10:.1f}: n {n} miss {m / max(n, 1):.3f} share {m / max(tot, 1):.3f}" for b, (n, m) in enumerate(EBC)), flush=True)
print(f"VAL before: block {BLK} {vB:.3f} useful tokens/round (by mode {pm}), first 8 positions {v8:.3f} ({nv} anchors, {time.time() - t0:.0f}s)", flush=True)
log({"event": "val", "step": 0, "valB": vB, "val8": v8, "n": nv, "by_mode": pm, "args": vars(a)})
if a.eval_only:
    raise SystemExit

keys = [k for k in train_by for _ in range(src_rep.get(k, 1))]
n_turn_visits = int(a.epochs * sum(len(train_by[k]) for k in keys))
print(f"sampling pool: {len(keys)} trace entries ({len(train_by)} traces), {sum(len(train_by[k]) for k in keys)} weighted turn files", flush=True)
steps = max(1, n_turn_visits // a.accum)
params = list(P.values())
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
warm = max(10, int(a.warmup * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
print(f"steps {steps} (accum {a.accum} turns, {a.anchors} anchors/turn), lr {a.lr}, warmup {warm}, gamma {a.gamma}", flush=True)


def producer(qb):   # groups of 4 traces loaded together, their turns shuffled; anchors resampled every visit
    visits = 0
    while visits < n_turn_visits:
        random.shuffle(keys)
        for g in range(0, len(keys), 4):
            trs = [Trace(train_by[k]) for k in keys[g: g + 4]]
            items = [(tr, r) for tr in trs for r in tr.turns]; random.shuffle(items)
            for tr, r in items:
                out = tr.sample(r, a.anchors)
                if out is not None:
                    qb.put((out[0], r["_mode"], tr, r)); visits += 1
                    if visits >= n_turn_visits:
                        break
            del trs, items
            if visits >= n_turn_visits:
                break
    qb.put(None)


qb = queue.Queue(maxsize=16)
threading.Thread(target=producer, args=(qb,), daemon=True).start()
def save(d, why):
    os.makedirs(d, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in P.items()}
    sd["fc.weight"] = fd.get_tensor("fc.weight"); sd["hidden_norm.weight"] = fd.get_tensor("hidden_norm.weight")
    save_file(sd, os.path.join(d, "model.safetensors"))
    cfg = json.load(open(os.path.join(a.drafter, "config.json"))); cfg["dflash_config"]["block_size"] = BLK
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    print(f"SAVED step {step} -> {d} ({why})", flush=True)


def hard_batch(b, tr, r):   # post-rejection anchors: where the server's next round starts after the current model drafts at p
    nh = max(1, int(round(b[2].shape[0] * a.hard_frac)))
    x, _ = walk_accept([b[0], b[1]] + [z[:nh] for z in b[2:]])
    lim = r["prompt_len"] + len(r["output_ids"]) - (BLK - 1)
    nxt = sorted({int(p) + int(k) for p, k in zip(b[3][:nh].tolist(), x) if int(p) + int(k) < lim})
    out = tr.sample(r, 0, anchors=nxt) if nxt else None
    if out is None and nxt:   # some anchors lack contiguous features: keep those that work
        nxt = [p for p in nxt if tr.sample(r, 0, anchors=[p]) is not None]
        out = tr.sample(r, 0, anchors=nxt) if nxt else None
    return (to_dev(out[0]) if out else None), nh


bmode = {"off": None, "on": None}
best = vB; step = 0; acc_n = 0; ts = time.time(); parts_acc = [0.0, 0.0, 0.0, 0.0]; loss_acc = 0.0
while True:
    item = qb.get()
    if item is None:
        break
    b, mode, tr, r = item; b = to_dev(b); mw = a.on_weight if mode == "on" else a.off_weight
    bh = None
    if a.hard_frac > 0:
        bh, nh = hard_batch(b, tr, r)
        if bh is not None:
            b = [b[0], b[1]] + [z[nh:] for z in b[2:]]
    n_u = b[2].shape[0]; n_h = bh[2].shape[0] if bh is not None else 0
    for bb, nb in ((b, n_u), (bh, n_h)):
        if bb is None or nb == 0:
            continue
        with torch.autocast(dev, dtype=torch.bfloat16):
            loss, parts = loss_on(bb)
        f = nb / (n_u + n_h)
        (mw * f * loss / a.accum).backward(); loss_acc += f * loss.item()
        parts_acc = [x + f * y for x, y in zip(parts_acc, parts)]
    acc_n += 1
    if acc_n < a.accum:
        continue
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    log({"step": step, "loss": loss_acc / acc_n, "hard": parts_acc[0] / acc_n, "soft": parts_acc[1] / acc_n, "sel": parts_acc[2] / acc_n, "tkl": parts_acc[3] / acc_n,
         "grad_norm": float(gn), "lr": sched.get_last_lr()[0], "step_s": round(time.time() - ts, 3), "queue": qb.qsize(),
         "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "t": round(time.time() - t0, 1)})
    if step % 25 == 0:
        print(f"step {step}/{steps} loss {loss_acc / acc_n:.4f} (hard {parts_acc[0] / acc_n:.3f} soft {parts_acc[1] / acc_n:.3f} sel {parts_acc[2] / acc_n:.3f} tkl {parts_acc[3] / acc_n:.3f}) "
              f"lr {sched.get_last_lr()[0]:.2e} {time.time() - ts:.2f}s/step {time.time() - t0:.0f}s", flush=True)
    acc_n = 0; loss_acc = 0.0; parts_acc = [0.0, 0.0, 0.0, 0.0]; ts = time.time(); step += 1
    if step % a.val_every == 0 or step == steps:
        vB, v8, nv, pm = validate()
        print(f"VAL step {step}: block {BLK} {vB:.3f} (by mode {pm}), first 8 positions {v8:.3f} (best {best:.3f})", flush=True)
        log({"event": "val", "step": step, "valB": vB, "val8": v8, "n": nv, "by_mode": pm, "t": round(time.time() - t0, 1)})
        if vB > best:
            best = vB; save(a.out, f"val {vB:.3f}")
        for m in ("off", "on"):   # separate best checkpoints per thinking mode (server-like chain metric)
            v = pm.get("chain_" + m)
            if v is not None and (bmode[m] is None or v > bmode[m]):
                bmode[m] = v; save(f"{a.out}_{m}", f"chain_{m} {v:.3f}")
print(f"FT2_DONE best block-{BLK} val {best:.3f} {time.time() - t0:.0f}s", flush=True)
