"""Generate the article's SVG figures from the evidence in data/ (no hand-entered results).

  python make_charts.py

Palette checked with a CVD/contrast validator: teal #00806a (this work), blue #2a78d6 (target
verify), orange #eb6834 (draft); neutral gray marks the stock reference and idle time.
Every mark carries a <title> so values show on hover without JavaScript.
"""

import csv
import json
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).parent
DATA, ASSETS = ROOT / "data", ROOT / "assets"
INK, INK2, MUTED, GRID = "#1e2b34", "#52626b", "#7d8a90", "#e4e6e1"
TEAL, BLUE, ORANGE, GRAY, GRAY_LIGHT = "#00806a", "#2a78d6", "#eb6834", "#9aa4aa", "#d9dedf"
SLATE = "#4f5d66"  # neutral: eager bookkeeping (not a series of interest)


def r0(v):
    """Round half up for labels (Python's round() is half-to-even: 814.5 -> 814)."""
    return int(v + 0.5)
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"  # the blog's --sans
SERIF = "Georgia,'Times New Roman',serif"  # the blog's --serif (headings)


# ---------------------------------------------------------------- evidence readers
def compare_rows(name):
    """Rows of a scripts/compare.py markdown table, keyed by (workload, variant)."""
    rows = {}
    for line in (DATA / "runs" / name).read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 11 and c[0] in ("code_1k", "code_4k", "mtbench"):
            rows[(c[0], c[1])] = {"tpot": float(c[4]), "gain": c[6], "tps": float(c[7]), "e2e": float(c[8]),
                                  "accept": float(c[10])}
    return rows


def analysis(name):
    t = (DATA / "profiler" / name).read_text(encoding="utf-8")
    m = re.search(r"round p50 ([\d.]+) us = target ([\d.]+) \+ draft ([\d.]+) \+ eager ([\d.]+) "
                  r"\+ idle host-late ([\d.]+) \+ idle dep/sync ([\d.]+)", t)
    return dict(zip(["round", "target", "draft", "eager", "idle_host", "idle"], (float(x) / 1e3 for x in m.groups())))


def ledger(n):
    return json.loads((DATA / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[n - 1])


def launch_mean(run, variant, needle, regs=None):
    """Mean kernel duration (us) and the full line for `variant`'s kernel matching `needle` in one run."""
    text = (DATA / "profiler" / "launch_configs.txt").read_text(encoding="utf-8")
    block = text.split(f"## {run}")[1].split("\n## ")[0]
    for line in block.splitlines():
        if line.split()[:1] == [variant] and needle in line and (regs is None or f"regs {regs} " in line):
            return float(re.search(r"mean\s+([\d.]+) us", line).group(1)), line
    raise KeyError((run, variant, needle))


# ---------------------------------------------------------------- svg helpers
class Svg:
    def __init__(self, w, h, title, desc):
        self.w, self.h, self.parts = w, h, []
        self.head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" '
                     f'aria-labelledby="t d" font-family="{FONT}">'
                     f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
                     '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
                     'patternTransform="rotate(45)"><rect width="6" height="6" fill="#f3f4f2"/>'
                     f'<line x1="0" y1="0" x2="0" y2="6" stroke="{GRAY_LIGHT}" stroke-width="2.4"/></pattern></defs>')

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, fill=INK2, anchor="start", weight=400, family=None):
        fam = f' font-family="{family}"' if family else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                 f'font-weight="{weight}"{fam}>{escape(s)}</text>')

    def rect(self, x, y, w, h, fill, tip=None, rx=0, stroke=None, dash=None):
        st = f' stroke="{stroke}" stroke-width="1.2"' if stroke else ""
        st += f' stroke-dasharray="{dash}"' if dash else ""
        t = f"<title>{escape(tip)}</title>" if tip else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}"{st}>{t}</rect>')

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                 f'stroke-width="{width}"{d}/>')

    def legend(self, x, y, items):
        for label, fill in items:
            self.rect(x, y - 10, 11, 11, fill, rx=2)
            self.text(x + 16, y, label, 12.5, INK2)
            x += 24 + 7.1 * len(label)

    def save(self, name):
        (ASSETS / name).write_text(self.head + "".join(self.parts) + "</svg>\n", encoding="utf-8")
        print("wrote assets/" + name)


def bar_top(s, x, y, w, h, fill, tip):
    """Vertical bar with a 4px rounded data end and square baseline."""
    r = min(4, h)
    s.add(f'<path d="M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + w - r:.1f} '
          f'Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z" fill="{fill}"><title>{escape(tip)}</title></path>')


def bar_right(s, x, y, w, h, fill, tip, stroke=None, dash=None):
    r = min(4, w)
    st = (f' stroke="{stroke}" stroke-width="1.3"' if stroke else "") + (f' stroke-dasharray="{dash}"' if dash else "")
    s.add(f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h - r:.1f} '
          f'Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{fill}"{st}>'
          f'<title>{escape(tip)}</title></path>')


# ---------------------------------------------------------------- figure 1: final results
def fig_results():
    rows = compare_rows("final_vs_stock_L74_L75.txt")
    # one code configuration for both code rows (its 4K result ties the 64K config within noise)
    best = {"code_1k": "S5K4RH32QMDCF", "code_4k": "S5K4RH32QMDCF", "mtbench": "S5K4RHQMD"}
    names = {"code_1k": "Code, 1K-token prompt", "code_4k": "Code, 4K-token prompt", "mtbench": "MT-Bench"}
    s = Svg(680, 330, "Decode throughput, stock SGLang vs my best configuration",
            "Paired runs L74 and L75, same session. Bars are 1000 divided by median TPOT.")
    left, base, top, ymax = 56, 268, 52, 250
    y = lambda v: base - (base - top) * v / ymax
    for v in range(0, ymax + 1, 50):
        s.line(left, y(v), 664, y(v))
        s.text(left - 8, y(v) + 4, str(v), 12, MUTED, "end")
    s.text(16, 160, "output tok/s", 12, MUTED, "middle")
    s.parts[-1] = s.parts[-1].replace("<text ", '<text transform="rotate(-90 16 160)" ', 1)
    gw = (664 - left) / 3
    for i, w in enumerate(best):
        st, b = rows[(w, "S2")], rows[(w, best[w])]
        cx = left + gw * i + gw / 2
        for j, (r, fill, lab) in enumerate(((st, GRAY, "stock"), (b, TEAL, best[w]))):
            x = cx - 54 + j * 58
            bar_top(s, x, y(r["tps"]), 50, base - y(r["tps"]), fill,
                    f"{names[w]} · {lab}: {r['tps']:.1f} tok/s (TPOT {r['tpot']:.3f} ms, accept {r['accept']:.2f})")
            s.text(x + 25, y(r["tps"]) - 7, f"{r['tps']:.0f}", 13, INK, "middle", 600)
        gain = b["gain"].split(" ")[0]
        s.text(cx, base + 22, names[w], 13, INK, "middle", 600)
        s.text(cx, base + 40, f"paired gain {gain}%", 12.5, INK2, "middle")
    s.legend(left, 26, [("Stock SGLang 0.5.21 (NEXTN, 2 steps, Triton)", GRAY),
                        ("My configuration (code: 32K + context rows; chat: 64K)", TEAL)])
    s.save("fig-results.svg")


# ---------------------------------------------------------------- figure 2: round timeline
def fig_timeline():
    lanes = []
    for name, label, ref in (("stock_s2_code1k", "Stock, 2 draft steps", 2.475),
                             ("final_code1k", "Final, 5 draft steps", 3.156)):
        d = json.loads((DATA / "profiler" / f"timeline_{name}.json").read_text())
        lanes.append((label, d, ref))
    s = Svg(680, 250, "One speculative round on the GPU, before and after (code 1K, median round)",
            "Kernel intervals from CUPTI traces L3 and L55, classified by profiling/analyze.py.")
    left, right, ms_max = 150, 586, 21.0
    x = lambda us: left + (right - left) * us / (ms_max * 1000)
    colors = {"target": BLUE, "draft": ORANGE, "eager": SLATE}
    for v in range(0, 22, 3):
        s.line(x(v * 1000), 46, x(v * 1000), 196)
        s.text(x(v * 1000), 212, f"{v}", 12, MUTED, "middle")
    s.text((left + right) / 2, 230, "time within one round (ms)", 12, MUTED, "middle")
    for i, (label, d, accept) in enumerate(lanes):
        y0 = 62 + i * 72
        R = d["round_us"]
        s.rect(x(0), y0, x(R) - x(0), 30, "url(#hatch)", f"{label}: idle (GPU waiting) inside a {R / 1e3:.2f} ms round")
        tot = {"target": 0.0, "draft": 0.0, "eager": 0.0}
        for kind in ("eager", "draft", "target"):
            segs = sorted((g["start"], g["end"]) for g in d["segments"] if g["kind"] == kind)
            merged = []
            for a, b in segs:
                if merged and a <= merged[-1][1] + 2:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            for a, b in merged:
                tot[kind] += b - a
                s.rect(x(a), y0, max(x(b) - x(a), 0.8), 30, colors[kind])
        tip = (f"{label}: round {R / 1e3:.2f} ms = verify {tot['target'] / 1e3:.2f} + draft {tot['draft'] / 1e3:.2f} "
               f"+ eager {tot['eager'] / 1e3:.2f} ms + idle; {accept:.2f} tokens committed per round")
        s.rect(x(0), y0, x(R) - x(0), 30, "transparent", tip)
        s.text(left - 10, y0 + 13, label, 13, INK, "end", 600)
        s.text(left - 10, y0 + 29, f"{R / 1e3:.1f} ms · {accept:.2f} tok/round", 12, INK2, "end")
        s.text(x(R) + 6, y0 + 20, f"{R / accept / 1e3:.2f} ms/token", 12, INK2)
    s.legend(left, 26, [("target verify", BLUE), ("draft", ORANGE), ("eager bookkeeping", SLATE),
                        ("GPU idle", "url(#hatch)")])
    s.save("fig-round-timeline.svg")


# ---------------------------------------------------------------- figure 3: draft head
def fig_draft_head():
    h6 = json.loads((DATA / "profiler" / "head_bench_20261024-215917-head.json").read_text())["variants"]
    h8 = json.loads((DATA / "profiler" / "head_bench_20261024-220702-head.json").read_text())["variants"]
    # INT4 (uint4b8) Marlin kernel in the final draft graph; scalar-type id 1125899906842624 in the template
    us32, _ = launch_mean("20261026-054627-local", "S5K4RH32QMD", "Marlin<__nv_bfloat16, 1125899")
    rows = [("BF16, full vocab (cuBLAS GEMV)", h6["bf16"]["us_M1"], h6["bf16"]["weight_mib"], GRAY, "isolated"),
            ("FP8 W8A16, full vocab (Marlin)", h6["w8a16_marlin"]["us_M1"], h6["w8a16_marlin"]["weight_mib"], TEAL,
             "isolated"),
            ("INT4 W4A16 g32, full vocab", h8["w4a16_marlin_g32"]["us_M1"], h8["w4a16_marlin_g32"]["weight_mib"], TEAL,
             "isolated"),
            ("INT4 g32, 32K hot rows", us32, 36.0, TEAL, "in-model trace")]  # +36 MiB: server log, ledger [L41]
    s = Svg(680, 262, "Draft output-head time per call", "Batch 1 (M=1). Isolated: profiling/head_bench.py, "
            "ledger L6 and L8. 32K rows: kernel time in the final draft graph, trace L55.")
    left, right, top = 236, 600, 30
    xmax = 3000
    x = lambda v: left + (right - left) * v / xmax
    for v in range(0, 3001, 1000):
        s.line(x(v), top - 6, x(v), top + 4 * 44 - 8)
        s.text(x(v), top + 4 * 44 + 8, f"{v:,}", 12, MUTED, "middle")
    s.text((left + right) / 2, top + 4 * 44 + 26, "microseconds per call (lower is better)", 12, MUTED, "middle")
    for i, (lab, us, mib, fill, src) in enumerate(rows):
        y = top + i * 44
        bar_right(s, left, y, x(us) - left, 24, fill, f"{lab}: {us:,.1f} us per call, {mib:,.1f} MiB of weights ({src})")
        s.text(left - 10, y + 11, lab, 13, INK, "end", 600)
        s.text(left - 10, y + 26, f"{r0(mib):,} MiB of weights · {src}", 11.5, MUTED, "end")
        s.text(x(us) + 8, y + 17, f"{r0(us):,} µs", 13, INK, "start", 600)
    s.save("fig-draft-head.svg")


# ---------------------------------------------------------------- figure 5: split-KV SM fill
def fig_splitkv():
    # stock verify attention (128-register _fwd_kernel; the 255-register one is the draft extend)
    t_fwd, l_fwd = launch_mean("20261025-185811-local", "S2", "_fwd_kernel ", regs=128)
    t_s1, l_s1 = launch_mean("20261025-190409-local", "S2K", "_verify_prefix_stage1")
    t_s2, _ = launch_mean("20261025-190409-local", "S2K", "_verify_combine_stage2")
    ctas = lambda line: eval(re.search(r"grid (\[[^\]]+\])", line).group(1))
    g1, g2 = ctas(l_fwd), ctas(l_s1)
    n1, n2 = g1[0] * g1[1] * g1[2], g2[0] * g2[1] * g2[2]
    s = Svg(680, 232, "Verify attention at 4K context: CTAs launched vs the 36 SMs",
            "Launch configurations and mean kernel times from CUPTI traces L19 (stock) and L20 (split-KV).")
    left, cell, gap = 16, 14.0, 3.6
    for i, (lab, sub, n, fill, tip) in enumerate((
            ("Stock: Triton _fwd_kernel", f"grid {tuple(g1)} = {n1} CTAs · {t_fwd:.1f} µs per call", n1, BLUE,
             f"stock verify attention: {n1} CTAs, at most {min(n1, 36)} of 36 SMs busy; {t_fwd:.1f} us per call"),
            ("Split-KV: stage 1 + merge", f"grid {tuple(g2)} = {n2} CTAs · {t_s1:.1f} + {t_s2:.1f} µs per call", n2,
             TEAL, f"split-KV stage 1: {n2} CTAs cover all 36 SMs; {t_s1:.1f} us + merge {t_s2:.1f} us per call"))):
        y0 = 24 + i * 88
        s.text(left, y0, lab, 13.5, INK, "start", 600)
        s.text(left + 36 * (cell + gap) - gap, y0, f"{min(n, 36)} of 36 SMs with work", 12.5, INK2, "end")
        s.text(left, y0 + 19, sub, 12, INK2)
        for k in range(36):
            busy = k < n
            s.rect(left + k * (cell + gap), y0 + 30, cell, 24, fill if busy else "#ffffff", tip, rx=2,
                   stroke=None if busy else GRAY_LIGHT)
    s.text(left, 214, "One square = one SM. Filled = at least one CTA of this kernel can run there.", 12, MUTED)
    s.save("fig-splitkv-sms.svg")


# ---------------------------------------------------------------- figure 6: depth
def fig_depth():
    rows = compare_rows("depth_sweep_L35_L36.txt")
    steps = {2: "S28K4", 3: "S38K4R", 4: "S4K4R", 5: "S5K4R"}
    series = [("code_1k", "Code 1K", TEAL), ("mtbench", "MT-Bench", BLUE)]
    s = Svg(680, 292, "Draft depth: accepted tokens per round vs throughput",
            "Paired runs L35 and L36 (INT4 draft head, split-KV verify, ReplaySSM at 3 to 5 steps).")
    panels = [("accepted tokens per round", "accept", 1.8, 3.4, 0.4, "{:.1f}"),
              ("output tok/s (1000 / median TPOT)", "tps", 110, 180, 10, "{:.0f}")]
    for p, (title, key, lo, hi, stepv, fmt) in enumerate(panels):
        x0, x1, top, base = 48 + p * 338, 258 + p * 338, 58, 238
        X = lambda st: x0 + (x1 - x0) * (st - 2) / 3
        Y = lambda v: base - (base - top) * (v - lo) / (hi - lo)
        s.text(x0, 42, title, 13, INK, "start", 600)
        v = lo
        while v <= hi + 1e-9:
            s.line(x0, Y(v), x1, Y(v))
            s.text(x0 - 8, Y(v) + 4, fmt.format(v), 11.5, MUTED, "end")
            v += stepv
        for st in steps:
            s.text(X(st), base + 18, str(st), 12, MUTED, "middle")
        s.text((x0 + x1) / 2, base + 36, "draft steps", 12, MUTED, "middle")
        for w, lab, col in series:
            pts = [(X(st), Y(rows[(w, v_)][key]), rows[(w, v_)][key], st) for st, v_ in steps.items()]
            s.add(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b, *_ in pts)}" fill="none" '
                  f'stroke="{col}" stroke-width="2"/>')
            for a, b, val, st in pts:
                s.add(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4.5" fill="{col}" stroke="#ffffff" stroke-width="2">'
                      f'<title>{lab}, {st} steps: {fmt.format(val) if key == "tps" else f"{val:.2f}"}</title></circle>')
            a, b, val, _ = pts[-1]
            s.text(a + 9, b + 4, lab, 12, INK2)
    s.legend(52, 18, [(lab, col) for _, lab, col in series])
    s.save("fig-depth.svg")


# ---------------------------------------------------------------- figure 7: ceiling
def fig_ceiling():
    a = analysis("analysis_20261026-054627-local_S5K4RH32QMD.txt")
    accept = ledger(55)["metrics"]["S5K4RH32QMD|code_1k"]["accept_len_mean"]
    floor = 10.22  # 3.763 GB of BF16 verify weights / 368 GB/s, ledger L38 (profiling/gemm_bench.py)
    rows = [("Measured round (profiler)", 1000 * accept / a["round"], a["round"], accept, "measured"),
            ("Without idle + eager", 1000 * accept / (a["round"] - a["idle"] - a["eager"]),
             a["round"] - a["idle"] - a["eager"], accept, "theoretical"),
            ("Verify only (draft free)", 1000 * accept / a["target"], a["target"], accept, "theoretical"),
            ("Verify at bandwidth floor", 1000 * accept / floor, floor, accept, "idealized"),
            ("Floor, 4.0 tokens per round", 1000 * 4.0 / floor, floor, 4.0, "hypothetical")]
    s = Svg(680, 286, "Throughput ceilings for the code 1K configuration",
            "R = 1000 x committed tokens per round / round time. Inputs: trace L55, gemm_bench floor L38.")
    left, right, top, xmax = 220, 590, 34, 400
    x = lambda v: left + (right - left) * v / xmax
    for v in range(0, xmax + 1, 100):
        s.line(x(v), top - 8, x(v), top + len(rows) * 42 - 10)
        s.text(x(v), top + len(rows) * 42 + 6, str(v), 12, MUTED, "middle")
    s.text((left + right) / 2, top + len(rows) * 42 + 24, "output tok/s", 12, MUTED, "middle")
    for i, (lab, r, ms, acc, kind) in enumerate(rows):
        y = top + i * 42
        tip = f"{lab}: 1000 x {acc:.2f} / {ms:.2f} ms = {r:.0f} tok/s ({kind})"
        if kind == "measured":
            bar_right(s, left, y, x(r) - left, 24, TEAL, tip)
        else:
            bar_right(s, left, y, x(r) - left, 24, "url(#hatch)", tip, stroke=GRAY, dash="4 3")
        s.text(left - 10, y + 11, lab, 13, INK, "end", 600)
        s.text(left - 10, y + 26, f"{acc:.2f} tok / {ms:.2f} ms · {kind}", 11.5, MUTED, "end")
        s.text(x(r) + 8, y + 17, f"{r0(r)}", 13, INK, "start", 600)
    s.save("fig-ceiling.svg")


# ---------------------------------------------------------------- figure: experiment setup (structure, no data)
def fig_setup():
    s = Svg(680, 296, "Experiment setup: prompts, the speculative round inside the SGLang server, and the outputs",
            "Fixed prompts enter a local SGLang server in WSL2 on the RTX 5070. Each round drafts tokens, verifies "
            "them with the BF16 target, commits the accepted prefix and schedules the next round. Benchmark and "
            "profiler outputs feed the results. Badges mark the four findings.")
    s.add(f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
          f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')

    def arrow(x1, y1, x2, y2):
        s.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.4" marker-end="url(#arr)"/>')

    # one muted hue per finding (tint fill, mid stroke, dark badge with white text)
    HUE = {1: ("#fdeee7", "#e07a4f", "#b9521f"), 2: ("#e7eff9", "#5b8fd0", "#24599e"),
           3: ("#efecf6", "#8c7bb8", "#5f4f91"), 4: ("#fcf7dc", "#d6bd4f", "#806a12")}

    def badge(cx, y, n):
        s.rect(cx - 33, y, 66, 17, HUE[n][2], f"Finding {n} from the first trace", rx=8.5)
        s.text(cx, y + 12.5, f"Finding {n}", 10.5, "#ffffff", "middle")

    def box(x, y, w, h, title, n=None):
        """Title only, centred; finding blocks get their hue and badge, the rest stay neutral."""
        fill, stroke = HUE[n][:2] if n else ("#ffffff", GRAY)
        s.rect(x, y, w, h, fill, rx=8, stroke=stroke)
        cx = x + w / 2
        s.text(cx, y + (26 if n else h / 2 + 5), title, 13.5, INK, "middle", 400, SERIF)
        if n:
            badge(cx, y + 36, n)

    box(14, 114, 128, 52, "Fixed prompts")
    arrow(142, 140, 165, 140)

    s.rect(166, 12, 342, 272, "#fbfbfa", rx=10, stroke=GRAY, dash="5 4")
    s.text(180, 36, "SGLang server", 13.5, INK, "start", 400, SERIF)
    s.text(494, 36, "local · WSL2 · RTX 5070", 11, MUTED, "end")
    box(180, 52, 148, 64, "Draft (MTP)", 1)
    box(352, 52, 148, 64, "Target verify", 2)
    box(352, 152, 148, 64, "Accept and commit")
    box(180, 152, 148, 64, "Scheduler (CPU)", 4)
    arrow(328, 84, 351, 84)
    arrow(426, 116, 426, 151)
    arrow(352, 184, 329, 184)
    arrow(254, 152, 254, 117)
    s.text(340, 131, "one speculative", 11, MUTED, "middle")
    s.text(340, 144, "round", 11, MUTED, "middle")
    s.rect(180, 228, 320, 44, HUE[3][0], rx=8, stroke=HUE[3][1])
    s.text(192, 246, "GPU memory (8 GB)", 13.5, INK, "start", 400, SERIF)
    s.text(192, 262, "weights  ·  KV cache  ·  recurrent state", 11, INK2)
    badge(452, 241.5, 3)

    arrow(508, 66, 525, 66)
    box(526, 40, 140, 52, "Benchmark")
    arrow(596, 92, 596, 119)
    box(526, 120, 140, 52, "Results")
    arrow(508, 218, 525, 218)
    box(526, 192, 140, 52, "Profiler trace")
    s.save("fig-setup.svg")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    for f in (fig_setup, fig_results, fig_timeline, fig_draft_head, fig_splitkv, fig_depth, fig_ceiling):
        f()
