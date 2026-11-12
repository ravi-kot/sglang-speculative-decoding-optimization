"""Generate article SVGs directly from the curated CSV files."""

import csv
from html import escape
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

INK = "#1e2b34"
MUTED = "#60717a"
GRID = "#e6e8e3"
TEAL = "#126c60"
TEAL_2 = "#78aaa1"
SLATE = "#7f8d93"
PAPER = "#ffffff"


def load(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, out = rows[0], []
    for row in rows[1:]:
        # The supplied benchmark CSV has one unquoted comma in the bb8 label.
        # Repair that field while preserving the source file unchanged.
        while len(row) > len(header):
            row[1] = row[1] + "," + row.pop(2)
        out.append(dict(zip(header, row)))
    return out


def shell(title, subtitle, width=1000, height=600):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
        f'<rect width="{width}" height="{height}" rx="18" fill="{PAPER}"/>',
        f'<text x="56" y="58" fill="{INK}" font-family="system-ui,sans-serif" font-size="27" font-weight="700">{escape(title)}</text>',
        f'<text x="56" y="84" fill="{MUTED}" font-family="system-ui,sans-serif" font-size="15">{escape(subtitle)}</text>',
    ]


def finish(parts, path):
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def throughput():
    rows = load(DATA / "bench" / "bench.csv")
    order = ["bstk", "bb8", "bb16", "bbest"]
    labels = ["Stock MTP-4", "Block 8 BEST3", "Block 16 + AVL", "Block 24 final"]
    values = {(r["arm"], r["mode"]): r for r in rows}
    p = shell("Independent final benchmark", "512 held-out turns · reweighted median decode throughput · 95% bootstrap CI", 1080, 620)
    left, top, bottom, chart_h = 92, 125, 520, 395
    for tick in range(0, 1101, 200):
        y = bottom - tick / 1100 * chart_h
        p += [f'<line x1="{left}" y1="{y:.1f}" x2="1020" y2="{y:.1f}" stroke="{GRID}"/>',
              f'<text x="78" y="{y+5:.1f}" text-anchor="end" fill="{MUTED}" font-family="system-ui" font-size="14">{tick}</text>']
    p.append(f'<text x="24" y="330" transform="rotate(-90 24 330)" fill="{MUTED}" font-family="system-ui" font-size="15">Decode tokens/s</text>')
    group_w, bar_w = 215, 64
    for i, arm in enumerate(order):
        gx = 135 + i * group_w
        for j, mode in enumerate(("off", "on")):
            r = values[(arm, mode)]
            val, lo, hi = map(float, (r["tok_s"], r["ci_lo"], r["ci_hi"]))
            x = gx + j * 76
            y = bottom - val / 1100 * chart_h
            h = bottom - y
            color = TEAL if mode == "off" else SLATE
            opacity = (0.56, 0.72, 0.86, 1.0)[i]
            p.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="5" fill="{color}" fill-opacity="{opacity}"/>')
            p.append(f'<text x="{x+bar_w/2}" y="{y-12:.1f}" text-anchor="middle" fill="{INK}" font-family="system-ui" font-size="15" font-weight="700">{val:.0f}</text>')
            ylo, yhi = bottom - lo / 1100 * chart_h, bottom - hi / 1100 * chart_h
            p += [f'<line x1="{x+bar_w/2}" y1="{ylo:.1f}" x2="{x+bar_w/2}" y2="{yhi:.1f}" stroke="{INK}" stroke-width="2"/>',
                  f'<line x1="{x+22}" y1="{ylo:.1f}" x2="{x+42}" y2="{ylo:.1f}" stroke="{INK}" stroke-width="2"/>',
                  f'<line x1="{x+22}" y1="{yhi:.1f}" x2="{x+42}" y2="{yhi:.1f}" stroke="{INK}" stroke-width="2"/>']
        p.append(f'<text x="{gx+70}" y="552" text-anchor="middle" fill="{INK}" font-family="system-ui" font-size="14">{escape(labels[i])}</text>')
    p += [f'<rect x="800" y="40" width="15" height="15" rx="3" fill="{TEAL}"/><text x="823" y="53" fill="{INK}" font-family="system-ui" font-size="14">Thinking off</text>',
          f'<rect x="922" y="40" width="15" height="15" rx="3" fill="{SLATE}"/><text x="945" y="53" fill="{INK}" font-family="system-ui" font-size="14">Thinking on</text>']
    finish(p, ASSETS / "final-throughput.svg")


def context_speedup():
    rows = load(DATA / "bench" / "context_speedup.csv")
    p = shell("Speedup holds across context length", "Median paired per-turn speedup over stock SGLang MTP-4 within each context bin", 1000, 560)
    left, bottom, chart_h = 90, 465, 330
    for tick in (1, 1.5, 2, 2.5, 3):
        y = bottom - (tick - 1) / 2.2 * chart_h
        p += [f'<line x1="{left}" y1="{y:.1f}" x2="950" y2="{y:.1f}" stroke="{GRID}"/>',
              f'<text x="75" y="{y+5:.1f}" text-anchor="end" fill="{MUTED}" font-family="system-ui" font-size="14">{tick:.1f}x</text>']
    for i, row in enumerate(rows):
        label = row["context"].replace("-", "–")
        gx = 145 + i * 205
        for j, mode in enumerate(("off", "on")):
            ratio = float(row[f"{mode}_vs_stock"])
            x = gx + j * 72
            y = bottom - (ratio - 1) / 2.2 * chart_h
            color = TEAL if mode == "off" else SLATE
            p.append(f'<rect x="{x}" y="{y:.1f}" width="60" height="{bottom-y:.1f}" rx="5" fill="{color}"/>')
            p.append(f'<text x="{x+30}" y="{y-10:.1f}" text-anchor="middle" fill="{INK}" font-family="system-ui" font-size="15" font-weight="700">{ratio:.2f}x</text>')
        p.append(f'<text x="{gx+66}" y="497" text-anchor="middle" fill="{INK}" font-family="system-ui" font-size="15">{escape(label)}</text>')
    p += [f'<rect x="744" y="40" width="15" height="15" rx="3" fill="{TEAL}"/><text x="767" y="53" fill="{INK}" font-family="system-ui" font-size="14">Thinking off</text>',
          f'<rect x="866" y="40" width="15" height="15" rx="3" fill="{SLATE}"/><text x="889" y="53" fill="{INK}" font-family="system-ui" font-size="14">Thinking on</text>']
    finish(p, ASSETS / "context-speedup.svg")


def drafter_blocks():
    rows = load(DATA / "drafter_offline.csv")
    selected = {}
    for r in rows:
        raw_name = r["drafter"]
        if raw_name in {"Run 1", "Run 1 (eval only)"}:
            name = "Run 1"
        elif raw_name in {"Run 2", "Run 2 (eval only)"}:
            name = "Run 2"
        else:
            name = raw_name
        block = int(r["block"])
        if name in {"Coral (eval only)", "Run 1", "Run 2"} and block in {8, 10, 12, 14, 16}:
            selected.setdefault(name, {})[block] = float(r["chain_on"])
    p = shell("A larger block needed preserved early accuracy", "Offline thinking-on tokens per round on held-out validation traces", 1000, 560)
    left, bottom, chart_h = 100, 460, 315
    for tick in (5, 5.5, 6, 6.5, 7):
        y = bottom - (tick - 5) / 2.2 * chart_h
        p += [f'<line x1="{left}" y1="{y:.1f}" x2="950" y2="{y:.1f}" stroke="{GRID}"/>',
              f'<text x="82" y="{y+5:.1f}" text-anchor="end" fill="{MUTED}" font-family="system-ui" font-size="14">{tick:.1f}</text>']
    styles = [("Run 1", TEAL_2, "Run 1"), ("Run 2", TEAL, "Run 2 + KL")]
    start = selected["Coral (eval only)"][8]
    start_x = left
    start_y = bottom - (start - 5) / 2.2 * chart_h
    p.append(f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="7" fill="{PAPER}" stroke="{SLATE}" stroke-width="3"><title>Released block-8 start: {start:.2f}</title></circle>')
    p.append(f'<text x="{start_x+12:.1f}" y="{start_y-13:.1f}" fill="{SLATE}" font-family="system-ui" font-size="13" font-weight="700">Released start</text>')
    for name, color, label in styles:
        points = []
        for block in sorted(selected.get(name, {})):
            x = left + (block - 8) / 8 * 820
            y = bottom - (selected[name][block] - 5) / 2.2 * chart_h
            points.append((x, y, block, selected[name][block]))
        if not points:
            continue
        p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y,_,_ in points)}" fill="none" stroke="{color}" stroke-width="4"/>')
        for x, y, _, val in points:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{PAPER}" stroke="{color}" stroke-width="3"><title>{label}: {val:.2f}</title></circle>')
    for block in (8, 10, 12, 14, 16):
        x = left + (block - 8) / 8 * 820
        p.append(f'<text x="{x:.1f}" y="493" text-anchor="middle" fill="{INK}" font-family="system-ui" font-size="15">{block}</text>')
    p.append(f'<text x="510" y="525" text-anchor="middle" fill="{MUTED}" font-family="system-ui" font-size="15">Maximum draft block</text>')
    for i, (_, color, label) in enumerate(styles):
        x = 708 + i * 120
        p += [f'<line x1="{x}" y1="108" x2="{x+24}" y2="108" stroke="{color}" stroke-width="4"/>',
              f'<text x="{x+31}" y="113" fill="{INK}" font-family="system-ui" font-size="13">{label}</text>']
    finish(p, ASSETS / "drafter-blocks.svg")


def avl_cost():
    rows = load(DATA / "avl_moe_cost.csv")
    grouped = {16: [], 24: []}
    for r in rows:
        grouped[int(r["graph_rows_M"])].append((int(r["verify_length_L"]), float(r["moe_ms_per_round_40_layers"])))
    p = shell("Verification length has a measurable MoE cost", "Graph-timed MoE work across 40 layers with real routing and row remapping", 1000, 560)
    left, bottom, chart_h = 100, 460, 330
    for tick in (1, 2, 3, 4, 5, 6):
        y = bottom - (tick - 1) / 5.4 * chart_h
        p += [f'<line x1="{left}" y1="{y:.1f}" x2="950" y2="{y:.1f}" stroke="{GRID}"/>',
              f'<text x="82" y="{y+5:.1f}" text-anchor="end" fill="{MUTED}" font-family="system-ui" font-size="14">{tick}</text>']
    for m, color in ((16, TEAL), (24, SLATE)):
        points = []
        for length, cost in grouped[m]:
            x = left + (length - 1) / 23 * 820
            y = bottom - (cost - 1) / 5.4 * chart_h
            points.append((x, y, length, cost))
        p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y,_,_ in points)}" fill="none" stroke="{color}" stroke-width="4"/>')
        for x, y, length, cost in points:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{PAPER}" stroke="{color}" stroke-width="3"><title>M={m}, L={length}: {cost:.3f} ms</title></circle>')
    for tick in (1, 4, 8, 12, 16, 20, 24):
        x = left + (tick - 1) / 23 * 820
        p.append(f'<text x="{x:.1f}" y="493" text-anchor="middle" fill="{INK}" font-family="system-ui" font-size="14">{tick}</text>')
    p.append(f'<text x="510" y="525" text-anchor="middle" fill="{MUTED}" font-family="system-ui" font-size="15">Rows verified, L</text>')
    p.append(f'<text x="26" y="320" transform="rotate(-90 26 320)" fill="{MUTED}" font-family="system-ui" font-size="15">MoE ms per round</text>')
    p += [f'<line x1="760" y1="48" x2="785" y2="48" stroke="{TEAL}" stroke-width="4"/><text x="793" y="53" fill="{INK}" font-family="system-ui" font-size="14">M=16 graph</text>',
          f'<line x1="875" y1="48" x2="900" y2="48" stroke="{SLATE}" stroke-width="4"/><text x="908" y="53" fill="{INK}" font-family="system-ui" font-size="14">M=24</text>']
    finish(p, ASSETS / "avl-cost.svg")


def system_overview():
    p = shell(
        "How one speculative round works",
        "One request; the Qwen3.6 target weights and numeric formats remain unchanged",
        1120,
        520,
    )
    p.append(
        '<defs><marker id="overview-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        f'fill="{MUTED}"/></marker></defs>'
    )

    cards = [
        (40, "1", "DFlash2 drafts", "Propose 8, 16, or 24", "positions in parallel", "Longer-block training"),
        (305, "2", "Target verifies", "Qwen3.6 checks the", "candidate prefix", "AVL + MoE remap"),
        (570, "3", "Useful work shrinks", "Attention follows the", "selected prefix", "L-skip"),
        (835, "4", "Prefix commits", "Accepted tokens advance", "the request", "Exact M=24 kernels"),
    ]
    for i, (x, n, title, line1, line2, optimization) in enumerate(cards):
        p.append(f'<rect x="{x}" y="130" width="245" height="205" rx="14" fill="#fafbf9" stroke="{GRID}" stroke-width="2"/>')
        color = SLATE if i in (0, 3) else TEAL
        p.append(f'<circle cx="{x+30}" cy="161" r="17" fill="{color}"/>')
        p.append(f'<text x="{x+30}" y="167" text-anchor="middle" fill="white" font-family="system-ui" font-size="15" font-weight="700">{n}</text>')
        p.append(f'<text x="{x+20}" y="210" fill="{INK}" font-family="system-ui" font-size="20" font-weight="700">{escape(title)}</text>')
        p.append(f'<text x="{x+20}" y="244" fill="{MUTED}" font-family="system-ui" font-size="16">{escape(line1)}</text>')
        p.append(f'<text x="{x+20}" y="271" fill="{MUTED}" font-family="system-ui" font-size="16">{escape(line2)}</text>')
        p.append(f'<rect x="{x+20}" y="291" width="205" height="30" rx="6" fill="#edf5f2"/>')
        p.append(f'<text x="{x+122.5}" y="311" text-anchor="middle" fill="{TEAL}" font-family="system-ui" font-size="13" font-weight="700">{escape(optimization)}</text>')
        if i < len(cards) - 1:
            p.append(f'<line x1="{x+249}" y1="230" x2="{x+260}" y2="230" stroke="{MUTED}" stroke-width="2" marker-end="url(#overview-arrow)"/>')

    p.append(f'<rect x="40" y="370" width="1040" height="58" rx="10" fill="#eef4f1" stroke="#dbe8e0"/>')
    p.append(f'<text x="560" y="395" text-anchor="middle" fill="{INK}" font-family="system-ui" font-size="15" font-weight="700">The optimization target: more accepted tokens without proportional verification cost</text>')
    p.append(f'<text x="560" y="417" text-anchor="middle" fill="{MUTED}" font-family="system-ui" font-size="14">Fixed throughout: target weights, FP8 deployment formats, greedy target verification, and one-GPU serving</text>')
    p.append(f'<text x="560" y="474" text-anchor="middle" fill="{MUTED}" font-family="system-ui" font-size="14">The next sections follow this round from the drafter to the GPU kernels, then measure the combined result.</text>')
    finish(p, ASSETS / "system-overview.svg")


def avl_lskip_flow():
    p = shell(
        "One cutoff, two kinds of saved work",
        "Illustrative round with a fixed M=24 CUDA graph and AVL selecting L=8",
        1120,
        470,
    )
    p.append(
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        f'fill="{MUTED}"/></marker></defs>'
    )
    cards = [
        (40, "1", "DFlash2", "Proposes 24", "parallel positions", SLATE),
        (255, "2", "AVL", "Selects L = 8", "for this round", TEAL),
        (470, "3", "MoE remap", "Rows 8–23 reuse", "row 0 experts", TEAL),
        (685, "4", "L-skip", "Selects the 1-tile", "FlashInfer plan", TEAL_2),
        (900, "5", "Acceptance", "Positions 8–23", "cannot commit", SLATE),
    ]
    for i, (x, n, title, line1, line2, color) in enumerate(cards):
        p.append(f'<rect x="{x}" y="135" width="180" height="168" rx="13" fill="#fafbf9" stroke="{GRID}" stroke-width="2"/>')
        p.append(f'<circle cx="{x+28}" cy="164" r="16" fill="{color}"/><text x="{x+28}" y="170" text-anchor="middle" fill="white" font-family="system-ui" font-size="15" font-weight="700">{n}</text>')
        p.append(f'<text x="{x+18}" y="210" fill="{INK}" font-family="system-ui" font-size="19" font-weight="700">{escape(title)}</text>')
        p.append(f'<text x="{x+18}" y="242" fill="{MUTED}" font-family="system-ui" font-size="16">{escape(line1)}</text>')
        p.append(f'<text x="{x+18}" y="268" fill="{MUTED}" font-family="system-ui" font-size="16">{escape(line2)}</text>')
        if i < len(cards) - 1:
            p.append(f'<line x1="{x+184}" y1="219" x2="{x+210}" y2="219" stroke="{MUTED}" stroke-width="2" marker-end="url(#arrow)"/>')
    p.append(f'<rect x="255" y="323" width="395" height="42" rx="8" fill="#edf5f2"/>')
    p.append(f'<text x="452" y="349" text-anchor="middle" fill="{TEAL}" font-family="system-ui" font-size="16" font-weight="700">AVL reduces MoE expert work</text>')
    p.append(f'<rect x="685" y="323" width="395" height="42" rx="8" fill="#f0f4f3"/>')
    p.append(f'<text x="882" y="349" text-anchor="middle" fill="{TEAL}" font-family="system-ui" font-size="16" font-weight="700">L-skip reduces attention work</text>')
    p.append(f'<rect x="40" y="394" width="1040" height="42" rx="8" fill="#f5f7f4" stroke="{GRID}"/>')
    p.append(f'<text x="560" y="420" text-anchor="middle" fill="{INK}" font-family="system-ui" font-size="15">The CUDA graph keeps M=24 tensors; L changes useful work and the acceptance cutoff, not the captured shape.</text>')
    finish(p, ASSETS / "avl-lskip-flow.svg")


def block24_overhead():
    p = shell(
        "How the block-24 overhead came down",
        "Additional round time relative to block 16 + AVL on the 128-turn engineering set",
        1000,
        520,
    )
    left, right = 315, 930
    top, row_gap = 135, 76
    rows = [
        ("Plain block 24", 1.90, SLATE),
        ("After L-skip", 1.56, TEAL_2),
        ("After exact fixes, off", 1.29, TEAL),
        ("After exact fixes, on", 0.76, TEAL),
    ]
    for tick in (0, 0.5, 1.0, 1.5, 2.0):
        x = left + tick / 2.0 * (right - left)
        p.append(f'<line x1="{x:.1f}" y1="112" x2="{x:.1f}" y2="424" stroke="{GRID}"/>')
        p.append(f'<text x="{x:.1f}" y="454" text-anchor="middle" fill="{MUTED}" font-family="system-ui" font-size="14">{tick:.1f}</text>')
    for i, (label, value, color) in enumerate(rows):
        y = top + i * row_gap
        width = value / 2.0 * (right - left)
        p.append(f'<text x="{left-18}" y="{y+24}" text-anchor="end" fill="{INK}" font-family="system-ui" font-size="16" font-weight="650">{escape(label)}</text>')
        p.append(f'<rect x="{left}" y="{y}" width="{width:.1f}" height="36" rx="6" fill="{color}"/>')
        p.append(f'<text x="{left+width+12:.1f}" y="{y+24}" fill="{INK}" font-family="system-ui" font-size="16" font-weight="700">+{value:.2f} ms</text>')
    p.append(f'<text x="{(left+right)/2:.1f}" y="486" text-anchor="middle" fill="{MUTED}" font-family="system-ui" font-size="15">Additional milliseconds per speculative round · lower is better</text>')
    finish(p, ASSETS / "block24-overhead.svg")


def block24_profile():
    p = shell(
        "Why block 24 regressed before the fixes",
        "Nsight Systems attribution relative to block 16 + AVL; only labeled timings are separately quantified",
        1120,
        600,
    )
    p.append(f'<rect x="42" y="118" width="1036" height="76" rx="12" fill="#eef4f1" stroke="#dbe8e0"/>')
    p.append(f'<text x="70" y="151" fill="{TEAL}" font-family="system-ui" font-size="25" font-weight="750">+1.90 ms / round</text>')
    p.append(f'<text x="70" y="177" fill="{MUTED}" font-family="system-ui" font-size="15">Plain block 24 overhead before L-skip and exact M=24 fixes</text>')
    cards = [
        (42, "Bole solve", "T=24 padded to 32", "+0.39 ms / round", "BV8 + 2 warps"),
        (304, "Dense FP8 GEMMs", "Split-K + reduction", "+0.44 ms / round", "Narrow path to M≤24"),
        (566, "Full attention", "Third 64-row query tile", "Observed in trace", "L-skip by AVL L"),
        (828, "BA + convolution", "Fast paths ended at 16", "Observed in trace", "Remaining exact headroom"),
    ]
    for x, title, cause, evidence, fix in cards:
        p.append(f'<rect x="{x}" y="222" width="240" height="190" rx="12" fill="#ffffff" stroke="{GRID}" stroke-width="2"/>')
        p.append(f'<text x="{x+18}" y="258" fill="{INK}" font-family="system-ui" font-size="18" font-weight="700">{escape(title)}</text>')
        p.append(f'<text x="{x+18}" y="290" fill="{MUTED}" font-family="system-ui" font-size="15">{escape(cause)}</text>')
        color = TEAL if evidence.startswith("+") else SLATE
        p.append(f'<rect x="{x+18}" y="314" width="204" height="35" rx="6" fill="#f3f6f3"/>')
        p.append(f'<text x="{x+120}" y="337" text-anchor="middle" fill="{color}" font-family="system-ui" font-size="14" font-weight="700">{escape(evidence)}</text>')
        p.append(f'<line x1="{x+120}" y1="360" x2="{x+120}" y2="378" stroke="{MUTED}" stroke-width="2"/>')
        p.append(f'<path d="M {x+115} 373 L {x+120} 381 L {x+125} 373" fill="none" stroke="{MUTED}" stroke-width="2"/>')
        p.append(f'<text x="{x+120}" y="400" text-anchor="middle" fill="{INK}" font-family="system-ui" font-size="14">{escape(fix)}</text>')
    p.append(f'<rect x="42" y="444" width="1036" height="86" rx="12" fill="#f5f7f4" stroke="{GRID}"/>')
    p.append(f'<text x="70" y="478" fill="{INK}" font-family="system-ui" font-size="17" font-weight="700">After the targeted fixes</text>')
    p.append(f'<text x="70" y="507" fill="{MUTED}" font-family="system-ui" font-size="15">Block-24 overhead fell to +1.29 ms thinking off and +0.76 ms thinking on; the 128-turn result reproduced on two boots.</text>')
    p.append(f'<text x="42" y="566" fill="{MUTED}" font-family="system-ui" font-size="13">The four cards are a profiler map, not a stacked total. Attention and BA/conv were observed in the trace but not assigned standalone milliseconds.</text>')
    finish(p, ASSETS / "block24-profile.svg")


if __name__ == "__main__":
    throughput()
    context_speedup()
    drafter_blocks()
    avl_cost()
    system_overview()
    avl_lskip_flow()
    block24_overhead()
    block24_profile()
    print("generated 8 SVG figures")

