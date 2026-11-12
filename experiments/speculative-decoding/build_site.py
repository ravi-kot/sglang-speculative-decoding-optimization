"""Build index.html from article.md (static page: no JavaScript, no external requests).

  python build_site.py

Markdown via markdown-it (as in my previous post), code highlighted at build time by Pygments,
equations as native MathML, and ledger citations like [L53] linked to data/ledger.jsonl.
"""

import json
import re
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

ROOT = Path(__file__).parent
TITLE = "Optimizing Speculative Decoding on an RTX 5070"
SUBTITLE = "CUDA, quantized drafting, split-KV attention, and the limits of memory bandwidth."
DESCRIPTION = ("How I made SGLang's NEXTN speculative decoding for Qwen3.5-2B up to 2.1x faster on an 8 GB laptop GPU "
               "without touching the BF16 target model.")
DATE = "October 27, 2026"


# ---------------------------------------------------------------- MathML helpers
def mi(x): return f"<mi>{escape(x)}</mi>"
def mn(x): return f"<mn>{escape(x)}</mn>"
def mo(x): return f"<mo>{escape(x)}</mo>"
def mt(x): return f"<mtext>{escape(x)}</mtext>"
def row(*xs): return "<mrow>" + "".join(xs) + "</mrow>"
def frac(a, b): return f"<mfrac>{a}{b}</mfrac>"
def sub(a, b): return f"<msub>{a}{b}</msub>"
def sp(w="1.2em"): return f'<mspace width="{w}"/>'
def unit(u): return sp("0.25em") + mt(u)


def eq(*xs, label=None):
    tag = f'<span class="eq-label">{escape(label)}</span>' if label else ""
    return f'<div class="eq"><math display="block">{row(*xs)}</math>{tag}</div>'


TPOT, R, A, T = mt("TPOT"), mi("R"), mi("A"), sub(mi("T"), mt("round"))
EQS = {
    "tpot": eq(TPOT, mo("="), frac(row(sub(mi("t"), mt("end-to-end")), mo("−"), sub(mi("t"), mt("first token"))),
                                   row(mi("N"), mo("−"), mn("1"))), label="N = output tokens"),
    "tps": eq(R, unit("[tokens/s]"), mo("="), frac(mn("1000"), row(TPOT, unit("[ms]")))),
    "gain": eq(mt("gain"), mo("="), row(mo("("), frac(mn("1"), mi("r")), mo("−"), mn("1"), mo(")")), mo("×"), mn("100%"),
               sp(), mi("r"), mo("="), sub(mt("median"), mi("i")),
               row(mo("("), frac(sub(TPOT, row(mi("i"), mo(","), mt("new"))), sub(TPOT, row(mi("i"), mo(","), mt("stock")))),
                   mo(")"))),
    "gemv": eq(mi("t"), mo("≈"), frac(mt("bytes"), mt("bandwidth")), mo("="),
               frac(mt("1.017 GB"), mt("346 GB/s")), mo("≈"), mt("2.94 ms"), label="measured: 2,939 µs"),
    "kv": eq(mn("6"), mo("×"), mn("2"), mo("×"), mn("2"), mo("×"), mn("256"), mo("×"), mt("2 B"), mo("+"),
             mt("2,048 B"), mo("="), mt("14,336 B per token"),
             label="layers × (K,V) × KV heads × head dim × BF16, + draft layer"),
    "round": eq(R, unit("[tokens/s]"), mo("="), frac(row(mn("1000"), mo("×"), A), row(T, unit("[ms]"))),
                label="A = committed tokens per round"),
    "step": eq(mt("add a step only if"), sp(), frac(row(mi("Δ"), A), A), mo(">"), frac(row(mi("Δ"), T), T)),
    "depth45": eq(frac(row(mi("Δ"), A), A), mo("="), frac(mn("3.29"), mn("3.06")), mo("−"), mn("1"), mo("="),
                  mt("7.5%"), sp(), frac(row(mi("Δ"), T), T), mo("="), frac(mn("19.03"), mn("17.71")), mo("−"),
                  mn("1"), mo("="), mt("7.5%"), label="from four to five draft steps"),
    "head32k": eq(mn("32,768"), mo("×"), mn("2,048"), mo("×"), mt("0.5 B"), mo("+"),
                  mn("32,768"), mo("×"), mn("64"), mo("×"), mt("2 B"), mo("="), mt("32 MiB"), mo("+"),
                  mt("4 MiB"), mo("="), mt("36 MiB"), label="INT4 weights + group-32 scales"),
    "floor": eq(sub(mi("T"), mt("min")), mo("≈"), frac(mt("bytes per verify"), mt("bandwidth")), mo("="),
                frac(mt("3.763 GB"), mt("368 GB/s")), mo("≈"), mt("10.2 ms")),
    "ceil": eq(R, mo("="), frac(row(mn("1000"), mo("×"), mn("3.16")), mt("14.98 ms")), mo("≈"),
               mt("211 tokens/s"), label="profiled round [L55]"),
}


# ---------------------------------------------------------------- markdown
FMT = HtmlFormatter(nowrap=True)


def code_block(content, lang, _attrs):
    try:
        lexer = get_lexer_by_name(lang or "text")
    except Exception:
        lexer = get_lexer_by_name("text")
    return f'<pre class="code"><code class="language-{escape(lang or "text")}">{highlight(content, lexer, FMT)}</code></pre>'


md = MarkdownIt("commonmark", {"html": True, "highlight": code_block}).enable("table")
src = (ROOT / "article.md").read_text(encoding="utf-8")
body = md.render(src)

# equations
body = re.sub(r"<p>\{\{eq:(\w+)\}\}</p>", lambda m: EQS[m.group(1)], body)
missing = re.findall(r"\{\{eq:(\w+)\}\}", body)
assert not missing, missing


# heading anchors + contents
def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", t).lower()).strip("-")


heads = []


def heading(level):
    def f(m):
        text = m.group(1)
        explicit = re.search(r"\s*\{#([\w-]+)\}\s*$", text)  # "## Title {#anchor}"
        if explicit:
            text, s = text[: explicit.start()], explicit.group(1)
        else:
            s = slug(text)
        if level == 2:
            heads.append((s, re.sub(r"<[^>]+>", "", text)))
        return f'<h{level} id="{s}">{text}</h{level}>'
    return f


body = re.sub(r"<h2>(.*?)</h2>", heading(2), body)
body = re.sub(r"<h3>(.*?)</h3>", heading(3), body)

# figure and table numbering in order of appearance; {{ref:label}} resolves to "Figure N" / "Table N"
labels, count = {}, {"figure": 0, "table": 0}
for kind, label in re.findall(r"\{\{(figure|table):([\w-]+)\}\}", body):
    count[kind] += 1
    labels[label] = (kind, count[kind])
body = re.sub(r"<p>\{\{table:([\w-]+)\}\}\s*(.*?)</p>",
              lambda m: f'<p class="caption" id="table-{m.group(1)}"><span class="cap-num">Table '
                        f'{labels[m.group(1)][1]}.</span> {m.group(2)}</p>', body, flags=re.S)
body = re.sub(r"<figcaption>\{\{figure:([\w-]+)\}\}\s*",
              lambda m: f'<figcaption id="figure-{m.group(1)}"><span class="cap-num">Figure '
                        f'{labels[m.group(1)][1]}.</span> ', body)
body = re.sub(r"\{\{ref:([\w-]+)\}\}", lambda m: f'<a href="#{labels[m.group(1)][0]}-{m.group(1)}">'
              f'{labels[m.group(1)][0].capitalize()} {labels[m.group(1)][1]}</a>', body)
leftover = re.findall(r"\{\{[^}]*\}\}", body.replace("{{toc}}", ""))
assert not leftover, leftover
items = "".join(f'<li><a href="#{s}">{escape(t)}</a></li>' for s, t in heads)
# inline list for narrow screens; a sticky sidebar copy is shown instead on wide screens
toc = f'<nav class="toc toc-inline" aria-label="Contents"><p class="toc-head">Contents</p><ol>{items}</ol></nav>'
sidebar = f'<aside class="toc-side" aria-label="On this page"><p class="toc-head">On this page</p><ol>{items}</ol></aside>'
body = body.replace("<p>{{toc}}</p>", toc)

# tables scroll on narrow screens; figure drawings get a scroll wrapper; external links open in a new tab
body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
body = re.sub(r"(<figure[^>]*>\s*)(<img [^>]+>)", r'\1<div class="fig-body">\2</div>', body)
body = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener noreferrer"', body)


LEDGER = [json.loads(l) for l in (ROOT / "data" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]


def cite(m):
    if m.group(0).startswith("<a"):
        return m.group(0)
    n = int(m.group(1))
    hyp = escape(LEDGER[n - 1].get("hypothesis", "")[:140], quote=True)
    return f'<a class="cite" href="data/ledger.html#L{n}" title="Ledger entry {n}: {hyp}">L{n}</a>'


body = re.sub(r'<a [^>]*>.*?</a>|\[L(\d+)\]', cite, body)


def ledger_page():
    """data/ledger.html: every experiment, one anchor per entry, so citations land on the evidence."""
    rows = []
    for n, e in enumerate(LEDGER, 1):
        verdict = escape(str(e.get("verdict") or "(pass 1 of a paired A/B; the verdict is recorded on pass 2)"))
        rows.append(f'<section class="entry" id="L{n}"><h3>L{n} <span>{escape(e["run_id"])} · {escape(str(e.get("status")))}'
                    f'</span></h3><p><strong>Hypothesis.</strong> {escape(e.get("hypothesis", ""))}</p>'
                    f'<p><strong>Result.</strong> {verdict}</p></section>')
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>Experiment ledger</title>'
            f'<link rel="stylesheet" href="../style.css"></head><body class="ledger">'
            f'<header class="site"><a href="../index.html">Ravi Kothari</a><span>Experiment ledger</span></header>'
            f'<main><article><header class="post"><h1>Experiment ledger</h1><p class="subtitle">Every experiment '
            f'behind <a href="../index.html">the article</a>, including failures, in the order it ran. Citations such '
            f'as L53 link here. Raw records: <a href="ledger.jsonl">ledger.jsonl</a>.</p></header>'
            f'{"".join(rows)}</article></main></body></html>\n')
    (ROOT / "data" / "ledger.html").write_text(html, encoding="utf-8")


ledger_page()

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(TITLE)}</title>
<meta name="description" content="{escape(DESCRIPTION)}">
<meta name="author" content="Ravi Kothari">
<meta name="theme-color" content="#faf9f6">
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="site"><a class="brand" href="index.html">Ravi Kothari</a><a class="repo-link" href="https://github.com/ravi-kot/sglang-speculative-decoding-optimization" target="_blank" rel="noopener noreferrer">GitHub repository</a></header>
<main>
<article>
<header class="post">
<h1>{escape(TITLE)}</h1>
<p class="subtitle">{escape(SUBTITLE)}</p>
<p class="meta">{DATE} · RTX 5070 Laptop GPU (SM120, 8 GB) · SGLang 0.5.21 · Qwen3.5-2B BF16</p>
</header>
{body}
</article>
{sidebar}
</main>
<footer class="site-footer">Measured on one RTX 5070 Laptop GPU. Data, code and the full experiment ledger are in this repository.</footer>
</body>
</html>
"""
(ROOT / "index.html").write_text(page, encoding="utf-8")
print(f"index.html: {len(page):,} bytes, {page.count('<h2')} sections, {count['table']} numbered tables "
      f"({page.count('<table>')} total), {count['figure']} figures, {page.count('<math')} equations; "
      f"ledger.html: {len(LEDGER)} entries")
