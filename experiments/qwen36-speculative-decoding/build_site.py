"""Build the static article from article.md."""

import re
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

ROOT = Path(__file__).parent
TITLE = "Optimizing Qwen3.6 Speculative Decoding: 975 tok/s on One RTX PRO 6000"
SUBTITLE = "Block diffusion, cost-aware verification, and exact kernel fixes on an RTX PRO 6000."
DESCRIPTION = "A measured path from stock SGLang to 975 tok/s for Qwen3.6-35B-A3B-FP8 on one RTX PRO 6000."
DATE = "November 11, 2026"

FMT = HtmlFormatter(nowrap=True)


def code_block(content, lang, _attrs):
    try:
        lexer = get_lexer_by_name(lang or "text")
    except Exception:
        lexer = get_lexer_by_name("text")
    rendered = highlight(content, lexer, FMT)
    return f'<pre class="code"><code class="language-{escape(lang or "text")}">{rendered}</code></pre>'


md = MarkdownIt("commonmark", {"html": True, "highlight": code_block}).enable("table")
body = md.render((ROOT / "article.md").read_text(encoding="utf-8"))


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-")


heads = []


def heading(level):
    def replace(match):
        text = match.group(1)
        explicit = re.search(r"\s*\{#([\w-]+)\}\s*$", text)
        if explicit:
            text, anchor = text[: explicit.start()], explicit.group(1)
        else:
            anchor = slug(text)
        if level == 2:
            heads.append((anchor, re.sub(r"<[^>]+>", "", text)))
        return f'<h{level} id="{anchor}">{text}</h{level}>'

    return replace


body = re.sub(r"<h2>(.*?)</h2>", heading(2), body)
body = re.sub(r"<h3>(.*?)</h3>", heading(3), body)

labels, counts = {}, {"figure": 0, "table": 0}
for kind, label in re.findall(r"\{\{(figure|table):([\w-]+)\}\}", body):
    counts[kind] += 1
    labels[label] = (kind, counts[kind])

body = re.sub(
    r"<p>\{\{table:([\w-]+)\}\}\s*(.*?)</p>",
    lambda m: f'<p class="caption" id="table-{m.group(1)}"><span class="cap-num">Table '
    f'{labels[m.group(1)][1]}.</span> {m.group(2)}</p>',
    body,
    flags=re.S,
)
body = re.sub(
    r"<figcaption>\{\{figure:([\w-]+)\}\}\s*",
    lambda m: f'<figcaption id="figure-{m.group(1)}"><span class="cap-num">Figure '
    f'{labels[m.group(1)][1]}.</span> ',
    body,
)
body = re.sub(
    r"\{\{ref:([\w-]+)\}\}",
    lambda m: f'<a href="#{labels[m.group(1)][0]}-{m.group(1)}">'
    f'{labels[m.group(1)][0].capitalize()} {labels[m.group(1)][1]}</a>',
    body,
)

items = "".join(f'<li><a href="#{anchor}">{escape(title)}</a></li>' for anchor, title in heads)
toc = f'<nav class="toc toc-inline" aria-label="Contents"><p class="toc-head">Contents</p><ol>{items}</ol></nav>'
sidebar = f'<aside class="toc-side" aria-label="On this page"><p class="toc-head">On this page</p><ol>{items}</ol></aside>'
body = body.replace("<p>{{toc}}</p>", toc)

leftover = re.findall(r"\{\{[^}]*\}\}", body)
assert not leftover, leftover

body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
body = re.sub(r"(<figure[^>]*>\s*)(<img [^>]+>)", r'\1<div class="fig-body">\2</div>', body)
body = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener noreferrer"', body)

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
<header class="site"><a class="brand" href="../../index.html">Ravi Kothari</a><a class="repo-link" href="https://github.com/ravi-kot/sglang-speculative-decoding-optimization/tree/main/experiments/qwen36-speculative-decoding">GitHub repository</a></header>
<main>
<article>
<header class="post">
<h1>{escape(TITLE)}</h1>
<p class="subtitle">{escape(SUBTITLE)}</p>
<p class="meta">{DATE} · RTX PRO 6000 Blackwell Server Edition · SGLang 0.5.21 · Qwen3.6-35B-A3B-FP8</p>
</header>
{body}
</article>
{sidebar}
</main>
<footer class="site-footer">Measured on one RTX PRO 6000 Blackwell Server Edition. Curated data and implementation files are included with this article.</footer>
<script>
(() => {{
  const headings = [...document.querySelectorAll("article h2[id]")];
  const links = [...document.querySelectorAll(".toc-side a")];
  const byId = new Map(links.map((link) => [link.hash.slice(1), link]));
  let ticking = false;

  function updateActiveSection() {{
    let current = null;
    for (const heading of headings) {{
      if (heading.getBoundingClientRect().top <= 180) current = heading;
      else break;
    }}
    for (const link of links) {{
      const active = Boolean(current && link === byId.get(current.id));
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }}
    ticking = false;
  }}

  addEventListener("scroll", () => {{
    if (!ticking) {{
      requestAnimationFrame(updateActiveSection);
      ticking = true;
    }}
  }}, {{ passive: true }});
  updateActiveSection();
}})();
</script>
</body>
</html>
"""

(ROOT / "index.html").write_text(page, encoding="utf-8")
print(f"index.html: {len(page):,} bytes, {len(heads)} sections, {counts['table']} tables, {counts['figure']} figures")

