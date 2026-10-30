"""Trees for basal-navigate (experimental, 2.5.1): documents -> pages -> passages, optionally sections under long
pages, folders as topics, or topics found by k-means (basal[navigate]: scikit-learn).

  build(["docs/"], split_long=60)        -> the nodes, for basal.navigate.Tree (basal-navigate build writes them)
  cluster(texts, 100, embed=None)        -> topic assignments (recursive with max_children)

Documents: Markdown (# headings; paragraphs, list items and table rows are passages), HTML (h1-h6 headings; p, li, tr,
dt, dd, pre, blockquote are passages), plain text (blank-line paragraphs) and JSONL ({"id"?, "title"?, "text" |
"markdown" | "html", "group"?} per line). A page's description is its title and its top headings; a passage keeps its
heading path. Passages longer than max_chars are split at sentence ends; with min_chars, short neighbours under one
heading are merged. A cluster is described without generation: its most central titles and its top terms.
"""
import html.parser
import json
import re
from pathlib import Path

from .tasks import units

SUFFIXES = {".md": "markdown", ".markdown": "markdown", ".html": "html", ".htm": "html", ".txt": "text",
            ".jsonl": "jsonl"}
PAGE_CHARS = 300
CENTRAL, TERMS = 6, 6
DESCRIBE = {"en": ("Pages such as: {titles}. Key terms: {terms}.", "Folder {name}: {titles}"),
            "pl": ("Strony takie jak: {titles}. Kluczowe słowa: {terms}.", "Folder {name}: {titles}")}
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def _clean(s):
    return re.sub(r"\s+", " ", s).strip()


def parse_markdown(text):
    """(title or None, [(heading path, passage text)]): ATX headings; paragraphs, list items and table rows."""
    title, path, out, buf, fence = None, [], [], [], None

    def flush():
        t = _clean(" ".join(buf)) if fence is None else "\n".join(buf).strip()
        if t:
            out.append((tuple(h for _, h in path), t))
        buf.clear()
    for line in text.splitlines():
        if fence is not None:
            if line.strip().startswith(fence):
                flush()
                fence = None
            else:
                buf.append(line)
            continue
        if line.strip().startswith(("```", "~~~")):
            flush()
            fence = line.strip()[:3]
            continue
        m = HEADING.match(line)
        if m:
            flush()
            level, h = len(m.group(1)), _clean(m.group(2))
            if level == 1 and title is None and not out:
                title = h
                continue
            path = [x for x in path if x[0] < level] + [(level, h)]
        elif not line.strip():
            flush()
        elif ITEM.match(line) or line.lstrip().startswith("|"):
            flush()
            if not re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line):  # a table's separator row
                buf.append(ITEM.sub("", line, count=1).strip().strip("|").replace("|", " | "))
            flush()
        else:
            buf.append(line)
    flush()
    return title, out


class _HTML(html.parser.HTMLParser):
    UNITS = {"p", "li", "tr", "dt", "dd", "pre", "blockquote", "caption", "figcaption"}
    SKIP = {"script", "style", "noscript", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.path, self.out, self.buf, self.where, self.skip, self.cells = None, [], [], [], None, 0, []

    def flush(self):
        if self.where == "tr" and self.cells:
            self.cells.append(_clean("".join(self.buf)))
            t = " | ".join(c for c in self.cells if c)
        else:
            t = _clean("".join(self.buf))
        if t and self.where not in (None, "title") and not self.where.startswith("h"):
            self.out.append((tuple(h for _, h in self.path), t))
        elif t and self.where == "title" and self.title is None:
            self.title = t
        elif t and self.where and self.where[0] == "h" and self.where[1:].isdigit():
            level = int(self.where[1:])
            if level == 1 and not self.out and self.title in (None, t):  # the page title (also given by <title>)
                self.title = t
            else:
                self.path = [x for x in self.path if x[0] < level] + [(level, t)]
        self.buf, self.cells = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.UNITS or tag == "title" or re.fullmatch(r"h[1-6]", tag):
            if self.where != "tr" or tag not in ("p",):
                self.flush()
                self.where = tag
        elif tag in ("td", "th") and self.where == "tr":
            self.cells.append(_clean("".join(self.buf)))
            self.buf = []
        elif tag == "br":
            self.buf.append(" ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == self.where:
            self.flush()
            self.where = None

    def handle_data(self, data):
        if not self.skip and self.where is not None:
            self.buf.append(data)


def parse_html(text):
    """(title or None, [(heading path, passage text)]): h1-h6 headings; text outside the unit tags is left out."""
    p = _HTML()
    p.feed(text)
    p.close()
    p.flush()
    return p.title, p.out


def parse_text(text):
    return None, [((), _clean(b)) for b in re.split(r"\n[ \t]*\n", text) if b.strip()]


def split_long(text, max_chars):
    """Pieces of at most about max_chars, cut at sentence ends (a longer sentence is its own piece)."""
    if not max_chars or len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for a, b in units(text, "sentence"):
        s = text[a:b]
        if cur and len(cur) + 1 + len(s) > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    return out + ([cur] if cur else [])


def passages(blocks, max_chars=2000, min_chars=0):
    """Blocks -> passages: long ones split, short neighbours under one heading merged (min_chars)."""
    out = []
    for path, t in blocks:
        for piece in split_long(t, max_chars):
            if min_chars and out and out[-1][0] == path and len(out[-1][1]) < min_chars:
                out[-1] = (path, out[-1][1] + "\n" + piece)
            else:
                out.append((path, piece))
    return out


def documents(inputs):
    """(id, title, group, kind, text) of every document under the inputs (folders: in sorted order)."""
    for inp in inputs:
        root = Path(inp)
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES) \
            if root.is_dir() else [root]
        for f in files:
            kind = SUFFIXES.get(f.suffix.lower())
            if kind is None:
                raise ValueError(f"{f}: unknown document type (one of {', '.join(SUFFIXES)})")
            rel = f.relative_to(root) if root.is_dir() else Path(f.name)
            if kind == "jsonl":
                with open(f, encoding="utf-8") as h:
                    for n, line in enumerate(h):
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        k = next((x for x in ("markdown", "html", "text") if isinstance(d.get(x), str)), None)
                        if k is None:
                            raise ValueError(f"{f}:{n + 1}: a document needs text, markdown or html")
                        yield str(d.get("id", f"{rel.with_suffix('')}:{n + 1}")), d.get("title"), d.get("group"), \
                            k, d[k]
            else:
                yield str(rel.with_suffix("")), None, str(rel.parent) if str(rel.parent) != "." else None, kind, \
                    f.read_text(encoding="utf-8", errors="replace")


def page_description(title, blocks):
    tops = list(dict.fromkeys(p[0] for p, _ in blocks if p))
    s = title + (": " + "; ".join(tops) if tops else "")
    return s if len(s) <= PAGE_CHARS else s[: PAGE_CHARS - 1].rstrip() + "…"


def build(inputs, split_long=0, group_dirs=False, clusters=0, max_children=0, max_chars=2000, min_chars=0, lang=None,
          embed=None, seed=0):
    """The nodes of a tree (dicts for basal.navigate.Tree) from documents: [topics ->] pages [-> sections] ->
    passages. split_long: a section level under pages with more than this many passages; group_dirs: folders (or
    JSONL "group") as topics; clusters: k-means topics (cluster(); embed: your embedding function)."""
    if group_dirs and clusters:
        raise ValueError("give group_dirs or clusters, not both")
    pages, nodes = [], []
    parse = {"markdown": parse_markdown, "html": parse_html, "text": parse_text}
    for doc_id, title, group, kind, text in documents(inputs):
        t, blocks = parse[kind](text)
        title = _clean(title or t or doc_id.rsplit("/", 1)[-1])
        ps = passages(blocks, max_chars, min_chars)
        if ps:
            pages.append(dict(id=doc_id, title=title, group=group, passages=ps, text=page_description(title, blocks)))
    if not pages:
        raise ValueError("no passages found in the inputs")
    lang = lang or _majority_lang(p["title"] for p in pages)
    parent = {}
    if group_dirs:
        for g in dict.fromkeys(p["group"] for p in pages if p["group"]):
            members = [p["title"] for p in pages if p["group"] == g]
            nodes.append({"id": f"dir:{g}", "parent": None, "kind": "topic", "title": g,
                          "text": DESCRIBE[lang][1].format(name=g, titles="; ".join(members[:CENTRAL]))})
        parent = {p["id"]: f"dir:{p['group']}" for p in pages if p["group"]}
    elif clusters:
        texts = [p["title"] + ". " + (p["passages"][0][1][:300] if p["passages"] else "") for p in pages]
        for topic in cluster(texts, clusters, embed=embed, max_children=max_children, lang=lang, seed=seed,
                             titles=[p["title"] for p in pages]):
            nodes.append({"id": topic["id"], "parent": topic["parent"], "kind": "topic", "text": topic["text"]})
            for m in topic["members"]:
                parent[pages[m]["id"]] = topic["id"]
    for p in pages:
        nodes.append({"id": p["id"], "parent": parent.get(p["id"]), "kind": "page", "title": p["title"],
                      "text": p["text"]})
        sections = {}
        if split_long and len(p["passages"]) > split_long:
            for path, _ in p["passages"]:
                if path and path[0] not in sections:
                    sections[path[0]] = f"{p['id']}#s{len(sections) + 1}"
                    subs = list(dict.fromkeys(q[1] for q, _ in p["passages"] if len(q) > 1 and q[0] == path[0]))
                    nodes.append({"id": sections[path[0]], "parent": p["id"], "kind": "section", "title": path[0],
                                  "text": path[0] + (": " + "; ".join(subs) if subs else "")[:PAGE_CHARS]})
        for n, (path, t) in enumerate(p["passages"], 1):
            x = {"id": f"{p['id']}#{n}", "parent": sections.get(path[0], p["id"]) if path else p["id"],
                 "kind": "passage", "text": t}
            if path:
                x["path"] = " > ".join(path)
            nodes.append(x)
    return nodes


def _majority_lang(titles):
    from .prompt import lang_of
    titles = list(titles)
    return "pl" if sum(lang_of(t) == "pl" for t in titles) * 2 > len(titles) else "en"


def cluster(texts, k, embed=None, max_children=0, lang="en", seed=0, titles=None):
    """Topics over texts by k-means: [{"id", "parent", "text", "members"}] (members: indices of texts, only on the
    deepest topics). Vectors: embed(texts) when given (rows are normalised), else TF-IDF reduced by SVD. A topic of
    more than max_children texts is split again (recursive k-means), into ceil(n / max_children) or k topics, the
    smaller. Described by its most central titles and top TF-IDF terms."""
    try:
        import numpy as np
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
    except ImportError as e:
        raise SystemExit("topics by k-means need scikit-learn: pip install 'basal[navigate]'") from e
    titles = titles or texts
    tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=1, max_features=1 << 18,
                            token_pattern=r"(?u)\b\w\w+\b")
    T = tfidf.fit_transform(texts)
    vocab = tfidf.get_feature_names_out()
    if embed is not None:
        X = normalize(np.asarray(embed(texts), dtype=np.float32))
    else:
        dims = min(256, T.shape[1] - 1, T.shape[0] - 1)
        X = normalize(TruncatedSVD(dims, random_state=seed).fit_transform(T)) if dims >= 2 else T.toarray()
    out = []

    def split(idx, n, parent, prefix):
        n = max(1, min(n, len(idx)))
        km = (MiniBatchKMeans(n, random_state=seed, n_init=3, batch_size=4096) if len(idx) > 20000 else
              KMeans(n, random_state=seed, n_init=3)).fit(X[idx])
        for c in range(n):
            members = idx[km.labels_ == c]
            if not len(members):
                continue
            tid = f"{prefix}{len([o for o in out if o['parent'] == parent]) + 1}"
            centre = km.cluster_centers_[c]
            near = members[np.argsort(-(X[members] @ centre))][:CENTRAL]
            w = np.asarray(T[members].mean(axis=0)).ravel()
            terms = [vocab[j] for j in np.argsort(-w)[:TERMS] if w[j] > 0]
            topic = {"id": tid, "parent": parent, "text": DESCRIBE[lang][0].format(
                titles="; ".join(titles[j] for j in near), terms=", ".join(terms))}
            out.append(topic)
            if max_children and len(members) > max_children and len(members) < len(idx):
                split(members, min(k, -(-len(members) // max_children)), tid, tid + ".")
            else:
                topic["members"] = [int(j) for j in members]
    split(np.arange(len(texts)), k, None, "topic:")
    for t in out:
        t.setdefault("members", [])
    return out
