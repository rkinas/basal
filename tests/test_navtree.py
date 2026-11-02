"""Trees for basal-navigate (basal/navtree.py) without a model: Markdown, HTML, text and JSONL documents parsed into
pages, sections and passages, folders as topics, the tree file, and k-means topics (with scikit-learn)."""
import json

import pytest

from basal import navigate, navtree
from basal.navigate import Tree


MD = """# Renew a passport

Intro line one
continues here.

## Fees

- Card: 30
- Cash: 35

| item | price |
|---|---|
| photo | 5 |

### Refunds

```
code block
```

## Where
At the office.
"""

HTML = """<html><head><title>Visa guide</title><script>var x = 1;</script></head><body>
<h1>Visa guide</h1><p>Visas take <b>two</b> weeks.</p>
<h2>Costs</h2><ul><li>Adult: 60</li><li>Child: 30</li></ul>
<table><tr><th>Type</th><th>Days</th></tr><tr><td>Tourist</td><td>90</td></tr></table>
<h3>Waivers</h3><p>Some countries need none.</p></body></html>"""


def test_parse_markdown_html_text():
    title, blocks = navtree.parse_markdown(MD)
    assert title == "Renew a passport"
    assert blocks == [((), "Intro line one continues here."), (("Fees",), "Card: 30"), (("Fees",), "Cash: 35"),
                      (("Fees",), "item | price"), (("Fees",), "photo | 5"), (("Fees", "Refunds"), "code block"),
                      (("Where",), "At the office.")]
    title, blocks = navtree.parse_html(HTML)
    assert title == "Visa guide"
    assert blocks == [((), "Visas take two weeks."), (("Costs",), "Adult: 60"), (("Costs",), "Child: 30"),
                      (("Costs",), "Type | Days"), (("Costs",), "Tourist | 90"),
                      (("Costs", "Waivers"), "Some countries need none.")]
    assert navtree.parse_text("a\nb\n\n\nc") == (None, [((), "a b"), ((), "c")])
    title, blocks = navtree.parse_html("<h1>Overview</h1><p>x</p><h1>Next</h1><p>y</p>", keep_h1=True)
    assert title is None and blocks == [(("Overview",), "x"), (("Next",), "y")]
    title, blocks = navtree.parse_markdown("# Overview\nx", keep_h1=True)
    assert title is None and blocks == [(("Overview",), "x")]


def test_split_and_merge_passages():
    long = "First sentence here. " * 30
    pieces = navtree.split_long(long.strip(), 100)
    assert all(len(p) <= 100 for p in pieces) and " ".join(pieces) == long.strip()
    merged = navtree.passages([(("A",), "x"), (("A",), "y"), (("B",), "z")], min_chars=5)
    assert merged == [(("A",), "x\ny"), (("B",), "z")]


def test_build_pages_sections_and_dirs(tmp_path):
    (tmp_path / "travel").mkdir()
    (tmp_path / "travel" / "passport.md").write_text(MD, encoding="utf-8")
    (tmp_path / "travel" / "visa.html").write_text(HTML, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Plain one.\n\nPlain two.", encoding="utf-8")
    (tmp_path / "more.jsonl").write_text(json.dumps({"id": "d1", "title": "Doc one", "markdown": "## A\nText a."})
                                         + "\n", encoding="utf-8")
    nodes = navtree.build([str(tmp_path)], split_long=3, group_dirs=True)
    t = Tree(nodes)
    by = t.nodes
    assert by["travel/passport"]["kind"] == "page" and by["travel/passport"]["parent"] == "dir:travel"
    assert by["travel/passport"]["text"] == "Renew a passport: Fees; Where"
    assert by["dir:travel"]["text"] == "Folder travel: Renew a passport; Visa guide"
    assert by["travel/passport#1"]["parent"] == "travel/passport"  # no heading: directly under its long page
    s = by["travel/passport#2"]["parent"]
    assert by[s]["kind"] == "section" and by[s]["text"] == "Fees: Refunds" and by["travel/passport#2"]["path"] == "Fees"
    assert by["notes"]["parent"] is None and t.leaves("notes") == ["notes#1", "notes#2"]
    assert by["d1"]["title"] == "Doc one" and by["d1#1"]["path"] == "A"
    out = tmp_path / "tree.jsonl"
    navigate.main(["build", str(tmp_path / "travel"), "-o", str(out)])
    assert Tree.load(out).nodes["passport"]["kind"] == "page"


def test_cluster_topics():
    pytest.importorskip("sklearn")
    texts = [f"passport visa travel border {i}" for i in range(12)] + [f"tax income return refund {i}"
                                                                         for i in range(12)]
    topics = navtree.cluster(texts, 2, max_children=0)
    assert len(topics) == 2 and sorted(len(t["members"]) for t in topics) == [12, 12]
    assert all(t["text"].startswith("Pages such as: ") for t in topics)
    assert not any(" the" in t["text"].split("Key terms:")[1] for t in navtree.cluster(
        [f"the {x} and the {x}s in {i}" for x in ("visa", "tax") for i in range(6)], 2))  # no stop words as terms
    deep = navtree.cluster(texts, 2, max_children=5)
    assert any(t["parent"] for t in deep) and sum(len(t["members"]) for t in deep) == 24
    vec = navtree.cluster(texts, 2, embed=lambda xs: [[1.0, 0.0] if "tax" in x else [0.0, 1.0] for x in xs])
    assert sorted(len(t["members"]) for t in vec) == [12, 12]
