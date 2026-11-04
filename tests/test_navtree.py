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


def test_jsonl_document_fields_are_checked(tmp_path):
    """A JSONL document whose title is not a string is refused with its file and line, as one without text is (not a
    TypeError from deep inside the parser)."""
    f = tmp_path / "docs.jsonl"
    f.write_text(json.dumps({"id": "a", "title": "Raport", "text": "Jeden."}) + "\n"
                 + json.dumps({"id": "b", "title": 2024, "text": "Dwa."}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"docs.jsonl:2: .*title"):
        navtree.build([str(f)])
    f.write_text(json.dumps({"id": "b", "title": None, "text": "Dwa."}) + "\n", encoding="utf-8")
    assert Tree(navtree.build([str(f)])).nodes["b"]["title"] == "b"  # no title: the id, as before


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


def test_polish_topics_by_k_means():
    """--clusters with --lang pl: the Polish stop words went to scikit-learn as a tuple, which it refuses."""
    pytest.importorskip("sklearn")
    texts = [f"paszport wiza i podróż na granicy {i}" for i in range(6)] + [f"podatek dochodowy i zwrot podatku {i}"
                                                                          for i in range(6)]
    topics = navtree.cluster(texts, 2, lang="pl")
    assert sorted(len(t["members"]) for t in topics) == [6, 6]
    assert all(t["text"].startswith("Strony takie jak: ") for t in topics)
    terms = [w for t in topics for w in t["text"].split("Kluczowe słowa: ")[1].rstrip(".").split(", ")]
    assert terms and not set(terms) & set(navtree.STOP_PL)
    assert {"gdy", "go"} <= set(navtree.STOP_PL)


def test_build_polish_clusters_from_the_command_line(tmp_path):
    pytest.importorskip("sklearn")
    for i in range(4):
        (tmp_path / f"wiza{i}.md").write_text(f"# Wiza {i}\n\nWiza i paszport na granicy, opłata {i}.", encoding="utf-8")
        (tmp_path / f"podatek{i}.md").write_text(f"# Podatek {i}\n\nZwrot podatku dochodowego, termin {i}.",
                                                encoding="utf-8")
    out = tmp_path / "tree.jsonl"
    navigate.main(["build", str(tmp_path), "-o", str(out), "--clusters", "2", "--lang", "pl"])
    assert sum(x["kind"] == "topic" for x in Tree.load(out).nodes.values()) == 2


def test_the_same_page_id_from_two_inputs(tmp_path):
    """docs1/index.md and docs2/index.md were both the page "index" (the tree then refused the duplicate id): the
    page ids get their input in front, only when two inputs collide."""
    for d, text in (("docs1", "One."), ("docs2", "Two.")):
        (tmp_path / d).mkdir()
        (tmp_path / d / "index.md").write_text(f"# Index {d}\n\n{text}", encoding="utf-8")
        (tmp_path / d / f"only_{d}.md").write_text(f"# Only {d}\n\nText.", encoding="utf-8")
    one, two = (tmp_path / "docs1").as_posix(), (tmp_path / "docs2").as_posix()
    t = Tree(navtree.build([one, two]))
    assert {i for i, x in t.nodes.items() if x["kind"] == "page"} == {
        f"{one}/index", f"{one}/only_docs1", f"{two}/index", f"{two}/only_docs2"}
    assert t.leaves(f"{two}/index") == [f"{two}/index#1"]
    alone = Tree(navtree.build([one]))  # one input (or no collision): the ids as before
    assert {i for i, x in alone.nodes.items() if x["kind"] == "page"} == {"index", "only_docs1"}
    with pytest.raises(ValueError, match="one input given twice"):
        navtree.build([one, one])


def test_the_tree_being_written_is_not_read_as_a_document(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nText of a.", encoding="utf-8")
    out = tmp_path / "tree.jsonl"
    for _ in range(2):  # the second build sees the first one's tree.jsonl in its input folder
        navigate.main(["build", str(tmp_path), "-o", str(out)])
        assert {i for i, x in Tree.load(out).nodes.items() if x["kind"] == "page"} == {"a"}
