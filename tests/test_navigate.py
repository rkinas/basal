"""basal-navigate (basal/navigate.py, basal/navtree.py) without a model: a tree navigated through the rank question of
a stub server (Server.decide with position probabilities from the candidates' texts), budgets, pins, extra passages,
the last stage and its prior, dialog and Polish wording; the tree file; documents parsed into trees."""
import asyncio

import pytest

from basal import navigate
from basal.navigate import Navigator, Tree, select

from test_rank import relevance
from test_server import FakeServer


class Client:
    """The request(body) of basal.client.Basal, answered by a stub server in process."""

    def __init__(self, weight):
        self.srv, self.bodies = FakeServer(relevance(weight)), []

    def request(self, body):
        self.bodies.append(body)
        return asyncio.run(self.srv._run(body))


def passport(text):
    t = text.lower()
    return 1 + 8 * ("passport" in t) + 2 * ("travel" in t)


NODES = [
    {"id": "t1", "parent": None, "kind": "topic", "text": "Taxes: paying and filing"},
    {"id": "t2", "parent": None, "kind": "topic", "text": "Travel: passport and visas"},
    {"id": "t3", "parent": None, "kind": "topic", "text": "Health: doctors"},
    {"id": "p1", "parent": "t1", "kind": "page", "title": "Pay tax", "text": "Pay tax online"},
    {"id": "p1#1", "parent": "p1", "kind": "passage", "text": "Pay your tax online with a card."},
    {"id": "p1#2", "parent": "p1", "kind": "passage", "text": "Deadlines are in January."},
    {"id": "p2", "parent": "t1", "kind": "page", "title": "File", "text": "File a return"},
    {"id": "p2#1", "parent": "p2", "kind": "passage", "text": "File the return by April."},
    {"id": "p3", "parent": "t2", "kind": "page", "title": "Passport", "text": "Renew a passport"},
    {"id": "p3#s1", "parent": "p3", "kind": "section", "title": "Fees", "text": "Fees: card, cash"},
    {"id": "p3#1", "parent": "p3#s1", "kind": "passage", "text": "A passport costs 30.", "path": "Fees"},
    {"id": "p3#2", "parent": "p3", "kind": "passage", "text": "Renew your passport at the office."},
    {"id": "p4", "parent": "t2", "kind": "page", "title": "Visa", "text": "Visas for travel"},
    {"id": "p4#1", "parent": "p4", "kind": "passage", "text": "Visas take two weeks."},
    {"id": "p5", "parent": "t3", "kind": "page", "title": "Doctors", "text": "Find a doctor"},
    {"id": "p5#1", "parent": "p5", "kind": "passage", "text": "Call the clinic."},
]


def test_navigate_prunes_by_level_and_ranks_the_passages_reached():
    c = Client(passport)
    nav = Navigator(c, Tree(NODES), levels=[{"keep": 1}], final={"top": 3})
    r = nav.search("How do I renew my passport?", k=5)
    assert [s["kept"] for s in r["stages"][:-1]] == [["t2"], ["p3"], ["p3#s1"]]
    assert [s["requests"] for s in r["stages"]] == [1, 1, 0, 2]  # one section: no request; passages: screen + choice
    ids = [h["id"] for h in r["passages"]]
    assert set(ids) == {"p3#1", "p3#2"} and all("choice" in h for h in r["passages"])
    first = c.bodies[0]
    assert first["state"] == "User's question: How do I renew my passport?" and first["lang"] == "en"
    q = first["questions"]["topic"]
    assert q["instructions"].startswith("Is the page with the information") and q["label"] == "Topic"
    last = c.bodies[-1]["questions"]["passage"]
    assert [x["key"] for x in last["candidates"]] == ["p3#1", "p3#2"]  # reading order
    assert last["candidates"][0] == {"key": "p3#1", "text": "[Fees] A passport costs 30.",
                                     "group": "Passport > Fees"}
    assert last["candidates"][1]["group"] == "Passport" and last["top"] == 3 and last["chunk"] == 0
    assert last["choice_instructions"] == navigate.WORDS["en"]["question"]["choice"]
    pages = c.bodies[1]["questions"]["page"]["candidates"]
    assert pages == [{"key": "p3", "text": "Renew a passport", "group": "Travel: passport and visas"},
                     {"key": "p4", "text": "Visas for travel", "group": "Travel: passport and visas"}]


def test_navigate_pin_extra_multiply_and_sure():
    c = Client(passport)
    nav = Navigator(c, Tree(NODES), levels=[{"keep": 3, "sure": 0.8}, {"keep": 1}], final={"top": 0, "multiply": True})
    r = nav.search("passport renewal", k=10, pin=["p1", "nope"], extra=["p5#1", "p3"])
    assert r["stages"][0]["kept"] == ["t2"]  # the top topic is sure (9 / 10 + travel): only it
    last = c.bodies[-1]["questions"]["passage"]
    keys = [x["key"] for x in last["candidates"]]
    assert keys == ["p1#1", "p1#2", "p3#1", "p3#2", "p5#1"]  # pinned page p1, extra p5#1 (p3 is no passage)
    prior = last["prior"]
    # the nearest judged ancestor: pages p1 and p5 were not judged, their topics were (1 / 2 each); p3 was (9 / 10)
    assert prior["p1#1"] == prior["p5#1"] == pytest.approx(0.5) and prior["p3#2"] == pytest.approx(0.9)
    assert "choice" not in last.get("choice_instructions", "") and last["top"] == 0
    assert {h["id"] for h in r["passages"]} == set(keys)


def test_navigate_whole_level_within_budget_needs_no_request():
    c = Client(passport)
    nav = Navigator(c, Tree(NODES), levels=[{"keep": 10}], final={"top": 2})
    r = nav.search("passport")
    assert [s["requests"] for s in r["stages"][:-1]] == [0, 0, 0] and len(c.bodies) == 1
    assert len(c.bodies[0]["questions"]["passage"]["candidates"]) == 7


def test_navigate_dialog_and_polish_wording():
    c = Client(passport)
    nav = Navigator(c, Tree(NODES), levels=[{"keep": 1}], final={"top": 2})
    nav.search([{"role": "user", "text": "My passport expired."}, {"role": "assistant", "text": "Which one?"},
                "User: the blue one"])
    assert c.bodies[0]["state"] == ("Dialogue (the latest turn last):\nUser: My passport expired.\nAgent: Which one?\n"
                                    "User: the blue one")
    assert c.bodies[-1]["questions"]["passage"]["instructions"] == "Should the agent's next reply be based on {candidate}?"
    c = Client(passport)
    Navigator(c, Tree(NODES), levels=[{"keep": 1}], final={"top": 2}).search("Jak odnowić paszport?")
    q = c.bodies[0]["questions"]["topic"]
    assert c.bodies[0]["lang"] == "pl" and q["label"] == "Temat" and q["list_title"] == "Tematy"
    assert c.bodies[0]["state"] == "Pytanie użytkownika: Jak odnowić paszport?"
    assert c.bodies[-1]["questions"]["passage"]["choice_instructions"].startswith("Który fragment")


def test_select_budgets():
    r = [("a", 0.9), ("b", 0.5), ("c", 0.2)]
    assert select(r, {"keep": 2}) == r[:2]
    assert select(r, {"keep": 3, "min_p": 0.4}) == r[:2]
    assert select(r, {"keep": 3, "min_p": 0.95}) == r[:1]
    assert select(r, {"keep": 2, "sure": 0.8}) == r[:1] and select(r, {"keep": 2, "sure": 0.95}) == r[:2]


def test_tree_file_and_errors(tmp_path):
    t = Tree(reversed(NODES))  # children may come before their parents in the file
    assert t.roots == ["t3", "t2", "t1"] and t.leaves("t2") == ["p4#1", "p3#2", "p3#1"]  # file order is reading order
    t = Tree(NODES)
    assert t.leaves("t2") == ["p3#1", "p3#2", "p4#1"] and t.leaf("p3#2") and not t.leaf("p3")
    t.save(tmp_path / "t.jsonl")
    assert Tree.load(tmp_path / "t.jsonl").nodes == t.nodes
    with pytest.raises(ValueError, match="duplicate"):
        Tree(NODES + [NODES[0]])
    with pytest.raises(ValueError, match="unknown parent"):
        Tree([{"id": "a", "parent": "x", "text": "a"}])
    with pytest.raises(ValueError, match="cycle"):
        Tree([{"id": "a", "parent": "b", "text": "a"}, {"id": "b", "parent": "a", "text": "b"}])


def test_navigate_passages_per_page_by_rank():
    nodes = [{"id": "root", "parent": None, "kind": "topic", "text": "All"}]
    for g, w in enumerate((1, 9, 5)):
        nodes.append({"id": f"p{g}", "parent": "root", "kind": "page", "text": f"page {w}"})
        nodes += [{"id": f"p{g}#{i}", "parent": f"p{g}", "kind": "passage", "text": f"text {i}"} for i in range(4)]
    c = Client(lambda t: float(t.split()[-1]) if t.startswith("page") else 1.0)
    nav = Navigator(c, Tree(nodes), levels=[{"keep": 1}, {"keep": 3, "min_p": 0}],
                    final={"top": 0, "per_page": [[1, 0], [2, 2]]})
    nav.search("q", pin=["p0#3"])
    keys = [x["key"] for x in c.bodies[-1]["questions"]["passage"]["candidates"]]
    # p1 ranked first: all 4; p2 second: its first 2; p0 third: none, but its pinned passage p0#3 still comes
    assert keys == ["p0#3", "p1#0", "p1#1", "p1#2", "p1#3", "p2#0", "p2#1"]


def test_navigate_pins_kept_at_their_level():
    """A pinned topic is kept and its pages compete within the page budget (not all its passages); a pinned page under
    a pruned topic still comes up at the page level."""
    c = Client(passport)
    nav = Navigator(c, Tree(NODES), levels=[{"keep": 1}, {"keep": 1}], final={"top": 0})
    r = nav.search("passport", pin=["t1", "p5"])
    assert r["stages"][0]["kept"] == ["t1", "t2"] and r["stages"][1]["kept"] == ["p3", "p5"]
    assert [x["key"] for x in c.bodies[1]["questions"]["page"]["candidates"]] == ["p1", "p2", "p3", "p4", "p5"]
    assert [x["key"] for x in c.bodies[-1]["questions"]["passage"]["candidates"]] == ["p3#1", "p3#2", "p5#1"]


def _items_tree():
    nodes = [{"id": f"t{t}", "parent": None, "kind": "topic", "text": f"Topic {t}", "title": f"Topic {t}"}
             for t in range(3)]
    for i in range(230):
        nodes += [{"id": f"p{i}", "parent": f"t{i % 3}", "kind": "page", "text": f"Page about item {i * 7 % 101}"},
                  {"id": f"p{i}#1", "parent": f"p{i}", "kind": "passage", "text": f"Item {i} costs {i % 13}."}]
    return Tree(nodes)


def _prompt_hash(prompt, ids):
    import hashlib
    p = int(hashlib.sha256("".join(map(chr, prompt)).encode()).hexdigest()[:8], 16) / 16 ** 8
    return [p, 1 - p][: len(ids)]


def test_navigate_split_at_off_by_default_one_request_per_level():
    """Default (split_at 0): every level goes out as one request whatever its size, so a server refuses a level above
    its --rank-max-candidates; split_at at or above a level's size sends the very same requests."""
    import json

    from basal.prompt import RequestError
    tree, runs = _items_tree(), []
    for kw in ({}, {"split_at": 0}, {"split_at": 230}, {"split_at": 10_000}):
        c = Client(lambda t: 1.0)
        c.srv.pos_probs = _prompt_hash
        r = Navigator(c, tree, levels=[{"keep": 3}, {"keep": 5}], final={"top": 0}, **kw).search("Which item costs 4?")
        runs.append((json.dumps(c.bodies), json.dumps(r["passages"]), [s["requests"] for s in r["stages"]]))
        assert [len(q["candidates"]) for b in c.bodies for q in b["questions"].values()] == [230, 5]
    assert all(x == runs[0] for x in runs)
    c = Client(lambda t: 1.0)
    c.srv.rank_max = 100
    with pytest.raises(RequestError, match="230 candidates, more than the 100 this server takes per request"):
        Navigator(c, tree, levels=[{"keep": 3}, {"keep": 5}], final={"top": 0}).search("Which item costs 4?")
    assert len(c.bodies) == 1
    c = Client(lambda t: 1.0)
    c.srv.rank_max = 100
    r = Navigator(c, tree, levels=[{"keep": 3}, {"keep": 5}], final={"top": 0}, split_at=100).search("Which item costs 4?")
    assert r["passages"] and [len(q["candidates"]) for b in c.bodies for q in b["questions"].values()] == [77, 77, 76, 5]


def test_navigate_split_at_level_in_parts_with_the_same_decisions():
    """split_at (experimental): a level with more candidates goes out as several requests, cut where the server's
    requests of `chunk` end: each candidate reads the same prompt (here every probability depends on the whole prompt),
    so the ranking is the one of a single request. A last stage that does not fit is refused, client side."""
    tree, runs = _items_tree(), []
    for split_at in (0, 120):
        c = Client(lambda t: 1.0)
        c.srv.pos_probs = _prompt_hash
        nav = Navigator(c, tree, levels=[{"keep": 3}, {"keep": 5}], final={"top": 0}, split_at=split_at)
        r = nav.search("Which item costs 4?", k=5)
        runs.append((r["passages"], [s["kept"] for s in r["stages"]],
                     [len(q["candidates"]) for b in c.bodies for q in b["questions"].values()]))
    assert runs[0][:2] == runs[1][:2]
    # 77 pages per topic (group): chunks of 50 + 27, 50 + 27, 50 + 26; parts of at most 120 candidates
    assert runs[0][2] == [230, 5] and runs[1][2] == [77, 77, 76, 5]
    nav = Navigator(Client(lambda t: 1.0), tree, levels=[{"keep": 3}, {"keep": 200}], final={"top": 0}, split_at=120)
    with pytest.raises(ValueError, match="last stage has 200 passages, more than split_at"):
        nav.search("Which item costs 4?")


def test_navigate_search_cli_settings_for_large_collections(tmp_path, monkeypatch, capsys):
    """basal-navigate search takes every setting of a large collection: levels, the last stage's per_page and
    multiply, and the experimental --split-at (off by default: one request per level)."""
    import json

    import basal.client
    clients = []

    class Stub(Client):
        def __init__(self, *_, **__):
            super().__init__(passport)
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False
    monkeypatch.setattr(basal.client, "Basal", Stub)
    tree = tmp_path / "tree.jsonl"
    Tree(NODES).save(tree)
    args = ["search", str(tree), "How do I renew my passport?", "--levels", '[{"keep": 2}, {"keep": 3}]', "--per-page",
            "[[1, 0], [2, 1]]", "--multiply", "--top", "0", "--final-chunk", "1", "--chunk", "1", "--json"]
    for extra, sizes in (([], [3]), (["--split-at", "2"], [2, 1])):
        clients.clear()
        navigate.main(args + extra)
        r = json.loads(capsys.readouterr().out)
        assert r["passages"] and r["stages"][0]["requests"] == 3  # 3 topics, the server's requests of 1
        assert [len(q["candidates"]) for b in clients[0].bodies[: len(sizes)] for q in b["questions"].values()][
            : len(sizes)] == sizes
        assert "prior" in clients[0].bodies[-1]["questions"]["passage"]
    assert navigate.parser().parse_args(["search", "t.jsonl", "q"]).split_at == 0
    sub = navigate.parser()._subparsers._group_actions[0].choices["search"]
    assert next(x for x in sub._actions if x.dest == "split_at").help.startswith("experimental")
