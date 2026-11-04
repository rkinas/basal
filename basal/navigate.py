"""basal-navigate (experimental, 2.5): retrieval by navigating a document tree with typed decisions, no index.

The corpus is a tree: topics, pages (and sections) down to passages. At each level the candidates are judged together:
one shared state holds the query and every candidate's description, and one yes / no question per candidate asks
whether its branch is worth opening (a `rank` question, basal/rank.py: one request to basal-serve per level, split into
requests of about 50 candidates when a level is larger). The best branches within the level's budget are opened, the
others pruned. The passages reached are judged the same way in one shared context, grouped under their pages in reading
order, and a final choice question reorders the top 20. Every request reads its state once, and the final choice
reads the same state again, which basal-serve --state-cache-gb keeps.

  basal-navigate build docs/ -o tree.jsonl                 (basal/navtree.py; Markdown, HTML, text or JSONL documents)
  basal-navigate search tree.jsonl "How do I renew my licence?" --url http://127.0.0.1:8000

  from basal.client import Basal
  from basal.navigate import Navigator, Tree
  nav = Navigator(Basal("http://127.0.0.1:8000", timeout=600), Tree.load("tree.jsonl"), levels=[{"keep": 10}])
  hits = nav.search("How do I renew my licence?")["passages"]       # [{"id", "score", "choice"?, "text"}, ...]

Tree file (JSONL, one node per line, children in reading order): {"id", "parent" (null at the top), "kind" ("topic",
"page", "section", "passage" or your own), "text" (what the model reads: a description, a title, the passage itself),
"title" (optional: the heading its children are listed under), "path" (optional: a passage's heading path)}.
A node without children is a passage.

Budgets (`levels`, one per depth, the last one repeats): "keep" (at most this many branches), "min_p" (only those
with at least this probability, always at least one), "sure" (only the top branch when its probability is at least
this), "chunk" (candidates per request). `pin`: node ids always kept at their level, also when a branch above them
was pruned (a conversation's recently used pages, or its help center); `extra`: passage ids added to the last stage
(e.g. a dense retriever's top passages). `final`: "top" (the choice shortlist, 0 = none), "chunk" (0 = one shared context when it fits the
model's context), "multiply" (a passage's score times its page's), "choice_context" (basal/rank.py), "per_page"
(passages per page by the page's rank, e.g. [[3, 0], [10, 10], [30, 3]]: every passage of the top 3 pages, the first
10 of pages 4-10, the first 3 of pages 11-30).

Very large collections (experimental, off by default): a level goes out as one request, which basal-serve refuses
above its --rank-max-candidates (10,000 by default). `split_at` (CLI --split-at N) sends a level with more than N
candidates as several requests of at most N, cut where the server's requests of `chunk` end, so every candidate reads
the same prompt as in one request; the engine batches the requests differently, so scores can differ slightly from a
one-request run. The last stage with a choice (top > 0) or one shared context (chunk 0) must still fit one request.
"""
import argparse
import json
import sys
import time

from .prompt import lang_of
from .rank import chunks

LABELS = {
    "en": {"topic": ("Topic", "Topics"), "page": ("Page", "Pages"), "section": ("Section", "Sections"),
           "passage": ("Passage", "Passages"), None: ("Item", "Items")},
    "pl": {"topic": ("Temat", "Tematy"), "page": ("Strona", "Strony"), "section": ("Sekcja", "Sekcje"),
           "passage": ("Fragment", "Fragmenty"), None: ("Pozycja", "Pozycje")},
}
WORDS = {  # the questions of each kind; "dialog": the query is a conversation and the agent replies next
    "en": {
        "question": {
            "topic": "Is the page with the information needed to answer the user's question likely in {candidate}?",
            "page": "Is the information needed to answer the user's question on {candidate}?",
            "section": "Is the information needed to answer the user's question in {candidate}?",
            "passage": "Does {candidate} contain information needed to answer the user's question?",
            "choice": "Which passage contains the information needed to answer the user's question?"},
        "dialog": {
            "topic": "Is the page with the information the agent needs for its next reply likely in {candidate}?",
            "page": "Is the information the agent needs for its next reply on {candidate}?",
            "section": "Is the information the agent needs for its next reply in {candidate}?",
            "passage": "Should the agent's next reply be based on {candidate}?",
            "choice": "Which passage should the agent's next reply be based on?"}},
    "pl": {
        "question": {
            "topic": "Czy {candidate} obejmuje stronę z informacją potrzebną do odpowiedzi na pytanie użytkownika?",
            "page": "Czy {candidate} zawiera informację potrzebną do odpowiedzi na pytanie użytkownika?",
            "section": "Czy {candidate} zawiera informację potrzebną do odpowiedzi na pytanie użytkownika?",
            "passage": "Czy {candidate} zawiera informację potrzebną do odpowiedzi na pytanie użytkownika?",
            "choice": "Który fragment zawiera informację potrzebną do odpowiedzi na pytanie użytkownika?"},
        "dialog": {
            "topic": "Czy {candidate} obejmuje stronę z informacją, której agent potrzebuje do następnej odpowiedzi?",
            "page": "Czy {candidate} zawiera informację, której agent potrzebuje do następnej odpowiedzi?",
            "section": "Czy {candidate} zawiera informację, której agent potrzebuje do następnej odpowiedzi?",
            "passage": "Czy {candidate} powinien być podstawą następnej odpowiedzi agenta?",
            "choice": "Na którym fragmencie powinna się opierać następna odpowiedź agenta?"}},
}
QUERY = {"en": ("User's question", "Dialogue (the latest turn last)", {"user": "User", "agent": "Agent"}),
         "pl": ("Pytanie użytkownika", "Rozmowa (ostatnia wypowiedź na końcu)", {"user": "Użytkownik", "agent": "Agent"})}
LEVEL = {"keep": 10}
FINAL = {"top": 20, "chunk": 0, "multiply": False, "choice_context": "auto"}
TITLE_CHARS = 200


class Tree:
    """The corpus tree: nodes by id, children in reading order."""

    def __init__(self, nodes):
        self.nodes, self.children, self.pos = {}, {None: []}, {}
        nodes = list(nodes)
        for n, x in enumerate(nodes):
            if not isinstance(x, dict) or not isinstance(x.get("id"), str) or not isinstance(x.get("text"), str):
                raise ValueError(f"node {n}: needs a string id and a string text")
            if x["id"] in self.nodes:
                raise ValueError(f"node {n}: duplicate id {x['id']!r}")
            self.nodes[x["id"]] = x
        for x in nodes:
            p = x.get("parent")
            if p is not None and p not in self.nodes:
                raise ValueError(f"node {x['id']!r}: unknown parent {p!r}")
            self.children.setdefault(p, []).append(x["id"])
        seen, stack = 0, [(i, 0) for i in reversed(self.children[None])]
        while stack:  # reading order: depth first, children in file order
            i, _ = stack.pop()
            self.pos[i] = seen
            seen += 1
            stack += [(c, 0) for c in reversed(self.children.get(i, ()))]
        if seen != len(self.nodes):
            raise ValueError("the tree has a cycle (a node is its own ancestor)")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.loads(line) for line in f if line.strip())

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for i in sorted(self.nodes, key=self.pos.get):
                f.write(json.dumps(self.nodes[i], ensure_ascii=False) + "\n")

    @property
    def roots(self):
        return self.children[None]

    def leaf(self, i):
        return not self.children.get(i)

    def leaves(self, i):
        """The passages under node i (or i itself), in reading order."""
        out, stack = [], [i]
        while stack:
            j = stack.pop()
            kids = self.children.get(j)
            if kids:
                stack += reversed(kids)
            else:
                out.append(j)
        return out

    def title(self, i):
        x = self.nodes[i]
        return (x.get("title") or x["text"])[:TITLE_CHARS]

    def group(self, i):
        """The heading a node is listed under: its parent's title, from the nearest page down (None at the top)."""
        parts, p = [], self.nodes[i].get("parent")
        while p is not None:
            parts.append(self.title(p))
            if self.nodes[p].get("kind") == "page":
                break
            p = self.nodes[p].get("parent")
        return " > ".join(reversed(parts)) or None

    def shown(self, i):
        x = self.nodes[i]
        return f"[{x['path']}] {x['text']}" if x.get("path") else x["text"]


def select(ranking, budget):
    """The kept (id, score) of a level's ranking (best first) under a budget {"keep", "min_p", "sure"}."""
    out = ranking[: budget.get("keep", LEVEL["keep"])]
    if budget.get("min_p") is not None:
        out = [x for x in out if x[1] >= budget["min_p"]] or ranking[:1]
    if budget.get("sure") is not None and ranking and ranking[0][1] >= budget["sure"]:
        out = ranking[:1]
    return out


def query_state(query, lang, turns=6):
    """The query as the state's head: a question, or the last `turns` turns of a conversation."""
    head, conv, roles = QUERY[lang]
    if isinstance(query, str):
        return f"{head}: {query}"
    lines = []
    for t in list(query)[-turns:]:
        if isinstance(t, str):
            lines.append(t)
        else:
            role = roles.get(t.get("role"), roles["agent"] if t.get("role") == "assistant" else t.get("role", ""))
            lines.append(f"{role}: {t['text']}")
    return conv + ":\n" + "\n".join(lines)


def _text_of(query):
    return query if isinstance(query, str) else " ".join(t if isinstance(t, str) else t.get("text", "") for t in query)


class Navigator:
    """Search a Tree through a basal-serve (client: basal.client.Basal, or any object with request(body) -> the
    response). levels: budgets per depth (the last one repeats); final: the last stage's settings; style "question" or
    "dialog" (default: dialog for a list of turns); lang "en" / "pl" (default: from the query); wording: your own
    question templates {kind: "... {candidate} ...", "choice": "..."}; order: "state_first" or "candidates_first";
    split_at (experimental, 0 = off): a level with more candidates goes out as several requests (set it to basal-serve's
    --rank-max-candidates)."""

    def __init__(self, client, tree, levels=None, final=None, style=None, lang=None, wording=None, chunk=50,
                 order="state_first", turns=6, split_at=0):
        self.client, self.tree, self.split_at = client, tree, split_at
        self.levels = [dict(b) for b in (levels or [LEVEL])]
        self.final = {**FINAL, **(final or {})}
        self.style, self.lang, self.wording, self.chunk, self.order, self.turns = style, lang, wording or {}, chunk, \
            order, turns

    def words(self, lang, style, kind):
        return self.wording.get(kind) or WORDS[lang][style].get(kind) or WORDS[lang][style]["section"]

    def label(self, lang, kind):
        return LABELS[lang].get(kind) or LABELS[lang][None]

    def ask(self, state, lang, style, groups, chunk, top=0, prior=None, context="auto"):
        """A rank question per kind of candidate, in one request (or several with split_at) -> ([(id, score, choice)]
        best first, the server's requests)."""
        qs = {}
        for kind, ids in groups.items():
            lab, title = self.label(lang, kind)
            cands = []
            for i in ids:
                g = self.tree.group(i)
                cands.append({"key": i, "text": self.tree.shown(i) if self.tree.leaf(i) else self.tree.nodes[i]["text"],
                              **({"group": g} if g else {})})
            q = {"type": "rank", "instructions": self.words(lang, style, kind), "candidates": cands, "label": lab,
                 "list_title": title, "chunk": chunk, "top": top, "order": self.order, "choice_context": context}
            if top:
                q["choice_instructions"] = self.words(lang, style, "choice")
            if prior:
                q["prior"] = {i: prior[i] for i in ids if i in prior}
            qs[kind or "item"] = q
        parts = self.parts(qs, chunk, top) if self.split_at else [qs]
        if len(parts) == 1:  # one request per level (the default)
            out = self.client.request({"state": state, "questions": qs, "lang": lang})
            ranking = sorted(((e["key"], e["score"], e.get("choice")) for a in out["answers"].values()
                              for e in a["ranking"]), key=lambda x: -x[1]) if len(qs) > 1 else \
                [(e["key"], e["score"], e.get("choice")) for a in out["answers"].values() for e in a["ranking"]]
            return ranking, out["usage"].get("requests", 1)
        rankings, n_req = {name: [] for name in qs}, 0
        for part in parts:
            out = self.client.request({"state": state, "questions": part, "lang": lang})
            n_req += out["usage"].get("requests", 1)
            for name, a in out["answers"].items():
                rankings[name] += [(e["key"], e["score"], e.get("choice")) for e in a["ranking"]]
        # stable: equal scores keep the candidates' order, as in one request
        rankings = {name: sorted(r, key=lambda x: -x[1]) for name, r in rankings.items()}
        ranking = [x for r in rankings.values() for x in r]
        return (sorted(ranking, key=lambda x: -x[1]) if len(qs) > 1 else ranking), n_req

    def parts(self, qs, chunk, top):
        """split_at: the rank questions of a level as requests of at most split_at candidates in all: whole requests
        of `chunk` (rank.chunks) per part, so each candidate is judged in the same request as without parts. A last
        stage (a choice, or one shared context) must fit one request."""
        n = sum(len(q["candidates"]) for q in qs.values())
        if n <= self.split_at:
            return [qs]
        if top or not chunk:
            raise ValueError(f"the last stage has {n} passages, more than split_at ({self.split_at}): lower keep or "
                             "per_page, or raise --split-at and basal-serve --rank-max-candidates")
        out, cur, size = [], {}, 0
        for name, q in qs.items():
            cands = q["candidates"]
            for idx in chunks([(c["key"], c["text"], c.get("group")) for c in cands], chunk):
                if size and size + len(idx) > self.split_at:
                    out.append(cur)
                    cur, size = {}, 0
                cur.setdefault(name, dict(q, candidates=[]))["candidates"] += [cands[i] for i in idx]
                size += len(idx)
        out.append(cur)
        for part in out:  # each part's prior names only its own candidates
            for q in part.values():
                if "prior" in q:
                    q["prior"] = {c["key"]: q["prior"][c["key"]] for c in q["candidates"] if c["key"] in q["prior"]}
        return out

    def depth(self, i):
        d, p = 0, self.tree.nodes[i].get("parent")
        while p is not None:
            d, p = d + 1, self.tree.nodes[p].get("parent")
        return d

    def cap(self, pool, rank_of, keep=()):
        """The passages of each judged page within its rank's allowance (final "per_page"); a page ranked beyond the
        last bound adds none; passages without a judged ancestor and pinned passages (keep) are kept."""
        out, used = [], {}
        for x in pool:
            if x in keep:
                out.append(x)
                continue
            p = self.tree.nodes[x].get("parent")
            while p is not None and p not in rank_of:
                p = self.tree.nodes[p].get("parent")
            if p is None:
                out.append(x)
                continue
            n = next((m for upto, m in self.final["per_page"] if rank_of[p] < upto), -1)
            if n == 0 or used.get(p, 0) < n:
                out.append(x)
                used[p] = used.get(p, 0) + 1
        return out

    def search(self, query, k=10, pin=(), extra=()):
        """The best passages for a query (a question, or a conversation: a list of turns {"role", "text"} or strings)
        -> {"passages": [{"id", "score", "choice"?, "text"}], "stages": [...], "latency_ms"}."""
        t0 = time.perf_counter()
        tree = self.tree
        lang = self.lang or lang_of(_text_of(query))
        style = self.style or ("question" if isinstance(query, str) else "dialog")
        state = query_state(query, lang, self.turns)
        pin = [p for p in pin if p in tree.nodes]
        pin_depth = {p: self.depth(p) for p in pin}
        frontier, pool, score, rank_of, stages, depth = list(tree.roots), [], {}, {}, [], 0
        seen = set()
        while True:
            pruned = [p for p in pin if pin_depth[p] == depth and p not in set(frontier)]  # a pruned branch's pin
            frontier = sorted(frontier + pruned, key=tree.pos.get) if pruned else frontier
            seen.update(frontier)
            pool += [i for i in frontier if tree.leaf(i)]
            inner = [i for i in frontier if not tree.leaf(i)]
            if not inner:
                break
            b = self.levels[min(depth, len(self.levels) - 1)]
            t1, n_req = time.perf_counter(), 0
            if len(inner) <= b.get("keep", LEVEL["keep"]) and b.get("min_p") is None and b.get("sure") is None:
                kept = inner  # the whole level fits the budget: no request
            else:
                groups = {}
                for i in inner:
                    groups.setdefault(tree.nodes[i].get("kind"), []).append(i)
                ranking, n_req = self.ask(state, lang, style, groups, b.get("chunk", self.chunk))
                score.update((i, s) for i, s, _ in ranking)
                rank_of.update((i, r) for r, (i, _, _) in enumerate(ranking))
                kept = [i for i, _ in select([(i, s) for i, s, _ in ranking], b)]
            kept = sorted(set(kept) | (set(pin) & set(inner)), key=tree.pos.get)
            stages.append({"depth": depth, "candidates": len(inner), "kept": kept, "requests": n_req,
                           "ms": round((time.perf_counter() - t1) * 1000, 2)})
            frontier = [c for i in kept for c in tree.children[i]]
            depth += 1
        if self.final.get("per_page"):  # passages per page by the page's rank: [[up to rank, at most n (0 = all)]]
            pool = self.cap(pool, rank_of, set(pin))
        reached = set(pool)
        for p in pin:  # a pin deeper than the search went: its passages join the last stage
            if p not in seen:
                pool += [x for x in tree.leaves(p) if x not in reached and not reached.add(x)]
        pool += [x for x in dict.fromkeys(extra) if x in tree.nodes and tree.leaf(x) and x not in reached
                 and not reached.add(x)]
        pool.sort(key=tree.pos.get)
        if not pool:
            return {"passages": [], "stages": stages, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
        f, prior = self.final, None
        if f["multiply"]:
            prior = {}
            for x in pool:
                p = tree.nodes[x].get("parent")
                while p is not None and p not in score:
                    p = tree.nodes[p].get("parent")
                prior[x] = score.get(p, 1.0)
        t1 = time.perf_counter()
        groups = {}
        for x in pool:
            groups.setdefault("passage", []).append(x)
        ranking, n_req = self.ask(state, lang, style, groups, f["chunk"], f["top"], prior, f["choice_context"])
        stages.append({"depth": depth, "candidates": len(pool), "kept": [], "requests": n_req,
                       "ms": round((time.perf_counter() - t1) * 1000, 2)})
        hits = []
        for i, s, c in ranking[:k]:
            hits.append({"id": i, "score": s, **({"choice": c} if c is not None else {}), "text": tree.nodes[i]["text"]})
        return {"passages": hits, "stages": stages, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


def parser():
    ap = argparse.ArgumentParser(prog="basal-navigate", description="Retrieval by navigating a document tree with "
                                 "typed decisions (experimental): build a tree, then search it through basal-serve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="a tree (JSONL) from Markdown, HTML, text or JSONL documents (basal/navtree.py)")
    b.add_argument("inputs", nargs="+", help="files or folders (.md, .markdown, .html, .htm, .txt, .jsonl)")
    b.add_argument("-o", "--out", required=True, help="the tree file to write (JSONL)")
    b.add_argument("--split-long", type=int, default=0, metavar="N",
                   help="a section level under pages with more than N passages (default: none)")
    b.add_argument("--group-dirs", action="store_true", help="folders become topics above their pages")
    b.add_argument("--clusters", type=int, default=0, metavar="K",
                   help="topics by k-means over the pages (needs basal[navigate]); 0 = none")
    b.add_argument("--max-children", type=int, default=0, metavar="M",
                   help="with --clusters: split a topic of more than M pages again (recursive k-means)")
    b.add_argument("--max-chars", type=int, default=2000, help="split longer passages (default 2000 characters)")
    b.add_argument("--min-chars", type=int, default=0, help="merge shorter passages under one heading (default off)")
    b.add_argument("--lang", choices=["en", "pl"], default=None, help="the language of generated descriptions")
    s = sub.add_parser("search", help="the best passages for a query (or a JSONL file of queries)")
    s.add_argument("tree", help="the tree file (JSONL)")
    s.add_argument("query", nargs="?", help="the question (or give --queries)")
    s.add_argument("--queries", help='JSONL: {"id", "query" | "turns", "pin"?, "extra"?} per line; writes JSONL')
    s.add_argument("--url", default="http://127.0.0.1:8000", help="basal-serve (default %(default)s)")
    s.add_argument("--api-key", default=None)
    s.add_argument("--timeout", type=float, default=600)
    s.add_argument("--k", type=int, default=10, help="passages returned (default 10)")
    s.add_argument("--keep", type=int, default=None, help="branches kept at every level (default 10)")
    s.add_argument("--levels", default=None, help='budgets per depth as JSON, e.g. \'[{"keep": 2, "sure": 0.8}, '
                   '{"keep": 10}]\'')
    s.add_argument("--top", type=int, default=FINAL["top"], help="the final choice shortlist (0 = none)")
    s.add_argument("--final-chunk", type=int, default=FINAL["chunk"],
                   help="passages per request in the last stage (0 = one shared context when it fits)")
    s.add_argument("--chunk", type=int, default=50, help="candidates per request at the other levels")
    s.add_argument("--split-at", type=int, default=0, metavar="N",
                   help="experimental: send a level with more than N candidates as several requests of at most N (set N "
                   "to basal-serve's --rank-max-candidates, 10000 by default); scores can differ slightly from one "
                   "request. Default: off (one request per level)")
    s.add_argument("--multiply", action="store_true", help="a passage's score times its page's")
    s.add_argument("--per-page", default=None, help="passages per page by the page's rank as JSON, e.g. "
                   "'[[3, 0], [10, 10], [30, 3]]' (every passage of the top 3 pages, 10 of pages 4-10, 3 of 11-30)")
    s.add_argument("--choice-context", choices=["auto", "screen", "shortlist"], default="auto")
    s.add_argument("--order", choices=["state_first", "candidates_first"], default="state_first")
    s.add_argument("--style", choices=["question", "dialog"], default=None)
    s.add_argument("--lang", choices=["en", "pl"], default=None)
    s.add_argument("--json", action="store_true", help="print the whole result as JSON")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.cmd == "build":
        from .navtree import build
        nodes = build(a.inputs, split_long=a.split_long, group_dirs=a.group_dirs, clusters=a.clusters,
                      max_children=a.max_children, max_chars=a.max_chars, min_chars=a.min_chars, lang=a.lang,
                      skip=[a.out])
        tree = Tree(nodes)
        tree.save(a.out)
        kinds = {}
        for x in tree.nodes.values():
            kinds[x.get("kind")] = kinds.get(x.get("kind"), 0) + 1
        print(f"{a.out}: " + ", ".join(f"{n} {k}" for k, n in kinds.items()), file=sys.stderr)
        return
    from .client import Basal
    if (a.query is None) == (a.queries is None):
        raise SystemExit("give a query or --queries")
    levels = json.loads(a.levels) if a.levels else [{"keep": a.keep or LEVEL["keep"]}]
    final = {"top": a.top, "chunk": a.final_chunk, "multiply": a.multiply, "choice_context": a.choice_context,
             **({"per_page": json.loads(a.per_page)} if a.per_page else {})}
    with Basal(a.url, timeout=a.timeout, api_key=a.api_key) as client:
        nav = Navigator(client, Tree.load(a.tree), levels, final, a.style, a.lang, chunk=a.chunk, order=a.order,
                        split_at=a.split_at)
        if a.query is not None:
            r = nav.search(a.query, k=a.k)
            if a.json:
                print(json.dumps(r, ensure_ascii=False, indent=1))
                return
            for n, h in enumerate(r["passages"], 1):
                print(f"{n:>3}  {h['score']:.3f}  {h['id']}  {h['text'][:100]}")
            print(f"{r['latency_ms']:.0f} ms, {sum(s['requests'] for s in r['stages'])} requests", file=sys.stderr)
            return
        with open(a.queries, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                q = json.loads(line)
                r = nav.search(q.get("turns") or q["query"], k=a.k, pin=q.get("pin", ()), extra=q.get("extra", ()))
                print(json.dumps({"id": q.get("id"), "passages": [h["id"] for h in r["passages"]],
                                  "scores": [h["score"] for h in r["passages"]], "latency_ms": r["latency_ms"],
                                  "stages": [{k: v for k, v in s.items() if k != "kept"} for s in r["stages"]]},
                                 ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
