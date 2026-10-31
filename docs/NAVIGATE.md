# Navigate and rank (experimental, 2.5)

Two features for retrieval with typed decisions, after *Retriever-Free Retrieval: Navigating Documents with Decision
Models* (Wan, Jünger, Du, Liu, Krishnan and Weinberger, arXiv 2610.35868):

- the **`rank` question type**: one request ranks thousands of candidates, more than the 255 options of a choice
  question;
- **`basal-navigate`**: retrieval without an index. The corpus is a tree (topics, pages, sections, passages); at each
  level the model judges the candidates together and only the best branches are opened.

*Experimental* (see [FEATURES.md](FEATURES.md)): the API may change in a minor release, and the models were not trained
on these prompts. Check the quality on your own data.

## Rank thousands of candidates

```json
{"state": "User's question: how long does a visa take?",
 "questions": {"best": {"type": "rank",
   "instructions": "Does {candidate} contain information needed to answer the user's question?",
   "candidates": [{"key": "visa#3", "text": "Visas take two weeks.", "group": "Visa guide"},
                  {"key": "visa#4", "text": "Adults pay 60.", "group": "Visa guide"},
                  {"key": "tax#1", "text": "Pay your tax online.", "group": "Taxes"}],
   "label": "Passage", "list_title": "Passages", "top": 20}}}
```

1. **Screening.** The candidates are numbered in a list (`Passage 7: <text>`, under a `[<group>]` line) and packed, in
   their order, into requests of `chunk` candidates (default 50; a group stays in one request when it fits). Each
   request's state is your state plus its list, and every candidate in it is one yes / no question, `{candidate}` in
   the instructions naming it. A request reads its state once and judges its candidates side by side, which makes
   their probabilities comparable; the paper found this gain levels off at about 50 candidates per request. All
   requests go through the server's queue together, so they are batched.
2. **Final choice** (`top` >= 2, default 20): one choice question over the top candidates by P(yes), its options their
   names in the list. Its probabilities reorder them; the others keep their order. When one request held every
   candidate, the choice reads that request's state again: with `basal-serve --state-cache-gb N` the state is a cache
   hit and only the question is computed. Otherwise it reads your state with a list of only the shortlisted candidates.

| field | default | |
|---|---|---|
| `instructions` | (required) | the yes / no question; `{candidate}` names the candidate (`Passage 7`) |
| `candidates` | (required) | `{key: text}`, `[text, ...]` (keys `"0"`, `"1"`, ...) or `[{"key", "text", "group"}]`; at most 10,000 per request, all its rank questions together (`basal-serve --rank-max-candidates N`) |
| `label`, `list_title` | `Candidate`, none | how the list names its entries, and its heading line |
| `chunk` | 50 | candidates per request; 0 = all in one request when it fits the model's context, else 50 |
| `top` | 20 | the final choice's shortlist, at most the model's options per question; 0 = no choice |
| `choice_instructions` | `Which candidate is the best match?` | the final choice's question |
| `choice_context` | `auto` | `screen` (the screening state; needs one request), `shortlist`, or `auto` (screen when it can) |
| `order` | `state_first` | `candidates_first` puts the list before your state, so requests about the same candidates share a cached prefix (`--state-cache-gb`) across different states |
| `prior` | none | `{key: weight}`: a candidate's score is P(yes) x weight (e.g. its page's probability) |
| `limit` | all | ranking entries returned |

The answer: `{"type": "rank", "ranking": [{"key", "score", "choice"?, "yes"?}, ...], "top": "<key>", "candidates": n,
"requests": n, "experimental": true}`, best first; `choice` on the shortlisted entries, `yes` (P(yes) before the prior)
when a prior is given. `usage` sums all requests (`usage.requests`). Other questions of the same request are answered
as usual over your state. Not available with media, `ask_if`, `coverage`, `max_error` or `evidence`.

In Python: `Basal(url, timeout=600).rank(state, "Does {candidate} ...?", candidates, top=20)` (`basal.client`); a large
ranking takes many engine calls, so give the client a timeout to match. `GET /health` counts each request of a rank
question as in flight, and `--log-decisions` logs each of them.

**Known limits (2.5).**

- **Latency is seconds to a minute per query, not the paper's 0.7-1.8 s** (measured there with a hosted decision model
  and many requests in parallel). basal-omni-1.0-max on one H100: a median of 24.1 s per MultiDoc2Dial turn and
  64.0 s per ConditionalQA question at 8 concurrent clients ([Measured results](#measured-results)).
- **Cost is one judgment per candidate.** basal-omni-1.0-max (`--mode fast`, one H100) made 78-135 judgments per
  second over 2k-15k-token states. A ConditionalQA question (652 guides, then the paragraphs of the top 3) took 932
  judgments on average: 14 requests of guide titles and one request of 279 paragraphs on average, plus the choice.
- **CPU time per request.** The server tokenizes the part a request's prompts share (template, state and list) once
  and each question's own tail after it, for the tokenizers of the basal models (Gemma 4, Qwen3.5, Bielik v3); with
  other tokenizers it tokenizes every prompt whole. One request of 300 passages over a 12,514-token state: 0.08 s of
  CPU instead of 3.4 s on one thread (0.52 s with the tokenizer's threads), on an Apple M4 Pro.

## basal-navigate

```bash
basal-navigate build docs/ -o tree.jsonl                  # Markdown, HTML, text or JSONL documents
basal-serve --model Remek/basal-omni-1.0-max --state-cache-gb 16 &
basal-navigate search tree.jsonl "How long does a visa take?" --url http://127.0.0.1:8000
```

```python
from basal.client import Basal
from basal.navigate import Navigator, Tree

nav = Navigator(Basal("http://127.0.0.1:8000", timeout=600), Tree.load("tree.jsonl"),
                levels=[{"keep": 2, "sure": 0.8}, {"keep": 10}], final={"top": 20})
for hit in nav.search("How long does a visa take?", k=5)["passages"]:
    print(hit["score"], hit["id"], hit["text"][:80])
```

At each level, one `rank` request (yes / no only) judges the candidates under the branches kept so far: the state is
the query and every candidate's description, the question for a page "Is the information needed to answer the user's
question on Page n?". The best branches within the level's budget are opened, the others pruned for this query. The
passages reached are judged in one shared context, grouped under their pages in reading order, and a final choice
reorders the top 20. The number of sequential requests grows with the depth of the tree, not with the size of the
corpus.

- **Budgets** (`levels`, one per depth; the last one repeats): `keep` (at most this many branches, default 10),
  `min_p` (only branches with at least this probability, always at least one), `sure` (only the top branch when its
  probability is at least this), `chunk` (candidates per request). A level that fits its budget needs no request.
- **Last stage** (`final`): `top` (default 20; 0 = no choice), `chunk` (default 0: one shared context when it fits),
  `multiply` (a passage's score times its page's), `choice_context`, `per_page` (passages per page by the page's rank:
  `[[3, 0], [10, 10], [30, 3]]` takes every passage of the top 3 pages, the first 10 of pages 4-10 and the first 3 of
  pages 11-30).
- **`search(query, k, pin, extra)`**: `query` is a question, or a conversation (a list of `{"role", "text"}` turns; the
  last six are read and the questions ask about the agent's next reply). `pin`: node ids always kept at their level,
  also when a branch above them was pruned (e.g. the pages a conversation used recently and their help center).
  `extra`: passage ids added to the last stage, e.g. the top passages of a dense retriever, which recovers evidence
  that pruning lost.
- **Language**: English or Polish questions and labels, from the query (`lang` to set it); `wording` replaces the
  question of a kind (`{"page": "... {candidate} ...", "choice": "..."}`).
- The result: `{"passages": [{"id", "score", "choice"?, "text"}], "stages": [{"depth", "candidates", "kept",
  "requests", "ms"}], "latency_ms"}`. `basal-navigate search --queries q.jsonl` runs a file of queries (`{"id",
  "query" | "turns", "pin"?, "extra"?}` per line) and writes one JSON line per query.

### The tree

A JSONL file, one node per line, children in reading order: `{"id", "parent" (null at the top), "kind" ("topic",
"page", "section", "passage" or your own), "text" (what the model reads: a description, a title, the passage), "title"
(optional: the heading its children are listed under), "path" (optional: a passage's heading path)}`. A node without
children is a passage. Trees need not be uniform: a long page may have sections while a short one holds its passages
directly.

`basal-navigate build` makes one from documents:

- pages are files (or JSONL lines `{"id"?, "title"?, "text" | "markdown" | "html", "group"?}`); a page reads as its
  title and top headings; passages are paragraphs, list items and table rows (HTML: `p`, `li`, `tr`, `dt`, `dd`,
  `pre`, `blockquote`) with their heading path; longer than `--max-chars` (2,000) they are split at sentence ends;
- `--split-long N` adds a section per top heading under pages with more than N passages;
- `--group-dirs` makes folders (or JSONL `group`) topics above their pages;
- `--clusters K [--max-children M]` groups the pages into K topics by k-means over TF-IDF vectors of their title and
  first passage, splitting a topic of more than M pages again; a topic reads as its most central titles and top terms.
  It needs `pip install 'basal[navigate]'` (scikit-learn); in Python, `basal.navtree.cluster(texts, k, embed=fn)`
  takes your own embedding function. The default install needs nothing new.

Descriptions decide what can be found: a branch whose description does not show what it holds is pruned. Pruning is
for this query only, and a larger `keep` trades requests for recall.

## Measured results

*Experimental*: measured once, for 2.5, with basal-omni-1.0-max (`--mode fast --state-cache-gb 16`, one H100). The
model was not trained on these datasets or these prompts; the datasets were used for evaluation only. Confidence
intervals are paired bootstrap 95% intervals (10,000 resamples of whole dialogues on MultiDoc2Dial, of questions on
ConditionalQA). The dense baseline is Qwen3-Embedding-8B over the same passages, with a task instruction. The paper's
numbers come from another model, corpus build and sample and are given for context only.

**basal-navigate**, each query against the whole corpus:

| dataset | metric | basal-navigate (choice over 20, default) | yes / no only (`top` 0) | dense | basal-navigate - dense | paper |
|---|---|---|---|---|---|---|
| MultiDoc2Dial validation: 1,003 turns of 150 dialogues (seed 0); 4 help centres, 413 pages, 4,110 passages | R@1 | **44.27** | 41.18 | 34.70 | +9.57 [5.71, 13.36] | 47.4 |
| | R@10 | 79.96 | 79.26 | 80.86 | -0.90 [-4.20, +2.28] | 83.6 |
| ConditionalQA dev: 271 questions with evidence; 652 guides, 56,123 paragraphs | Hit@1 | **63.47** | 64.94 | 44.28 | +19.19 [11.81, 26.94] | 69.7 |
| | R@10 | 66.56 | 70.30 | 55.94 | +10.62 [6.23, 15.00] | 71.9 |

- **The final choice helped on MultiDoc2Dial and hurt on ConditionalQA.** Against yes / no only: MultiDoc2Dial R@1
  +3.09 [0.93, 5.19]; ConditionalQA Hit@1 -1.48 [-5.17, +2.21] and R@10 -3.74 [-6.08, -1.37]. It stays on by default
  (`top` 20). To switch it off: `"top": 0` on a `rank` question, `final={"top": 0}` in `Navigator`,
  `basal-navigate search --top 0`.
- Settings. MultiDoc2Dial: the conversation's last six turns; keep 2 help centres (only the top one when its
  probability is at least 0.8), then 10 pages, plus the help centre of the most recently retrieved page and the two
  most recent pages (`levels=[{"keep": 2, "sure": 0.8}, {"keep": 10}]`, `pin`); every passage of those pages (121 on
  average) in one shared context. ConditionalQA: the question and the user's situation; the 652 guide titles and top
  headings in 14 requests of 50, keep 3 guides, then all their paragraphs (279 on average, 98-760) in one shared
  context.
- Speed at 8 concurrent clients: MultiDoc2Dial 1,003 turns in 3,147 s (median 24.1 s, 95th percentile 35.1 s per
  turn); ConditionalQA 271 questions in 2,233 s (median 64.0 s, 95th percentile 98.8 s per question).

**`rank` over 1,000 candidates** (ConditionalQA, one pool per question: its evidence paragraphs plus paragraphs taken
alternately from the dense and the BM25 ranking, 1,000 in all, shuffled; one `rank` request each, 20 requests of 50
plus the final choice):

| order | Hit@1 | R@10 |
|---|---|---|
| BM25 | 24.72 | 32.92 |
| dense | 44.28 | 55.94 |
| `rank`, yes / no only (`top` 0) | 57.93 | 65.99 |
| `rank`, choice over the top 20 (default) | **65.31** | 69.19 |
| `rank`, choice over the top 255 | 62.36 | 57.81 |

Yes / no only against dense: Hit@1 +13.65 [6.64, 21.03]. The choice over 20 against yes / no only: Hit@1 +7.38 [1.48,
13.28], R@10 +3.20 [1.32, 5.22]. A `rank` request over 1,000 candidates took a median of 27.7 s at 4 concurrent
clients.

