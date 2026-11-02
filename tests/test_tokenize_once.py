"""Tokenize once (basal/prompt.py tokens_once, Server.encode): the prompts of a request share the template and the
state, so their ids are the shared part's ids once plus each prompt's own tail. Small tokenizers of the three kinds
the check knows (a SentencePiece-style BPE like Gemma 4, a Metaspace BPE like Bielik v3, a byte-level BPE with
Qwen3.5's pattern): every result equals whole-prompt tokenization, and anything else falls back to it."""
import string

from tokenizers import AddedToken, Regex, Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from basal.prompt import ONCE_SPLIT, once_ids, render, tokens_once

from test_server import FakeServer, Tok

CHARS = string.ascii_letters + string.digits + string.punctuation + "\n\t▁ąęłńóśźżŁ"
TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")


def fast(t):
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>")
    tok.add_special_tokens({"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]})
    tok.chat_template = TEMPLATE
    return tok


def bpe(extra=(), merges=()):
    """Characters, a few merges (words with their space, newline runs) and `extra` tokens of whole strings."""
    vocab = {"<unk>": 0}
    for c in CHARS:
        vocab.setdefault(c, len(vocab))
    pairs = [("▁", "t"), ("h", "e"), ("▁t", "he"), ("\n", "\n"), ("e", "s"), ("t", "es"), ("▁", "a")]
    pairs += list(merges)
    for a, b in pairs:
        vocab.setdefault(a, len(vocab))
        vocab.setdefault(b, len(vocab))
        vocab.setdefault(a + b, len(vocab))
    for x in extra:
        vocab.setdefault(x, len(vocab))
    return models.BPE(vocab=vocab, merges=pairs, unk_token="<unk>")


def gemma_like(merges=()):
    t = Tokenizer(bpe(merges=merges))
    t.normalizer = normalizers.Replace(" ", "▁")
    t.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
    return fast(t)


def bielik_like():
    t = Tokenizer(bpe())
    t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first", split=False)
    return fast(t)


def qwen_like():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    t = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    t.normalizer = normalizers.NFC()
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(ONCE_SPLIT), "isolated"),
                                               pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    return fast(t)


def whole(tok, prompts):
    return tok(prompts, add_special_tokens=False).input_ids


STATE = "Klient: zażółć\n\n  the test,\nPassage 1: tests  \nPassage 2: ▁ąę!\n \n"


def prompts_of(tok, state=STATE):
    return [render(tok, state, q, ["Yes", "No"], "en", 2) for q in ("Does Passage 1 test?", "Does Passage 2 test?",
                                                                     "Is the test   about \n tea?")]


def test_each_kind_gives_the_ids_of_whole_prompts():
    for tok in (gemma_like(), bielik_like(), qwen_like()):
        nl = once_ids(tok)
        assert nl == whole(tok, ["\n"])[0]
        for state in (STATE, "", "x", "\n", "the\n\n", " \n\t", "Łódź\nŁ", "a\nb\nc " * 40):
            ps = prompts_of(tok, state)
            assert tokens_once(tok, ps, nl) == whole(tok, ps)
        raw = ["head\nshared text, then\n\nQuestion: a", "head\nshared text, then\n\nQuestion: b?", "head\nshared t"]
        assert tokens_once(tok, raw, nl) == whole(tok, raw)


def test_falls_back_to_whole_prompts():
    tok = gemma_like()
    nl = once_ids(tok)
    assert tokens_once(tok, ["one prompt\nonly"], nl) is None
    assert tokens_once(tok, ["no newline a", "no newline b"], nl) is None
    assert tokens_once(tok, ["shared\nA", "shared\nB"], nl) is None  # they part right after the newline
    assert tokens_once(tok, ["shared\n a", "shared\n b"], nl) is None  # only whitespace after it
    assert tokens_once(tok, ["", "x"], nl) is None


def test_refused_tokenizers():
    assert once_ids(Tok()) is None  # no tokenizers backend
    assert once_ids(gemma_like(merges=[("\n", "T")])) is None  # a token "\nT": a newline before T is no boundary
    t = Tokenizer(models.WordPiece(vocab={"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    assert once_ids(PreTrainedTokenizerFast(tokenizer_object=t)) is None  # not BPE
    t = Tokenizer(bpe())
    t.pre_tokenizer = pre_tokenizers.Whitespace()  # a pre-tokenizer the check does not know
    assert once_ids(fast(t)) is None
    tok = gemma_like()
    tok.backend_tokenizer.add_tokens([AddedToken("<x>", lstrip=True)])
    assert once_ids(tok) is None  # an added token that takes the newline before it


def test_server_encode_serves_the_whole_prompt_ids():
    srv = FakeServer([0.5, 0.5])
    srv.tok, srv.fmt = gemma_like(), 2
    srv.once = once_ids(srv.tok)
    ps = prompts_of(srv.tok)
    branches = [(0, j, [[0, 1]], [p], [[1, 2]]) for j, p in enumerate(ps)]
    n, longest = srv.encode(branches)
    ids = whole(srv.tok, ps)
    assert [b[3][0] for b in branches] == ids and n == sum(map(len, ids)) and longest == [len(x) for x in ids]
