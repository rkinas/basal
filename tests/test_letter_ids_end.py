"""The option codes of a long prompt are read at its end (answer_end), with the ids the whole prompt gives: a server's
first request with a long state no longer tokenizes the whole prompt once per option code (255 with prompt format 2),
which took a minute or more on a 1 MB state with the event loop blocked."""
from test_server import Tok

from basal.prompt import FORMATS, LETTER_TAIL, answer_end, letter_ids, render


class Counting(Tok):
    def __init__(self):
        self.chars = 0

    def __call__(self, text, add_special_tokens=False):
        self.chars += sum(map(len, text)) if isinstance(text, list) else len(text)
        return super().__call__(text, add_special_tokens)


def test_the_codes_of_a_long_prompt_are_read_at_its_end():
    tok = Counting()
    state = "Faktura 1240 z 15 marca 2024 r.\n" * 30000  # ~1 MB
    for fmt in (1, 2):
        prompt = render(tok, state, "Która?", ["a", "b", "c"], "pl", fmt)
        end = answer_end(tok, prompt)
        assert len(end) < 3 * LETTER_TAIL and prompt.endswith(end) and end.startswith("\n")
        k = len(FORMATS[fmt]["codes"]) if fmt == 1 else 26  # (the stand-in tokenizer: one id per character)
        tok.chars = 0
        assert letter_ids(tok, prompt, k, fmt) == [ord(c) for c in FORMATS[fmt]["codes"][:k]]
        assert tok.chars < 3 * len(prompt)  # (was k + 1 times the whole prompt)


def test_a_short_prompt_is_read_whole():
    tok = Tok()
    short = render(tok, "x", "Która?", ["a", "b"], "pl", 1)
    assert answer_end(tok, short) is short
