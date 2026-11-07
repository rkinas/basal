"""check_json (basal/request.py), the admission check every request body passes on the event loop: a body it takes
costs a walk without a location per element and memory that follows its nesting, not its breadth (the walk that named
a location for every element took 5.5 s and 45 times the body's size in memory on 32 MiB of [{}, ...], and held
/health that long). It refuses exactly what the naming walk refuses, with the same messages."""
import json
import math
import tracemalloc

import pytest
from test_http_fuzz_smoke import _cases

from basal.prompt import RequestError
from basal.request import MAX_DEPTH, _locate, _refused, check_json


@pytest.mark.parametrize("item", [b"{},", b"0,", b'"ab",', b'{"a": [1, "x"]},'])
def test_a_wide_body_is_checked_in_memory_that_does_not_grow_with_it(item):
    n = 2**21 // len(item)
    body = json.loads(b'{"state": [' + item * n + item[:-1] + b'], "questions": {"q": {"type": "noul"}}}')
    tracemalloc.start()
    try:
        check_json(body)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 64 * 1024, peak  # (the naming walk: 45 MiB for this 2 MiB body)


def deep(n, leaf=None):
    x = [] if leaf is None else [leaf]
    for _ in range(n - 1):
        x = [x]
    return x


EDGE = [{"questions": {"q": {"t": float("nan")}}}, {"state": float("nan")}, {"\ud800": 1}, ["\udead"], "\ud800", "é",
        float("inf"), 1e999, {"questions": [float("inf")]}, {"x": {"questions": float("nan")}}, [[]] * 3, {}, [],
        {"questions": {"q": {"criteria": ["a", -math.inf]}}}, {"questions": "\udfff"}, {"state": {"k\ud800": 1}},
        *(f(n) for n in (MAX_DEPTH - 1, MAX_DEPTH, MAX_DEPTH + 1, 300) for f in (
            deep, lambda n: {"questions": deep(n)}, lambda n: deep(n, "\ud800"), lambda n: {"questions": deep(n, 1e999)},
            lambda n: deep(n, {})))]


def bodies():
    yield from EDGE
    for _, data in _cases(3000, seed=11):
        try:
            yield json.loads(data)
        except (ValueError, RecursionError):
            pass


@pytest.mark.parametrize("finite", [("questions",), ()])
def test_it_refuses_what_the_naming_walk_refuses_and_with_its_message(finite):
    n = 0
    for body in bodies():
        try:
            _locate(body, MAX_DEPTH, finite)
            said = None
        except RequestError as e:
            said = str(e)
        assert _refused(body, MAX_DEPTH, finite) == (said is not None), (repr(body)[:200], said)
        try:
            check_json(body, finite=finite)
            got = None
        except RequestError as e:
            got = str(e)
        assert got == said
        n += 1
    assert n > 2000
