"""The identity harness itself on CPU: synthetic items, both orders, the comparison statistics."""
import pytest

from basal import identity
from tests.test_prompt2 import StubTok


def test_compare_statistics():
    ref = [[0.9, 0.1], [0.505, 0.495], [0.2, 0.8]]
    got = [[0.89, 0.11], [0.49, 0.51], [0.2, 0.8]]
    r = identity.compare(ref, got)
    assert r["readouts"] == 3 and r["disagreements"] == 1 and r["near_ties"] == 1
    assert abs(r["agreement"] - 2 / 3) < 1e-12 and abs(r["max_dp"] - 0.015) < 1e-9


@pytest.mark.parametrize("fmt", [1, 2])
def test_build_items(fmt):
    single, multi, long = identity.build(StubTok(), fmt, 3, 2, 1)
    assert len(single) == 3 and len(multi) == 2 and len(long) == 1
    toks, ids = multi[0]
    assert len(toks) == 10 and len(ids) == 10  # five questions x two orders over one state
    assert [len(x) for x in ids[::2]] == [4, 2, 3, 6, 2]
    assert toks[0][:50] == toks[2][:50]  # same state first
    assert len(long[0][0][0]) > 2 * len(single[0][0][0])


def test_loadtest_bodies():
    from basal import synth
    from basal.loadtest import body
    it = synth.items(1, 300, seed=1)[0]
    one = body(it)
    assert list(one["questions"]) == ["q0"] and len(one["questions"]["q0"]["criteria"]) == len(it["options"])
    five = body(it, questions=5)["questions"]
    assert len(five) == 5 and all(q["option_keys"] == "hide" for q in five.values())


def test_decisions_average_both_orders_in_canonical_order():
    # item 1: order 1 [0.7, 0.2, 0.1], reversed [0.3, 0.2, 0.5] -> canonical [0.5, 0.2, 0.3] averaged with order 1
    d = identity.decisions([[0.7, 0.2, 0.1], [0.3, 0.2, 0.5]])
    assert [round(x, 6) for x in d[0]] == [0.6, 0.2, 0.2]


@pytest.mark.parametrize("gates, status", [(["pass", "pass"], 0), (["pass", "FAIL"], 1)])
def test_bench_identity_exit_status(monkeypatch, gates, status):
    """basal-bench --identity exits 0 when every path passes its gate, 1 otherwise (not with the list of rows)."""
    import sys

    from basal import bench
    seen = []
    monkeypatch.setattr(identity, "main", lambda argv: seen.append(argv) or [{"gate": g} for g in gates])
    monkeypatch.setattr(sys, "argv", ["basal-bench", "--identity", "--model", "m"])
    assert bench.main() == status and seen == [["--model", "m"]]


@pytest.mark.parametrize("given", [False, True])
def test_identity_nvfp4_reads_the_model_folders_amax(monkeypatch, tmp_path, given):
    """--quant nvfp4: the engine under test gets the calibrated input amax, from --nvfp4-amax or the model folder's
    NVFP4_AMAX.json as basal-serve reads it (basal-omni refuses NVFP4 without it)."""
    import json

    (tmp_path / "NVFP4_AMAX.json").write_text(json.dumps({"amax": {"0.q_proj": 2.5}}), encoding="utf-8")
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"0.q_proj": 7.0}), encoding="utf-8")
    seen = []

    class Built(Exception):
        pass

    def backend(*args, **kw):
        seen.append(kw)
        raise Built

    monkeypatch.setattr(identity, "GraphBackend", backend)
    with pytest.raises(Built):
        identity.main(["--model", str(tmp_path), "--quant", "nvfp4"] + (["--nvfp4-amax", str(other)] if given else []))
    assert seen[0]["nvfp4_amax"] == ({"0.q_proj": 7.0} if given else {"0.q_proj": 2.5})
