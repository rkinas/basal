"""basal-selfcheck: the shipped reference files hold the requests basal.synth builds, and the comparison counts
agreement outside near ties, |dp| and truncated requests."""
import json
from pathlib import Path

import pytest

from basal import selfcheck

EXAMPLES = sorted((Path(selfcheck.__file__).parent / "examples").glob("selfcheck_*.json"))


def test_shipped_references_hold_the_requests_synth_builds():
    assert EXAMPLES, "no shipped reference file"
    built = selfcheck.build_requests()
    for path in EXAMPLES:
        ref = json.loads(path.read_text(encoding="utf-8"))
        assert [(i["id"], i["state"], i["questions"]) for i in ref["items"]] == \
            [(r["id"], r["state"], r["questions"]) for r in built], path.name
        for it in ref["items"]:
            assert set(it["answers"]) == set(it["questions"])
            for a in it["answers"].values():
                assert abs(sum(a["probabilities"].values()) - 1) < 1e-4


def test_compare_counts_near_ties_as_agreement():
    ref = {"a": 0.5, "b": 0.49, "c": 0.01}
    assert selfcheck.compare(ref, {"a": 0.5, "b": 0.49, "c": 0.01})[:3] == (True, True, 0.0)
    top, tie, dp, _, n = selfcheck.compare(ref, {"a": 0.48, "b": 0.51, "c": 0.01})  # flipped, but within 0.02
    assert (top, tie, n) == (False, True, 3) and dp == pytest.approx(0.02)
    top, tie, dp, _, _ = selfcheck.compare({"a": 0.9, "b": 0.1}, {"a": 0.2, "b": 0.8})  # a real disagreement
    assert (top, tie) == (False, False) and dp == pytest.approx(0.7)


def serve(ref, reverse=False, truncated=(), fail=()):
    """A stand-in for selfcheck.call: the reference answers, with the probabilities dealt out in the opposite order."""
    by_state = {i["state"]: i for i in ref["items"]}

    def call(url, path, body=None, timeout=300):
        if path == "/v1/models":
            return {"models": [{"name": ref["model"], "mode": "llamacpp", "temperature_per_prim": {}}]}
        it = by_state[body["state"]]
        if it["id"] in fail:
            raise SystemExit("HTTP 500")
        answers = {}
        for k, a in it["answers"].items():
            p = a["probabilities"]
            p = dict(zip(p, list(p.values())[::-1])) if reverse else dict(p)
            answers[k] = {"type": a["type"], "probabilities": p}
        return {"answers": answers, "usage": {"truncated": it["id"] in truncated}}
    return call


@pytest.fixture
def ref(tmp_path):
    r = json.loads(EXAMPLES[0].read_text(encoding="utf-8"))
    r["items"] = r["items"][:6]
    (tmp_path / "ref.json").write_text(json.dumps(r), encoding="utf-8")
    return r, str(tmp_path / "ref.json")


def test_check_passes_on_the_reference_itself_and_fails_on_a_shifted_server(monkeypatch, capsys, ref):
    r, path = ref
    monkeypatch.setattr(selfcheck, "call", serve(r))
    assert selfcheck.check("http://x", path, 0.98, "") == 0
    out = capsys.readouterr().out
    assert "top-1        1.0000" in out and "max 0.0000" in out and "PASS" in out
    monkeypatch.setattr(selfcheck, "call", serve(r, reverse=True))
    assert selfcheck.check("http://x", path, 0.98, "") == 1
    assert "FAIL" in capsys.readouterr().out


def test_check_leaves_out_truncated_requests_and_fails_on_errors(monkeypatch, capsys, ref):
    r, path = ref
    ids = [i["id"] for i in r["items"]]
    monkeypatch.setattr(selfcheck, "call", serve(r, truncated=ids[:2]))
    assert selfcheck.check("http://x", path, 0.98, "") == 0
    assert "4 compared, 2 cut by the context" in capsys.readouterr().out
    monkeypatch.setattr(selfcheck, "call", serve(r, fail=ids[:1]))
    assert selfcheck.check("http://x", path, 0.98, "") == 1
    assert "1 failed" in capsys.readouterr().out


def test_reference_lookup_by_name():
    name = EXAMPLES[0].stem.removeprefix("selfcheck_")
    assert selfcheck.find(name) == EXAMPLES[0]
    with pytest.raises(SystemExit, match="shipped references"):
        selfcheck.find("no-such-model")


def test_a_server_with_bucket_temperatures_only_is_calibrated(monkeypatch):
    """A format-2 calibration can temper choice questions by option count alone: that server is not raw."""
    entry = {"name": "m", "mode": "eager", "temperature_per_prim": {}, "temperature_per_bucket": {"choice": {"2-10": 1.7}}}
    monkeypatch.setattr(selfcheck, "call", lambda *a, **k: {"models": [entry]})
    assert selfcheck.server_info("http://x")["uncalibrated"] is False
    with pytest.raises(SystemExit, match="--no-calibration"):
        selfcheck.make_reference("http://x", "/nonexistent/ref.json")
    assert selfcheck.uncalibrated({"temperature_per_prim": {}, "temperature_per_bucket": {}})
    assert selfcheck.uncalibrated({"temperature_per_prim": {"choice": 1.0}})               # all 1.0: raw
    assert not selfcheck.uncalibrated({"temperature_per_prim": {"noul": 1.3}})
