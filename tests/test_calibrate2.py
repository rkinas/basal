"""basal-calibrate for prompt-format-2 models: calibration file 2 with choice temperatures per option-count bucket,
served back through basal-serve's loader (which uses the bucket before the type temperature)."""
import asyncio
import json
import re

import numpy as np
import pytest

import basal.server as server_module
from basal.calibrate import BUCKETS, buckets_cover, ece, fit, is_cert, main, served, table
from basal.server import Server, parser, temper, temperature
from tests.test_server import Tok

WEIGHT = re.compile(r"^[A-Za-z]{1,2}\. .*?~([0-9.e-]+)", re.M)  # format 2: the last option runs into the prefill


def rows(n, k, sharpen, prim="choice", seed=0):
    """Rows of an over-confident engine: it serves q ** sharpen while the gold answer is drawn from q."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        q = rng.dirichlet([0.3] * k)
        p = np.maximum(q ** sharpen, 1e-12)
        out.append(dict(prim=prim, p=list(p / p.sum()), gold=int(rng.choice(k, p=q)), group=f"{prim}{k}-{seed}-{i}"))
    return out


def test_bucket_temperatures_are_fitted_where_there_are_enough_decisions():
    data = rows(500, 4, 1.5) + rows(500, 20, 2.5, seed=1) + rows(30, 100, 3.0, seed=2) + rows(400, 2, 2.0, "noul", 3)
    shipped = {"prompt": 2, "temperature_per_prim": {"choice": 1.1, "noul": 1.2, "score": 1.3},
               "temperature_per_bucket": {"choice": {"27-255": 1.7}}}
    cal, report = fit(data, shipped, [0.05], 50, fmt=2)
    assert cal["prompt"] == 2 and "version" not in cal
    b = cal["temperature_per_bucket"]["choice"]
    assert 1.2 < b["2-10"] < 1.9 and 2.0 < b["11-26"] < 3.1   # the true 1.5 and 2.5, each from its own bucket
    assert 1.7 < b["27-255"] < 3.0     # ~21 FIT decisions: the fit (true 3.0) blended with the shipped 1.7 by n / 50
    assert cal["temperature_source"]["choice 27-255"].startswith("blended")
    kept, _ = fit(data, shipped, [0.05], 50, fmt=2, min_blend=0)
    assert kept["temperature_per_bucket"]["choice"]["27-255"] == 1.7     # --min-blend 0: the shipped one is kept
    assert 1.6 < cal["temperature_per_prim"]["noul"] < 2.5
    assert set(report["buckets"]) == {name for name, _, _ in BUCKETS}
    one, _ = fit(data, {}, [0.05], 50, fmt=1)  # format 1: the 1.5 file, no buckets
    assert "temperature_per_bucket" not in one and "prompt" not in one


def test_table_shows_the_temperatures_as_served():
    """A format-2 file whose 2-10 bucket (1.861) differs from its choice type temperature (1.498): the bucket row shows
    the shipped 1.861, the type row says the type temperature is not served where buckets cover every option count,
    and a bucket the file lacks shows the type temperature that serves it."""
    data = rows(600, 4, 2.0) + rows(300, 20, 2.0, seed=1)
    full = {"prompt": 2, "temperature_per_prim": {"choice": 1.498},
            "temperature_per_bucket": {"choice": {"2-10": 1.8607, "11-26": 1.9, "27-255": 2.1}}}
    _, report = fit(data, full, [0.05], 50, fmt=2)
    b = report["buckets"]["2-10"]
    assert b["T_shipped"] == 1.8607 and not b["T_shipped_type"] and not b["T_type"]
    assert report["types"]["choice"]["not_served"] == dict(shipped=True, fitted=True)
    text = table(report)
    assert "1.498 (not served: buckets) -> " in text
    assert re.search(r"^  2-10 .* 1\.861 -> \d\.\d{3} ", text, re.M)
    part = {"prompt": 2, "temperature_per_prim": {"choice": 1.498}, "temperature_per_bucket": {"choice": {"2-10": 1.8607}}}
    _, report = fit(data, part, [0.05], 10 ** 9, fmt=2, min_blend=0)  # nothing fitted: the shipped file as it is
    assert report["types"]["choice"]["not_served"] == dict(shipped=False, fitted=False)
    b = report["buckets"]["11-26"]
    assert b["T_shipped"] == 1.498 and b["T_shipped_type"] and b["T"] == 1.498 and b["T_type"]
    assert re.search(r"^  11-26 .* 1\.498 \(type\) -> 1\.498 \(type\)", table(report), re.M)
    assert "not served" not in table(report)
    assert buckets_cover({"choice": {"2-10": 1, "11-26": 1, "27-255": 1}})
    assert not buckets_cover({"choice": {"2-10": 1, "27-255": 1}}) and not buckets_cover({})


def test_type_ece_uses_the_bucket_temperature_as_served():
    """The choice row of the report measures ECE with the temperatures basal-serve applies: the option-count bucket
    temperature before the type one (here the only bucket, so the type row equals the bucket row)."""
    data = rows(600, 4, 2.0)
    shipped = {"prompt": 2, "temperature_per_prim": {"choice": 1.0},
               "temperature_per_bucket": {"choice": {"2-10": 2.0}}}
    cal, report = fit(data, shipped, [0.05], 10 ** 9, fmt=2, min_blend=0)  # shipped temperatures kept, not fitted
    t, b = report["types"]["choice"], report["buckets"]["2-10"]
    assert cal["temperature_per_bucket"]["choice"]["2-10"] == 2.0
    assert t["ece_shipped"] == b["ece_shipped"] and t["ece"] == b["ece"]
    cert = [r for r in data if is_cert(r["group"])]
    assert t["ece_shipped"] != ece(served(cert, {"choice": 1.0}))   # the type temperature alone is not what is served


class StubBackend:
    """The position probabilities are the weights written at the end of each option ("... ~0.7")."""
    policies = {}

    def __init__(self, *_, **__):
        self.tok = Tok()

    def run_shared(self, groups, policy=None):
        out = []
        for prompts, lids in groups:
            res = []
            for p, ids in zip(prompts, lids):
                w = [float(x) for x in WEIGHT.findall("".join(map(chr, p)))][: len(ids)]
                res.append([x / sum(w) for x in w])
            out.append(res)
        return out


@pytest.fixture
def engine2(tmp_path, monkeypatch):
    md = tmp_path / "model"
    md.mkdir()
    (md / "basal.json").write_text(json.dumps({"name": "stub", "prompt": 2}), encoding="utf-8")
    (md / "CALIBRATION.json").write_text(json.dumps({"prompt": 2, "temperature_per_prim": {"choice": 1.0}}), encoding="utf-8")
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: md)
    monkeypatch.setattr(server_module, "EagerBackend", StubBackend)
    return md


def labelled(path, n=1200, seed=0):
    """Choice requests with 3 options (sharpened 1.5x) and 15 options (2.5x), and yes/no questions."""
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            k, sharpen = ((3, 1.5), (15, 2.5), (2, 2.0))[i % 3]
            q = rng.dirichlet([0.3] * k)
            p = np.maximum(q ** sharpen, 1e-12)
            p /= p.sum()
            g = int(rng.choice(k, p=q))
            w = [f"~{x:.12g}" for x in p]
            if k == 2:
                question = dict(type="noul", criteria={"true": f"Yes {w[0]}", "false": f"No {w[1]}"},
                                gold="yes" if g == 0 else "no")
            else:
                question = dict(type="choice", criteria={f"k{j}": f"option {j} {w[j]}" for j in range(k)},
                                option_keys="hide", gold=f"k{g}")
            f.write(json.dumps({"id": f"c{i}", "state": f"Case {i}.",
                                "questions": {"q": dict(question, instructions="Which one?")}}) + "\n")


def test_end_to_end_writes_file_2_and_the_server_applies_the_buckets(tmp_path, engine2, capsys):
    labelled(tmp_path / "l.jsonl")
    out = tmp_path / "CALIBRATION.custom.json"
    main([str(tmp_path / "l.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(out)])
    assert "calibration file 2, engine eager" in capsys.readouterr().out
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert cal["prompt"] == 2 and cal["engine"] == "eager" and cal["fitted_on"]["orders"] == 1  # format 2: one order
    b = cal["temperature_per_bucket"]["choice"]
    assert 1.1 < b["2-10"] < 2.0 and 1.9 < b["11-26"] < 3.2 and "27-255" not in b
    srv = Server(parser().parse_args(["--mode", "eager", "--model", "stub", "--calibration", str(out)]))
    assert srv.cal == cal and srv.fmt == 2
    assert temperature(srv.cal, "choice", 15) == b["11-26"] and temperature(srv.cal, "choice", 40) == \
        cal["temperature_per_prim"]["choice"]

    async def decide(body):
        w = asyncio.get_running_loop().create_task(srv.worker())
        try:
            return await srv.decide(body)
        finally:
            w.cancel()
    weights = [0.5] + [0.5 / 14] * 14
    q = {"type": "choice", "instructions": "Which?", "option_keys": "hide",
         "criteria": {f"k{j}": f"option {j} ~{x}" for j, x in enumerate(weights)}}
    ans = asyncio.run(decide({"state": "x", "questions": {"q": q}}))["answers"]["q"]
    want = temper(server_module.torch.tensor(weights), b["11-26"]).tolist()
    assert max(abs(ans["probabilities"][f"k{j}"] - x) for j, x in enumerate(want)) < 1e-6


def test_cert_split_is_unchanged():
    assert sum(is_cert(f"g{i}") for i in range(1000)) in range(250, 350)
