"""A --calibration file fitted on another model (or revision, or adapter) keeps its temperatures only: its thresholds,
task thresholds, conformal levels and fitted fallback thresholds are not applied, every answer is marked unvalidated,
and GET /v1/models agrees. Identity compares whole repo ids when both sides are Hugging Face snapshots, and another
revision by its model files: a commit that changes only the card keeps the file. basal-calibrate --from-rows applies
basal-serve's rule for orders and layout to its baseline."""
import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from basal import server
from basal.calibrate import main
from basal.server import Server, calibration_mismatch, load_calibration, parser
from tests.test_calibrate import SHIPPED, StubBackend, engine, synthetic  # noqa: F401 (fixture)

CAL = {"prompt": 1, "temperature_per_prim": {"choice": 1.0, "noul": 1.0, "score": 1.0},
       "thresholds": {"0.05": {"confidence": 0.6, "per_type": {"noul": {"confidence": 0.6}}}},
       "task_thresholds": {"classify": {"0.05": 0.5}},
       "conformal": {"levels": {"0.9": {"per_type": {"noul": {"p_min": 0.2}}}}}}
NOUL = {"type": "noul", "instructions": "Covered?", "criteria": {"true": "Yes ~0.95", "false": "No ~0.05"},
        "max_error": 0.05}


def ask(srv, body):
    async def go():
        w = asyncio.get_running_loop().create_task(srv.worker())
        try:
            return await srv.decide(body)
        finally:
            w.cancel()
    return asyncio.run(go())["answers"]


def serve(path, model):
    return Server(parser().parse_args(["--mode", "eager", "--model", model, "--calibration", str(path)]))


def test_a_file_of_another_model_certifies_nothing(tmp_path, engine, capsys):  # noqa: F811
    f = tmp_path / "C.json"
    f.write_text(json.dumps(dict(CAL, fitted_on={"model": "Remek/basal-omni-1.0-max"})), encoding="utf-8")
    own = serve(f, "Remek/basal-omni-1.0-max")
    assert own.cal_mismatch is None and own.cal_unvalidated is None
    a = ask(own, {"state": "x", "questions": {"q": NOUL}})["q"]
    assert a["abstain"] is False and "calibration" not in a        # the same model: certified at 0.6
    cls = {"type": "classify", "labels": {"a": "Alpha ~0.9", "b": "Beta ~0.1"}, "max_error": 0.05}
    assert ask(own, {"state": "x", "questions": {"c": cls}})["c"]["automated"] == {"0.05": True}
    srv = serve(f, "Remek/basal-omni-1.1-max")
    assert "fitted on Remek/basal-omni-1.0-max" in srv.cal_mismatch
    assert "they are not applied" in capsys.readouterr().out
    assert srv.cal["temperature_per_prim"] == CAL["temperature_per_prim"]
    assert not {"thresholds", "task_thresholds", "conformal", "multi"} & set(srv.cal)
    assert server.certified_table(srv.cal) == {}                   # GET /v1/models and the answers agree: nothing
    assert ask(srv, {"state": "x", "questions": {"c": cls}})["c"]["automated"] == {"0.05": None}
    act = dict(NOUL, type="act", costs={"wrong": 2, "defer": 1})
    ans = ask(srv, {"state": "x", "questions": {"q": NOUL, "s": dict(NOUL, coverage=0.9), "a": act}})
    assert ans["q"]["abstain"] is True and ans["q"]["calibration"] == "unvalidated"
    assert ans["s"]["answer_set"] == ["true", "false"]             # no conformal level: every option
    assert ans["a"]["deferred"] is True and ans["a"]["calibration"] == "unvalidated"


def test_identity_compares_whole_repo_ids_of_snapshots(tmp_path):
    a = tmp_path / "models--OrgA--basal-x" / "snapshots" / "aaa"
    b = tmp_path / "models--OrgB--basal-x" / "snapshots" / "aaa"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    cal = {"fitted_on": {"model": "OrgA/basal-x", "revision": "aaa"}}
    assert calibration_mismatch(cal, "OrgA/basal-x", a) is None
    assert calibration_mismatch(cal, "orga/BASAL-X", a) is None                     # case does not matter
    assert "fitted on OrgA/basal-x, the served model is OrgB/basal-x" in calibration_mismatch(cal, "OrgB/basal-x", b)
    assert calibration_mismatch(cal, "/data/basal-x", tmp_path) is None             # a local copy: by its last part
    assert calibration_mismatch({"fitted_on": {"model": "/data/basal-x"}}, "OrgB/basal-x", b) is None


def test_load_calibration_and_the_shared_rule_agree(tmp_path):
    f = tmp_path / "C.json"
    for data, orders, layout, why in (({"orders": 2}, 1, "canonical", "fitted on 2 option order(s), served with 1"),
                                      ({"layout": "qfirst"}, 1, "canonical", "fitted on the qfirst layout"),
                                      ({"orders": 1}, 1, "canonical", None)):
        f.write_text(json.dumps(dict(CAL, **data)), encoding="utf-8")
        _, _, got = load_calibration(tmp_path, "eager", f, None, orders, layout=layout)
        assert got == server.unvalidated_reason(dict(CAL, **data), "C.json", orders, layout)
        assert (got is None) == (why is None) and (why is None or why in got)


@pytest.mark.parametrize("extra, why", [({"layout": "qfirst"}, "fitted on the qfirst layout"),
                                        ({"orders": 2}, "fitted on 2 option order(s), served with 1")])
def test_refit_drops_the_thresholds_of_a_baseline_basal_serve_would_not_apply(tmp_path, capsys, extra, why):
    base = tmp_path / "base.json"
    base.write_text(json.dumps(dict(CAL, **extra)), encoding="utf-8")
    head = dict(basal_calibrate_rows=1, fmt=1, kind="eager", quant=None, model="stub", engine="eager",
                data="labels.jsonl", requests=1, baseline=str(base), orders=1, layout="canonical")
    rows = [dict(prim="noul", group=f"g{i}", family="f", gold=0, p=[0.9, 0.1], line=i, name="q", keys=["true", "false"])
            for i in range(20)]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [head] + rows) + "\n", encoding="utf-8")
    main([str(path), "--from-rows", "--report-only", "--targets", "0.05"])
    out = capsys.readouterr()
    assert why in out.err and "not compared or kept" in out.err


# a model repository in a Hugging Face cache: revision R1, a commit that changes only the card (R2), one that changes
# the weights (R3) and one that changes the tokenizer (R4)
R1, R2, R3, R4 = "1" * 40, "2" * 40, "3" * 40, "4" * 40
FILES = {"config.json": b'{"layers": 2}', "tokenizer.json": b'{"vocab": 1}', "model.safetensors": b"weights 1",
         "README.md": b"card 1", "SHA256SUMS": b"sums 1", "CALIBRATION.json": json.dumps(SHIPPED).encode()}
CHANGES = {R1: {}, R2: {"README.md": b"card 2", "SHA256SUMS": b"sums 2", "basal.png": b"image", ".gitattributes": b"*"},
           R3: {"model.safetensors": b"weights 2"}, R4: {"tokenizer.json": b'{"vocab": 2}'}}


def snapshot(root, repo, rev, files):
    """A snapshot of `repo` at `rev` in the Hugging Face cache `root`: each file a link into blobs/, named by the sha256
    of its content (as the LFS files are)."""
    base = root / f"models--{repo.replace('/', '--')}"
    snap = base / "snapshots" / rev
    for name, content in files.items():
        blob = base / "blobs" / hashlib.sha256(content).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        f = snap / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.symlink_to(os.path.relpath(blob, f.parent))
    return snap


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """{revision: its snapshot} of Org/m, served by the stub engine (a snapshot path is served as it is)."""
    monkeypatch.setattr(server, "resolve", lambda name, *_, **__: Path(name))
    monkeypatch.setattr(server, "EagerBackend", StubBackend)
    return {rev: snapshot(tmp_path / "hub", "Org/m", rev, dict(FILES, **change)) for rev, change in CHANGES.items()}


def serve_snapshot(path, md, *extra):
    return Server(parser().parse_args(["--mode", "eager", "--model", str(md), "--calibration", str(path), *extra]))


def test_a_card_only_commit_keeps_the_thresholds_and_a_weight_change_drops_them(tmp_path, hub, capsys):
    synthetic(tmp_path / "labelled.jsonl")
    out = tmp_path / "CALIBRATION.custom.json"
    main([str(tmp_path / "labelled.jsonl"), "--mode", "eager", "--model", str(hub[R1]), "--out", str(out)])
    cal = json.loads(out.read_text(encoding="utf-8"))
    shutil.rmtree(hub[R1])  # the file records the model files: the fitted snapshot need not be in the cache
    capsys.readouterr()
    srv = serve_snapshot(out, hub[R2], "--strict-calibration")      # README, SHA256SUMS, an image, .gitattributes
    assert srv.cal_mismatch is None and srv.cal_unvalidated is None and "warning" not in capsys.readouterr().out
    assert srv.cal["thresholds"] == cal["thresholds"]
    for rev in (R3, R4):                                            # the weights, the tokenizer
        srv = serve_snapshot(out, hub[rev])
        assert f"fitted on revision {R1[:12]} of Org/m, the served revision {rev[:12]} has other model files" \
            in srv.cal_mismatch
        assert "thresholds" not in srv.cal and srv.cal["temperature_per_prim"] == cal["temperature_per_prim"]
        assert srv.cal_unvalidated and "they are not applied" in capsys.readouterr().out
        with pytest.raises(SystemExit, match="other model files.*--strict-calibration"):
            serve_snapshot(out, hub[rev], "--strict-calibration")
    fitted = cal["fitted_on"]
    assert (fitted["model"], fitted["revision"]) == ("Org/m", R1) and len(fitted["model_files"]) == 16


def test_a_file_without_model_files_is_compared_with_the_fitted_snapshot_in_the_cache(hub):
    """A file of basal-calibrate 2.6.0 or older records the revision only: the cache's snapshot of that revision, when
    it holds the same files as the served one, decides. Otherwise it is kept for its temperatures only, as before, and
    the warning says how to keep its thresholds. The Hub is not asked."""
    cal = {"fitted_on": {"model": "Org/m", "revision": R1}}
    assert calibration_mismatch(cal, str(hub[R2]), hub[R2]) is None
    assert "other model files" in calibration_mismatch(cal, str(hub[R3]), hub[R3])
    assert "other model files" in calibration_mismatch(cal, str(hub[R4]), hub[R4])
    assert calibration_mismatch(cal, str(hub[R1]), hub[R1]) is None  # the same revision: as before
    unknown = (f"fitted on revision {R1[:12]} of Org/m, the served revision is {R2[:12]}, and the cache cannot tell "
               "whether they hold the same model files")
    for name in ("config.json", "model.safetensors"):  # a snapshot fetched in part, then without its weights
        (hub[R1] / name).unlink()
        why = calibration_mismatch(cal, str(hub[R2]), hub[R2])
        assert unknown in why and f"--revision {R1}" in why and "refit with basal-calibrate" in why
    shutil.rmtree(hub[R1])                             # not in the cache
    assert unknown in calibration_mismatch(cal, str(hub[R2]), hub[R2])
    for bad in ("..", "../../models--Org--m", "main"):  # not a snapshot of this repository
        why = calibration_mismatch({"fitted_on": {"model": "Org/m", "revision": bad}}, str(hub[R2]), hub[R2])
        assert unknown.replace(R1[:12], bad[:12]) in why


def test_model_files_are_read_from_the_links_not_from_the_files(tmp_path, hub):
    cal = {"fitted_on": {"model": "Org/m", "revision": R1}}
    blob = (hub[R2] / "model.safetensors").resolve()
    (tmp_path / "store").mkdir()                       # a blob that links on, to a store elsewhere
    weights = blob.rename(tmp_path / "store" / "w")
    blob.symlink_to(weights)
    weights.chmod(0)                                   # the weights cannot be read: the links and sizes suffice
    try:
        assert calibration_mismatch(cal, str(hub[R2]), hub[R2]) is None
    finally:
        weights.chmod(0o644)
    copy = hub[R2] / "model.safetensors"               # a cache without symlinks: its files would have to be read
    copy.unlink()
    copy.write_bytes(FILES["model.safetensors"])
    assert "the cache cannot tell" in calibration_mismatch(cal, str(hub[R2]), hub[R2])


def test_another_model_adapter_or_engine_is_caught_as_before(tmp_path, hub):
    other = snapshot(tmp_path / "hub", "OrgB/m", R2, dict(FILES, **CHANGES[R2]))  # the same files, another repo
    old = {"model": "Org/m", "revision": R1}
    for fitted in (old, dict(old, model_files="0123456789abcdef")):
        why = calibration_mismatch({"fitted_on": fitted}, "OrgB/m", other)
        assert why == "it was fitted on Org/m, the served model is OrgB/m"
    why = calibration_mismatch({"fitted_on": dict(old, adapter="edu@0123456789ab")}, str(hub[R2]), hub[R2])
    assert why == "it was fitted with adapter edu@0123456789ab, the served adapter is none"
    # GGUF, Ollama and llama.cpp serve weights the model directory does not hold: the revision decides
    why = calibration_mismatch({"fitted_on": old}, str(hub[R2]), hub[R2], kind="gguf")
    assert why.startswith(f"it was fitted on revision {R1[:12]} of Org/m, the served revision is {R2[:12]} (")


def test_refit_from_rows_keeps_the_model_files(tmp_path):
    head = dict(basal_calibrate_rows=1, fmt=1, kind="eager", quant=None, model="Org/m", engine="eager",
                data="labels.jsonl", requests=1, baseline=None, orders=1, layout="canonical", revision=R1,
                model_files="0123456789abcdef")
    rows = [dict(prim="noul", group=f"g{i}", family="f", gold=i % 2, p=[0.9, 0.1], line=i, name="q",
                 keys=["true", "false"]) for i in range(20)]
    path, out = tmp_path / "rows.jsonl", tmp_path / "C.json"
    path.write_text("\n".join(json.dumps(x) for x in [head] + rows) + "\n", encoding="utf-8")
    main([str(path), "--from-rows", "--no-calibration", "--targets", "0.05", "--out", str(out)])
    fitted = json.loads(out.read_text(encoding="utf-8"))["fitted_on"]
    assert (fitted["revision"], fitted["model_files"]) == (R1, "0123456789abcdef")
