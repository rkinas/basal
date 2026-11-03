"""A calibration file holds for the option orders it was fitted with (fitted_on.orders): basal-serve serving other
orders loads CALIBRATION[.<engine>].o<N>.json when the model ships one, otherwise keeps the temperatures without the
thresholds and fitted fallbacks (max_error defers) and marks every answer unvalidated."""
import asyncio
import json

from basal.server import calibration_orders, load_calibration, models_payload
from tests.test_server import FakeServer

CAL = {"prompt": 2, "temperature_per_prim": {"choice": 1.2, "noul": 1.5},
       "thresholds": {"0.01": {"confidence": 0.9, "per_type": {"noul": {"confidence": 0.9}}}},
       "multi": {"abc": {"fallback": "x", "fallback_threshold": 0.2}}}


def write(d, name, orders, **extra):
    (d / name).write_text(json.dumps(dict(CAL, fitted_on={"orders": orders}, **extra)), encoding="utf-8")


def test_shipped_file_shapes_load(tmp_path):
    """A basal-1.5 file (fitted_on a text) loads unchanged; a training-side package fitted on two orders is
    unvalidated at one order."""
    (tmp_path / "CALIBRATION.json").write_text(json.dumps(dict(CAL, fitted_on="calib FIT split (70% by twin group)")), encoding="utf-8")
    cal, _, why = load_calibration(tmp_path, "graph", None, None, 2)
    assert why is None and cal["thresholds"]
    (tmp_path / "CALIBRATION.json").write_text(json.dumps(dict(CAL, fitted_on="calibration FIT split", orders=2)), encoding="utf-8")
    assert load_calibration(tmp_path, "graph", None, None, 1)[2] == ("CALIBRATION.json was fitted on 2 option "
                                                                      "order(s), served with 1")


def test_the_file_for_the_served_orders(tmp_path):
    write(tmp_path, "CALIBRATION.json", 1)
    cal, path, why = load_calibration(tmp_path, "graph", None, None, 1)
    assert why is None and path.name == "CALIBRATION.json" and cal["thresholds"]
    write(tmp_path, "CALIBRATION.o2.json", 2, engine="o2 file")
    cal, path, why = load_calibration(tmp_path, "graph", None, None, 2)           # --orders 2: the .o2 file
    assert why is None and path.name == "CALIBRATION.o2.json" and cal["engine"] == "o2 file"
    write(tmp_path, "CALIBRATION.fp8.json", 1)
    write(tmp_path, "CALIBRATION.fp8.o2.json", 2)
    assert load_calibration(tmp_path, "graph", None, "fp8", 2)[1].name == "CALIBRATION.fp8.o2.json"


def test_other_orders_without_their_file_are_unvalidated(tmp_path):
    write(tmp_path, "CALIBRATION.json", 2)                                        # fitted on two orders
    cal, path, why = load_calibration(tmp_path, "graph", None, None, 1)           # served with one, no .o1 file
    assert why == "CALIBRATION.json was fitted on 2 option order(s), served with 1"
    assert cal["temperature_per_prim"] == CAL["temperature_per_prim"] and "thresholds" not in cal and "multi" not in cal
    write(tmp_path, "CALIBRATION.o1.json", 1)
    explicit = load_calibration(tmp_path, "graph", str(tmp_path / "CALIBRATION.json"), None, 1)
    assert explicit[2] is not None                                                # --calibration is not replaced
    assert load_calibration(tmp_path, "graph", None, None, "auto")[2] is not None  # auto is its own order setting
    (tmp_path / "CALIBRATION.json").write_text(json.dumps(CAL), encoding="utf-8")                   # a file that does not say
    assert load_calibration(tmp_path, "graph", None, None, 1)[2] is None
    assert load_calibration(tmp_path, "graph", None, None, 1, off=True)[0] == {}


def test_orders_field():
    assert calibration_orders({"fitted_on": {"orders": 2}}) == 2 and calibration_orders({"orders": "1"}) == 1
    # the shapes of the shipped files: basal-1.5 (fitted_on a text, no orders) and the training side's packages
    # (fitted_on a text, a top-level orders)
    assert calibration_orders({"fitted_on": "calib FIT split (70% by twin group)"}) is None
    assert calibration_orders({"fitted_on": "calibration FIT split (70% by twin group)", "orders": 2}) == 2
    assert calibration_orders({"fitted_on": {"data": "x"}, "orders": 1}) == 1
    assert calibration_orders({"fitted_on": {"orders": "auto"}}) == "auto" and calibration_orders({}) is None
    assert models_payload("m", "fast", {}, orders=1)["models"][0]["orders"] == 1  # basal-calibrate --url records it


def test_answers_are_marked_and_max_error_defers():
    q = {"type": "act", "instructions": "Covered?", "criteria": {"true": "Yes", "false": "No"},
         "costs": {"wrong": 10, "defer": 1}, "max_error": 0.01}
    body = {"state": "x", "questions": {"a": q, "c": {"type": "choice", "criteria": ["p", "q"]}}}
    srv = FakeServer([0.99, 0.01], cal={k: v for k, v in CAL.items() if k not in ("thresholds", "multi")})
    srv.orders, srv.cal_unvalidated = 1, "CALIBRATION.json was fitted on 2 option order(s), served with 1"
    out = asyncio.run(srv._run(body))["answers"]
    assert out["a"]["calibration"] == out["c"]["calibration"] == "unvalidated"
    assert out["a"]["deferred"] is True and out["a"]["action"] == "defer"          # P(true) 0.99, but no threshold
    match = FakeServer([0.99, 0.01], cal=CAL)
    match.orders = 1
    plain = asyncio.run(match._run(body))["answers"]
    assert plain["a"]["deferred"] is False and plain["a"]["action"] == "true"     # 0.99 >= the noul threshold 0.9
    assert "calibration" not in plain["c"]                                        # nothing changes when the orders match


def test_orders_2le():
    """--orders 2le<N>: both option orders for questions with at most N options, one above."""
    import argparse
    import pytest
    from basal.server import default_orders, orders_arg, orders_for
    assert [orders_for("2le3", k) for k in (2, 3, 4, 10)] == [2, 2, 1, 1]
    assert orders_for(2, 9) == 2 and orders_for(1, 2) == 1 and orders_for("auto", 2) == 1
    assert default_orders("2le4", 2) == "2le4" and default_orders("2", 2) == 2 and default_orders(None, 2) == 1
    for bad in ("2le1", "3", "2lex", "le3"):
        with pytest.raises(argparse.ArgumentTypeError):
            orders_arg(bad)
    assert calibration_orders({"fitted_on": {"orders": "2le3"}}) == "2le3"
    srv = FakeServer(lambda p, ids: [1 / len(ids)] * len(ids))
    srv.orders = "2le3"
    seen = []
    srv.probs_for = lambda p, ids: seen.append(len(ids)) or [1 / len(ids)] * len(ids)
    body = {"state": "x", "questions": {"two": {"type": "choice", "criteria": ["a", "b"]},
                                        "five": {"type": "choice", "criteria": ["a", "b", "c", "d", "e"]}}}
    asyncio.run(srv._run(body))
    assert sorted(seen) == [2, 2, 5]                       # two orders for 2 options, one for 5


def test_the_file_for_the_served_orders_is_checked_for_the_layout_too(tmp_path):
    """The .oN file is picked after the layout check of CALIBRATION.json: its own layout is checked as well."""
    write(tmp_path, "CALIBRATION.json", 1)
    write(tmp_path, "CALIBRATION.o2.json", 2, layout="qfirst")
    cal, path, why = load_calibration(tmp_path, "graph", None, None, 2, layout="canonical")
    assert "thresholds" not in cal and why is not None
    write(tmp_path, "CALIBRATION.json", 1, layout="qfirst")                      # the same layout: the .o2 file loads
    cal, path, why = load_calibration(tmp_path, "graph", None, None, 2, layout="qfirst")
    assert why is None and path.name == "CALIBRATION.o2.json" and cal["thresholds"]
