"""Calibration file 2: a temperature per question type and option-count bucket, else the per-type temperature."""
import asyncio
import math

import pytest

from basal.server import calibration_file, calibration_name, check_calibration, temperature
from tests.test_prompt2 import choice, fmt2_server

CAL2 = {"prompt": 2,
        "temperature_per_prim": {"choice": 1.5, "noul": 2.0, "score": 1.0},
        "temperature_per_bucket": {"choice": {"2-10": 1.1, "11-26": 1.3, "27-255": 1.7}}}


def test_bucket_then_prim_then_one():
    assert [temperature(CAL2, "choice", k) for k in (2, 10, 11, 26, 27, 255)] == [1.1, 1.1, 1.3, 1.3, 1.7, 1.7]
    assert temperature(CAL2, "noul", 2) == 2.0          # no bucket for the type: the per-type temperature
    assert temperature({"temperature_per_prim": {"choice": 1.2}}, "choice", 7) == 1.2  # a 1.5 file
    partial = {"temperature_per_prim": {"choice": 1.5}, "temperature_per_bucket": {"choice": {"27-255": 1.9}}}
    assert temperature(partial, "choice", 5) == 1.5 and temperature(partial, "choice", 40) == 1.9
    assert temperature({}, "score", 4) == 1.0


def test_bad_files_are_refused_at_startup():
    check_calibration(CAL2, 2, "C.json")
    check_calibration({"temperature_per_prim": {"choice": 1.2}}, 1, "C.json")  # 1.5 file: no prompt key
    with pytest.raises(SystemExit, match="fitted for prompt format 2"):
        check_calibration(CAL2, 1, "C.json")
    with pytest.raises(SystemExit, match="bad bucket"):
        check_calibration({"temperature_per_bucket": {"choice": {"2..10": 1.0}}}, 1, "C.json")
    with pytest.raises(SystemExit, match="bad bucket"):
        check_calibration({"temperature_per_bucket": {"choice": {"26-11": 1.0}}}, 1, "C.json")


def test_per_engine_file_names():
    assert calibration_name("graph", None) is None and calibration_name("eager", None) is None
    assert calibration_name("graph", "fp8") == "fp8" and calibration_name("exit", "nvfp4") == "nvfp4"
    assert [calibration_name(k, None) for k in ("vllm", "sglang", "mlx", "ollama", "llamacpp", "tpu")] == \
        ["vllm", "sglang", "mlx", "ollama", "llamacpp", "tpu"]
    assert calibration_name("mlx", "q8") == "mlx" and calibration_name("mps", None) is None


def test_per_engine_file_when_shipped(tmp_path):
    (tmp_path / "CALIBRATION.json").write_text("{}")
    assert calibration_file(tmp_path, "tpu") == tmp_path / "CALIBRATION.json"  # none for the engine: the default
    (tmp_path / "CALIBRATION.tpu.json").write_text("{}")
    assert calibration_file(tmp_path, "tpu") == tmp_path / "CALIBRATION.tpu.json"
    assert calibration_file(tmp_path, "graph") == tmp_path / "CALIBRATION.json"
    assert calibration_file(tmp_path, "tpu", path="X.json").name == "X.json"  # --calibration wins


def tempered(p, T):
    z = [math.log(x) / T for x in p]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    return [x / sum(e) for x in e]


@pytest.mark.parametrize("k", [3, 12, 40])
def test_server_applies_the_bucket_temperature(k):
    pos = [0.6] + [0.4 / (k - 1)] * (k - 1)
    srv = fmt2_server(lambda prompt, ids: pos[: len(ids)])
    srv.orders, srv.cal = 1, CAL2  # one order: the canonical probabilities are the position probabilities
    ans = asyncio.run(srv._run(choice(k)))["answers"]["q"]
    want = tempered(pos, temperature(CAL2, "choice", k))
    assert max(abs(a - b) for a, b in zip(ans["probabilities"].values(), want)) < 1e-6
