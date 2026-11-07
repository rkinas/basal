"""basal-serve's start check of a calibration file (server.check_calibration) covers what it reads at a request, as it
covers the temperatures (#119): thresholds, conformal levels, fitted fallback thresholds and task thresholds. A hand-
edited or corrupt value passed the check and answered 500 at a request or at GET /v1/models; it is refused at start-up
with the key named. Values basal-calibrate writes (numbers, null for no certificate) pass unchanged, and numeric text
is read as a number, as a temperature is."""
import copy
import json

import pytest

from basal.server import certified_table, check_calibration, plain_limits

PUBLISHED_SHAPE = {  # (the shape of the published files: per-type thresholds, null for no certificate, task thresholds)
    "prompt": 2, "temperature_per_prim": {"choice": 1.04, "noul": 1.56},
    "temperature_per_bucket": {"choice": {"2-10": 1.1, "11-26": 1.2}},
    "thresholds": {"0.01": {"confidence": None, "per_type": {"choice": {"confidence": 0.98, "n": 300},
                                                             "noul": {"confidence": None, "reason": "too few"}}},
                   "0.05": {"confidence": 0.9, "per_type": {"choice": {"confidence": 0.9}}}},
    "task_thresholds": {"verify": {"0.05": 0.97}},
    "conformal": {"levels": {"0.9": {"per_type": {"choice": {"p_min": 0.2}, "noul": {"p_min": None}}}}},
    "multi": {"set1": {"fallback_threshold": 0.3}},
}


def test_the_files_basal_calibrate_writes_pass_unchanged():
    cal = copy.deepcopy(PUBLISHED_SHAPE)
    check_calibration(cal, 2, "CALIBRATION.json")
    assert cal == PUBLISHED_SHAPE
    check_calibration({}, 1, "CALIBRATION.json")
    check_calibration({"thresholds": {"0.05": None}, "multi": None, "conformal": {}}, 1, "CALIBRATION.json")


def bad(**parts):
    return dict(copy.deepcopy(PUBLISHED_SHAPE), **parts)


@pytest.mark.parametrize("cal, said", [
    ([], "a calibration file is a JSON object"),
    (bad(thresholds={"0.05": 5}), "thresholds['0.05'] must be an object"),
    (bad(thresholds=[]), "thresholds must be an object"),
    (bad(thresholds={"0.05": {"per_type": {"noul": "x"}}}), "thresholds['0.05'].per_type['noul'] must be an object"),
    (bad(thresholds={"0.05": {"per_type": ["noul"]}}), "thresholds['0.05'].per_type must be an object"),
    (bad(thresholds={"0.05": {"confidence": "high"}}), "thresholds['0.05'].confidence is 'high'"),
    (bad(thresholds={"0.05": {"per_type": {"noul": {"confidence": 1.5}}}}), "per_type['noul'].confidence is 1.5"),
    (bad(thresholds={"0.05": {"per_type": {"noul": {"confidence": float("nan")}}}}), "confidence is nan"),
    (bad(thresholds={"0.05": {"per_type": {"noul": {"confidence": True}}}}), "confidence is True"),
    (bad(thresholds={"five percent": {}}), "thresholds key 'five percent'"),
    (bad(conformal={"levels": {"0.9": {"per_type": {"noul": {"p_min": "low"}}}}}), "p_min is 'low'"),
    (bad(conformal={"levels": [0.9]}), "conformal.levels must be an object"),
    (bad(multi={"set1": "0.3"}), "multi['set1'] must be an object"),
    (bad(multi={"set1": {"fallback_threshold": -0.1}}), "multi['set1'].fallback_threshold is -0.1"),
    (bad(task_thresholds={"verify": {"0.05": "x"}}), "task_thresholds['verify']['0.05'] is 'x'"),
    (bad(task_thresholds={"verify": 0.97}), "task_thresholds['verify'] must be an object"),
    (bad(temperature_per_prim=[1.0]), "temperature_per_prim must be an object"),
    (bad(temperature_per_bucket={"choice": [1.0]}), "temperature_per_bucket['choice'] must be an object"),
])
def test_a_value_basal_serve_cannot_read_is_refused_at_start_up_with_its_key(cal, said):
    with pytest.raises(SystemExit, match=said.replace("[", r"\[").replace("]", r"\]").replace(".", r"\.")):
        check_calibration(cal, 2, "CALIBRATION.json")


def test_numeric_text_is_read_as_a_number_and_then_serves():
    """A confidence or p_min written as text (a hand edit): read as the number, as a temperature is; at a request it
    was compared with a float (TypeError, HTTP 500), and GET /v1/models failed on a thresholds entry that was a
    number."""
    cal = bad(thresholds={"0.05": {"per_type": {"noul": {"confidence": "0.9"}}}},
              conformal={"levels": {"0.9": {"per_type": {"noul": {"p_min": "0.2"}}}}})
    check_calibration(cal, 2, "CALIBRATION.json")
    assert cal["thresholds"]["0.05"]["per_type"]["noul"]["confidence"] == 0.9
    assert certified_table(cal) == {"0.05": {"noul": 0.9}}
    ans = plain_limits({}, {"type": "noul", "max_error": 0.05, "coverage": 0.9}, {"true": 0.7, "false": 0.3}, 0.7, cal)
    assert ans["abstain"] is True and ans["answer_set"] == ["true", "false"]
    assert json.dumps(cal)  # (still JSON)
