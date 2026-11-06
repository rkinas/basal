"""basal-serve's start-up checks: the temperatures of a calibration file, a missing --api-keys file."""
import pytest

from basal.server import check_calibration


def refused(cal):
    try:
        check_calibration(cal, 1, "CAL.json")
        return False
    except SystemExit:
        return True


@pytest.mark.parametrize("T", [0, 0.0, -1.0, float("nan"), float("inf"), False, "0", "nan", "inf", "abc", None, [1]])
def test_a_temperature_that_is_not_a_finite_number_above_zero_is_refused_at_start_up(T):
    assert refused({"prompt": 1, "temperature_per_prim": {"choice": T}})
    assert refused({"prompt": 1, "temperature_per_bucket": {"choice": {"2-10": T}}})


@pytest.mark.parametrize("cal", [{"temperature_per_prim": {"choice": 1.5}}, {"temperature_per_prim": {"choice": 2}},
                                 {"temperature_per_bucket": {"choice": {"2-10": 2.0}}},
                                 {"temperature_per_prim": {"choice": "1.2"}},
                                 {"temperature_per_bucket": {"choice": {"2-10": "1.1"}}},
                                 {"temperature_per_prim": {"nobody_asks": 0}}, {}])
def test_a_valid_calibration_is_accepted(cal):
    assert not refused({"prompt": 1, **cal})


def test_the_question_types_of_a_request_are_the_ones_whose_temperatures_are_checked():
    from basal import server
    assert server.QUESTION_TYPES == ("choice", "noul", "score", "act", "multi", "tool")
    with pytest.raises(server.RequestError, match="unknown type"):
        server.to_items("s", {"q": {"type": "nobody_asks", "instructions": "?"}}, 1)


@pytest.mark.parametrize("key", ["choice", "nobody_asks"])
def test_a_bad_bucket_name_is_refused_under_any_key(key):
    assert refused({"prompt": 1, "temperature_per_bucket": {key: {"x": 1.0}}})
