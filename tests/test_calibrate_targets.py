"""basal-calibrate's --targets bound and its warnings about bucket temperatures on the edge of the grid."""
import time

import pytest

from basal import calibrate as cal


@pytest.mark.parametrize("targets", ["1e-9", "1e-17", "0"])
def test_targets_below_one_in_a_million_are_refused_at_the_parser(targets):
    with pytest.raises(SystemExit, match="targets"):
        cal.main(["/nonexistent.jsonl", "--targets", targets])


def test_the_smallest_accepted_target_costs_little_and_normal_ones_pass_the_parser():
    t = time.time()
    assert cal.accepted_needed(1e-4, 1) > 0 and time.time() - t < 60
    assert cal.accepted_needed(0.05) == 59
    with pytest.raises(BaseException) as e:
        cal.main(["/nonexistent.jsonl", "--targets", "0.05"])
    assert "targets" not in str(e.value)
