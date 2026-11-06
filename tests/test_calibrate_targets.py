"""basal-calibrate's --targets bound and its warnings about bucket temperatures on the edge of the grid."""
import math
import random
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


def test_a_bucket_whose_temperature_sits_on_the_edge_of_the_grid_is_warned_about():
    rows = [dict(prim="choice", p=[0.999, 0.001], gold=0, group=str(i), family="f") for i in range(300)]
    rnd = random.Random(1)
    for i in range(300, 600):
        g = 0 if rnd.random() < 0.5 else rnd.randrange(1, 12)
        rows.append(dict(prim="choice", p=[0.5] + [0.5 / 11] * 11, gold=g, group=str(i), family="f"))
    _, report = cal.fit(rows, {}, [0.05], min_per_type=50, fmt=2)
    edge = [b for b, r in report["buckets"].items()
            if r.get("T") and not r.get("T_type") and any(math.isclose(r["T"], x) for x in cal.T_GRID)]
    assert edge, "the fixture no longer puts a bucket on the edge"
    assert any(f"choice {b} options" in w for b in edge for w in cal.warnings(report, [0.05]))
