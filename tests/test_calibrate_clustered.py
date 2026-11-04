"""The bound of a threshold counts decisions as independent: basal-calibrate warns when a type or task has more FIT
decisions than groups (claims or fields of one document share their errors), and stays quiet with one per group."""
import numpy as np

from basal import calibrate
from basal.calibrate import ALPHA, fit, fit_tasks, is_cert, task_warnings, threshold, warnings


def test_a_task_with_many_decisions_per_group_is_warned():
    rows = [dict(task="verify", group=f"doc{i // 30}", c=0.99, ok=True) for i in range(3000)]
    _, rep = fit_tasks(rows, [0.05])
    groups = len({r["group"] for r in rows if not is_cert(r["group"])})
    assert rep["verify"]["fit_groups"] == groups and rep["verify"]["n_fit"] == 30 * groups
    w = [x for x in task_warnings(rep) if "per group" in x]
    assert len(w) == 1 and w[0].startswith(f"verify: {30 * groups} FIT decisions in {groups} groups (30.0 per group)")
    single = [dict(r, group=f"c{i}") for i, r in enumerate(rows)]
    assert not [x for x in task_warnings(fit_tasks(single, [0.05])[1]) if "per group" in x]


def test_a_decision_type_with_shared_groups_is_warned():
    rows = [dict(prim="noul", p=[0.99, 0.01], gold=0, group=f"g{i // 3}", family="f") for i in range(900)]
    _, rep = fit(rows, {}, [0.05], min_per_type=10)
    assert rep["types"]["noul"]["fit_groups"] * 3 == rep["types"]["noul"]["n_fit"]
    assert any(x.startswith("noul: ") and "(3.0 per group)" in x for x in warnings(rep, [0.05]))
    alone = [dict(r, group=f"g{i}") for i, r in enumerate(rows)]
    assert not [x for x in warnings(fit(alone, {}, [0.05], min_per_type=10)[1], [0.05]) if "per group" in x]


def documents(rng, docs, per, mean, conc=2.0):
    """docs x per decisions; each document has its own error rate (beta, mean `mean`): its claims share their errors.
    Confidence is unrelated to the errors, so every set of accepted decisions has the error `mean` on average."""
    q = rng.beta(mean * conc, (1 - mean) * conc, docs)
    wrong = rng.random((docs, per)) < q[:, None]
    conf = rng.uniform(0.5, 1.0, (docs, per)).round(6)
    return list(zip(conf.ravel().tolist(), (~wrong).ravel().tolist())), [f"d{i}" for i in range(docs) for _ in range(per)]


def test_clustered_errors_are_certified_at_most_alpha_of_the_time():
    """100 documents x 30 claims at a mean error of 1.2%, target 1%: counting the claims as independent certified the
    target in about 10% of runs; counting them as the independent decisions they are worth stays under 5%."""
    rng = np.random.default_rng(2026)
    trials, naive, clus = 150, 0, 0
    for _ in range(trials):
        cs, groups = documents(rng, 100, 30, 0.012)
        naive += threshold(cs, 0.01) is not None
        clus += threshold(cs, 0.01, deff=calibrate.design_effect(cs, groups)[1]) is not None
    assert naive / trials > 0.06 and clus / trials <= ALPHA


def test_one_decision_per_group_or_unshared_errors_change_nothing():
    rng = np.random.default_rng(3)
    cs = [(round(float(c), 6), bool(rng.random() > 0.004)) for c in rng.uniform(0.5, 1.0, 3000)]
    alone = [f"g{i}" for i in range(3000)]
    assert calibrate.design_effect(cs, alone) == (0.0, 1.0)
    assert threshold(cs, 0.01, deff=1.0) == threshold(cs, 0.01)
    assert calibrate.effective(300, 3, 2.0) == (150, 2) and calibrate.effective(300, 3) == (300, 3)
    assert calibrate.design_effect_of([f"d{i // 3}" for i in range(300)], 0.5) == 2.0     # groups of 3, rho 0.5
    same = [(0.9, i % 30 != 0) for i in range(300)]                        # every error alone in its group of 10
    assert calibrate.icc([(c, (i // 10) % 3 != 0) for i, (c, _) in enumerate(same)],
                         [f"d{i // 10}" for i in range(300)]) == 1.0    # whole groups wrong together


def test_the_report_names_the_correlation_and_the_independent_count():
    rng = np.random.default_rng(4)
    cs, groups = documents(rng, 300, 20, 0.002, conc=0.5)
    rows = [dict(task="verify", group=g, c=c, ok=ok) for (c, ok), g in zip(cs, groups)]
    _, rep = fit_tasks(rows, [0.05])
    v = rep["verify"]
    assert v["icc"] > 0.2
    w = [x for x in task_warnings(rep) if "per group" in x][0]
    assert "go together within a group (ICC" in w and "fewer independent decisions" in w
    assert 1 + 19 * 0.2 < v["design_effect"] == 1 + 19 * v["icc"]          # groups of 20
