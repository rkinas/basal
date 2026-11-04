"""The bound of a threshold counts decisions as independent: basal-calibrate warns when a type or task has more FIT
decisions than groups (claims or fields of one document share their errors), and stays quiet with one per group."""
from basal.calibrate import fit, fit_tasks, is_cert, task_warnings, warnings


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
