"""A threshold certified on FIT is withdrawn when the held-out CERT decisions refute it (their one-sided 95% lower
bound of the error is above the target): CERT took no part in the fit, so a threshold that holds is withdrawn at most
alpha of the time, and one CERT cannot confirm for lack of decisions keeps only its note."""
import math

from basal import calibrate
from basal.calibrate import fit, fit_tasks, is_cert

FIT_G = [g for g in (f"g{i}" for i in range(2000)) if not is_cert(g)]
CERT_G = [g for g in (f"g{i}" for i in range(2000)) if is_cert(g)]


def test_a_threshold_cert_refutes_is_withdrawn():
    """100 correct FIT decisions and 100 wrong CERT ones at the same confidence: certified on FIT, refuted on CERT."""
    rows = [dict(prim="noul", p=[0.995, 0.005], gold=0, group=g, family="f") for g in FIT_G[:100]] + \
           [dict(prim="noul", p=[0.995, 0.005], gold=1, group=g, family="f") for g in CERT_G[:100]]
    cal, rep = fit(rows, {}, [0.05], min_per_type=10 ** 9, min_blend=0)
    x = cal["thresholds"]["0.05"]["per_type"]["noul"]
    assert x["confidence"] is None and x["refuted"] > 0.99 and x["cert_error"] == 1.0
    assert "CERT refutes it: 100 errors in 100 accepted" in x["reason"]
    assert calibrate.type_threshold(cal["thresholds"]["0.05"], "noul") is None      # basal-serve defers
    assert cal["gates"]["automated_at"]["0.05"] == 0.0
    tasks = [dict(task="classify", group=g, c=0.995, ok=True) for g in FIT_G[:100]] + \
            [dict(task="classify", group=g, c=0.995, ok=False) for g in CERT_G[:100]]
    thr, trep = fit_tasks(tasks, [0.05])
    assert thr == {} and trep["classify"]["targets"]["0.05"]["refuted"] > 0.99


def test_a_few_cert_errors_or_too_few_cert_decisions_do_not_withdraw():
    rows = [dict(prim="noul", p=[0.995, 0.005], gold=0, group=g, family="f") for g in FIT_G[:300]] + \
           [dict(prim="noul", p=[0.995, 0.005], gold=int(i < 2), group=g, family="f") for i, g in enumerate(CERT_G[:20])]
    x = fit(rows, {}, [0.05], min_per_type=10 ** 9, min_blend=0)[0]["thresholds"]["0.05"]["per_type"]["noul"]
    assert x["confidence"] is not None and x["cert_error"] == 0.1 and "refuted" not in x   # 2 / 20: not significant


def test_a_threshold_that_holds_is_refuted_at_most_alpha_of_the_time():
    """Exact: the chance that CERT refutes a threshold whose error is the target itself (or less) is at most alpha."""
    for risk, n in ((0.05, 400), (0.01, 1500), (0.05, 60), (0.01, 129)):
        for q in (risk, risk / 2):
            pmf = [math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(q)
                            + (n - k) * math.log1p(-q)) for k in range(n + 1)]
            assert sum(p for k, p in enumerate(pmf) if calibrate.refutes(k, n, risk)) <= calibrate.ALPHA
    assert calibrate.refutes(5, 100, 0.01) and not calibrate.refutes(2, 100, 0.01) and not calibrate.refutes(0, 0, 0.05)
