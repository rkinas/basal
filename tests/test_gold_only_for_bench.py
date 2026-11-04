"""Z-1113: gold is the bench business, not loadtest or ordercheck."""
import json

from basal import bench


def test_z2_a_file_with_a_letter_gold_is_taken_by_loadtest_and_ordercheck(tmp_path):
    """gold is the bench's business: basal-loadtest and basal-ordercheck read the file as they did on main."""
    import inspect

    from basal import loadtest, ordercheck
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(dict(state="s", question="q", options=["a", "b"], gold="B")) + "\n", encoding="utf-8")
    assert len(bench.questions_or_exit(str(path), 10)) == 1  # the reader both commands call
    assert loadtest.questions_or_exit is bench.questions_or_exit
    assert "check_gold" not in inspect.getsource(loadtest) + inspect.getsource(ordercheck)
    assert "questions_or_exit" in inspect.getsource(ordercheck)
