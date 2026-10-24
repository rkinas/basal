"""Batch inference: a JSONL file through a loaded model, the rows and the download, cancelling, and that batch requests
stay out of the statistics and the history."""
import json

from tests.e2e.support import EXAMPLES

QUESTIONS = EXAMPLES / "questions.jsonl"


def test_a_file_is_answered_line_by_line_and_can_be_downloaded(ui, text_model, tmp_path):
    model_id = text_model["id"]
    ui.load(model_id)
    requests_before = (ui.get("/api/stats")["models"].get(model_id) or {}).get("requests", 0)
    out = tmp_path / "answers.jsonl"
    job = ui.run_job("/api/batch", {"id": model_id, "input": str(QUESTIONS), "output": str(out), "concurrency": 4},
                     "batch")
    lines = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    result = job["result"]
    assert (result["items"], result["errors"], job["done"], job["total"]) == (len(lines), 0, len(lines), len(lines))
    assert 0 <= result["accuracy"] <= 1 and result["output"] == str(out)
    answers = [json.loads(line) for line in out.read_text().splitlines()]
    assert [a["id"] for a in answers] == [item.get("id", i) for i, item in enumerate(lines)]  # the input's order
    assert all(0 <= a["prediction"] < len(item["options"]) for a, item in zip(answers, lines))
    page = ui.get("/api/batch/rows", params={"offset": 5, "limit": 5})
    assert page["total"] == len(lines) and page["rows"] == answers[5:10]
    download = ui.get("/api/batch/download")
    assert [json.loads(line) for line in download.text.splitlines()] == answers
    assert (ui.get("/api/stats")["models"].get(model_id) or {}).get("requests", 0) == requests_before


def test_a_batch_can_be_cancelled_and_bad_files_are_refused(ui, text_model, tmp_path):
    model_id = text_model["id"]
    ui.load(model_id)
    many = tmp_path / "many.jsonl"
    many.write_text(QUESTIONS.read_text() * 50)
    ui.post("/api/batch", {"id": model_id, "input": str(many), "output": str(tmp_path / "many.out.jsonl"),
                           "concurrency": 1})
    ui.delete("/api/batch")
    assert ui.wait_job("batch")["phase"] == "cancelled"
    assert not (tmp_path / "many.out.jsonl").exists()
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"state": "only a state"}\n')
    assert "line 1" in ui.detail("POST", "/api/batch", {"id": model_id, "input": str(bad)}, 400)
    assert "cannot write" in ui.detail("POST", "/api/batch", {"id": model_id, "input": str(QUESTIONS),
                                                              "output": "/nonexistent/dir/a.jsonl"}, 400)
