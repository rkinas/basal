"""Batch inference from basal-ui: basal-run over a JSONL file (simple items or full /v1/systemone requests) against a
loaded model's private basal-serve; the answers go to a JSONL file, one line per input line. The requests go straight
to the child: they are not counted in the statistics or the history (as benchmarks)."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from basal import run

from .jobs import Cancelled


def default_output(input_path) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}.answers.jsonl")


def run_batch(job, fleet, model_id, input_path, output_path, concurrency=16, early_exit=None) -> dict:
    """The batch job: progress per answered line; a cancel stops the remaining requests (no output file then)."""
    runner = fleet.ready(model_id)
    total = sum(1 for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip())
    job.progress("running", 0, total)

    def progress(done, total):
        if job.cancel.is_set():
            raise Cancelled()
        job.progress("running", done, total)

    args = SimpleNamespace(input=str(input_path), output=str(output_path), url=runner.base_url + "/v1/systemone",
                           concurrency=concurrency, early_exit=early_exit, headers=runner._headers(), progress=progress)
    summary = asyncio.run(run.run(args))
    return {**summary, "input": str(input_path), "output": str(output_path), "model": model_id}


def rows(output_path, offset=0, limit=100) -> dict:
    """A page of a batch output file: {"rows", "total"}."""
    lines = [line for line in Path(output_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"rows": [json.loads(line) for line in lines[offset:offset + limit]], "total": len(lines)}
