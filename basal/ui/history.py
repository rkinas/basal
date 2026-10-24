"""The decisions basal-ui answered (playground, API, MCP) with their requests and responses, newest kept: a JSONL file
<settings stem>.history.jsonl next to the settings file (0600; media bytes are replaced by their size)."""
import json
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from .stats import answer_items, best_probability

PREVIEW = 160


def _strip_media(value):
    """Media bytes -> their size: `data` fields of basal media and data: URLs (OpenAI input_image.image_url)."""
    if isinstance(value, dict):
        return {k: (f"<{len(v)} bytes>" if k == "data" and isinstance(v, str) else _strip_media(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_media(v) for v in value]
    if isinstance(value, str) and value.startswith("data:") and len(value) > 64:
        return f"<{len(value)} bytes>"
    return value


def _best(answer: dict):
    for key in ("choice", "action", "tool", "label", "score"):
        if key in answer and answer[key] is not None:
            return str(answer[key])
    for key in ("selected", "labels", "tools"):  # multi, multi-label classify, multi tool
        if isinstance(answer.get(key), list):
            return ", ".join(map(str, answer[key]))
    for key in ("noul", "probability"):  # systemone noul, decisions-API predicate: the probability of yes
        if isinstance(answer.get(key), (int, float)):
            return "true" if answer[key] >= 0.5 else "false"
    return None


def _preview(body: dict) -> str:
    state = body.get("state", body.get("input", ""))
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return text[:PREVIEW]


def make_record(model, source, body, status, response_json, latency_ms) -> dict:
    answers = response_json.get("answers") if isinstance(response_json, dict) else None
    usage = response_json.get("usage") if isinstance(response_json, dict) else None
    questions = body.get("questions")
    items = answer_items(answers)
    return {"id": "h_" + uuid.uuid4().hex, "ts": time.time(), "model": model, "source": source, "status": status,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
            "questions": len(questions) if isinstance(questions, (dict, list)) else None,
            "state_preview": _preview(body),
            "answers": {name: {"type": a.get("type"), "best": _best(a), "confidence": best_probability(a)}
                        for name, a in items},
            "deferred": any(a.get("deferred") is True for _, a in items),
            "truncated": bool(usage.get("truncated")) if isinstance(usage, dict) else False,
            "request": _strip_media(body), "response": response_json}


def _summary(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in ("request", "response")}


class History:
    """Appends go to the file under the settings lock; a rotation rewrites the file from its own last `keep` lines,
    so a second basal-ui process on the same settings file loses nothing (its records are in the file, not in this
    process's memory, which only shows what this process recorded or loaded at startup)."""

    def __init__(self, path: Path, keep: int = 2000):
        self.path = Path(path)
        self.keep = keep
        self.lock = threading.Lock()
        self.records: deque = deque(maxlen=keep)
        self.lines = 0
        for record in self._read_file():
            self.records.append(record)

    def _read_file(self) -> list:
        """Every valid record in the file, counting its lines (corrupt ones included) into self.lines."""
        out = []
        self.lines = 0
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                self.lines += 1
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("id"), str):
                    out.append(record)
        except OSError:
            pass
        return out

    def append(self, record: dict) -> None:
        from . import settings

        line = json.dumps(record, ensure_ascii=False) + "\n"

        def write():
            if self.lines + 1 > self.keep + 500:
                kept = [*self._read_file(), record][-self.keep:]
                settings._write_private(self.path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
                self.lines = len(kept)
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(line)
            self.lines += 1

        with self.lock:
            self.records.append(record)
            settings._locked(write)()

    def list(self, model=None, source=None, q=None, limit=100, before=None, after=None) -> list[dict]:
        with self.lock:
            records = list(self.records)
        ids = [r["id"] for r in records]
        if before is not None:
            records = records[:ids.index(before)] if before in ids else []
        if after is not None:
            records = records[ids.index(after) + 1:] if after in ids else records
        needle = q.lower() if q else None
        out = []
        for record in reversed(records):
            if model and record.get("model") != model or source and record.get("source") != source:
                continue
            if needle and needle not in (str(record.get("state_preview", "")) + " " +
                                         str(record.get("model", ""))).lower():
                continue
            out.append(_summary(record))
            if len(out) >= limit:
                break
        return out

    def get(self, record_id: str) -> dict | None:
        with self.lock:
            return next((r for r in self.records if r["id"] == record_id), None)

    def models(self) -> "list[str]":  # `list` here is the method above
        with self.lock:
            return sorted({r["model"] for r in self.records if isinstance(r.get("model"), str)})

    def clear(self) -> int:
        from . import settings

        with self.lock:
            n = len(self.records)
            self.records.clear()
            settings._locked(lambda: settings._write_private(self.path, ""))()
            self.lines = 0
            return n
