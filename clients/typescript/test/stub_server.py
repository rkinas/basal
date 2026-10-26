"""basal-serve (the real HTTP app and request handling) on the repo's stub engine, for the TypeScript client's tests.

The stub engine reads the probabilities from the weights written at the end of each option ("... ~0.7"), as the
Python tests do (tests/test_calibrate.py), so a test chooses the answer it expects. The calibration file has answer
sets and certified thresholds (tests/test_conformal.py).

  python clients/typescript/test/stub_server.py --port 8123 [--error-format openai]

Prints "ready <port>" when it listens. Run it with the interpreter of an environment that has the repo's
dependencies installed (and the repo on PYTHONPATH, as for pytest)."""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

import basal.server as server_module  # noqa: E402
from tests.test_calibrate import WEIGHT, StubBackend  # noqa: E402

ANY = re.compile(r"~([0-9.e-]+)")


class Backend(StubBackend):
    """The tests' stub engine, and for prompts whose options carry no weights (the yes/no branches of a multi
    question, the extra options of a tool question): the first "~p" anywhere in the prompt is p(the first option, or yes), the
    rest is shared equally; without one, the options are equally likely."""

    def run_shared(self, groups, policy=None):
        out = []
        for prompts, lids in groups:
            res = []
            for p, ids in zip(prompts, lids):
                text = "".join(map(chr, p))
                w = [float(x) for x in WEIGHT.findall(text)][: len(ids)]
                if len(w) < len(ids):
                    first = ANY.search(text)
                    q = float(first.group(1)) if first else 1.0 / len(ids)
                    w = [q] + [(1 - q) / (len(ids) - 1)] * (len(ids) - 1)
                    if re.search(r"^A\. No$", text, re.M):  # the yes/no branches of a multi question come in both orders
                        w = w[::-1]
                res.append([x / sum(w) for x in w])
            out.append(res)
        return out


CALIBRATION = {
    "temperature_per_prim": {"choice": 1.0, "noul": 1.0, "score": 1.0, "multi": 1.0, "act": 1.0},
    "thresholds": {"0.05": {"confidence": 0.7, "per_type": {"choice": {"confidence": 0.7},
                                                             "noul": {"confidence": 0.9}}}},
    "conformal": {"levels": {"0.9": {"per_type": {"choice": {"p_min": 0.15}, "noul": {"p_min": 0.05},
                                                   "score": {"p_min": None}}}}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--error-format", default="basal", choices=["basal", "openai"])
    ap.add_argument("--api-key", default=None, help="not checked by basal-serve; here only to reject other tokens")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    model = tmp / "model"
    model.mkdir()
    (model / "CALIBRATION.json").write_text(json.dumps(CALIBRATION), encoding="utf-8")
    server_module.resolve = lambda *_, **__: model
    server_module.EagerBackend = Backend
    srv = server_module.Server(server_module.parser().parse_args(["--mode", "eager", "--model", "basal-stub"]))
    app = server_module.http_app(srv, "eager", None, a.error_format)
    if a.api_key:  # basal-serve has no auth of its own (a proxy does it): the stub checks the bearer token
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        class Auth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.url.path != "/health" and request.headers.get("authorization") != f"Bearer {a.api_key}":
                    return JSONResponse({"error": "unauthorized"}, status_code=401)
                return await call_next(request)
        app.add_middleware(Auth)
    print(f"ready {a.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
