"""A triage webhook: a helpdesk, n8n, Make or a mail gateway POSTs a ticket, and gets back the queue it goes to.

  BASAL_URL=http://127.0.0.1:8000 uvicorn app:app --app-dir examples/triage --port 9000
  curl -s localhost:9000/ticket -H 'content-type: application/json' -d '{"id": "123", "text": "Pobrano dwa razy 129 zł"}'

The answer: {"id", "queue", "priority", "category", "urgency", "needs_human", "automated", "answers"}.
- queue: the category when its answer is certified at MAX_ERROR (it did not abstain), else "human";
- priority: "high" for the top urgency level, else "normal";
- needs_human: the yes/no answer as true / false, or null when it abstains;
- automated: true when no answer abstained, so nobody needs to look at the ticket before it is routed.

Environment: BASAL_URL (http://127.0.0.1:8000), BASAL_API_KEY, MAX_ERROR (0.05; "" turns it off), LANG_PRESET (pl).
"""
import os
import sys
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).parent))
from run import predicted, triage_questions  # noqa: E402

from basal.client import AsyncBasal, BasalError  # noqa: E402


def route(answers):
    """The queue and priority of a ticket from its triage answers."""
    cat, urg, human = answers["category"], answers["urgency"], answers["needs_human"]
    return {
        "queue": "human" if cat.get("abstain") else cat["choice"],
        "priority": "high" if predicted(urg) == len(urg["probabilities"]) - 1 else "normal",
        "category": cat["choice"],
        "urgency": urg["legend"][str(predicted(urg))].split(":")[0],  # the level name, without the RULES text
        "needs_human": None if human.get("abstain") else predicted(human) == "yes",
        "automated": not any(a.get("abstain") for a in answers.values()),
    }


def make_app(basal=None, max_error=None, lang=None):
    max_error = max_error if max_error is not None else os.environ.get("MAX_ERROR", "0.05")
    max_error = float(max_error) if max_error not in ("", None) else None
    lang = lang or os.environ.get("LANG_PRESET", "pl")
    basal = basal or AsyncBasal(os.environ.get("BASAL_URL", "http://127.0.0.1:8000"),
                                api_key=os.environ.get("BASAL_API_KEY"), retries=2)

    async def ticket(request):
        body = await request.json()
        text = (body.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=422)
        try:
            answers = await basal.decide(text, triage_questions(text, lang, max_error, rules=True))
        except BasalError as e:
            return JSONResponse({"error": str(e)}, status_code=502)
        return JSONResponse({"id": body.get("id"), **route(answers), "answers": answers})

    return Starlette(routes=[Route("/ticket", ticket, methods=["POST"])])


app = make_app()
