"""MCP for agents: the typed decisions of a basal server as two tools of a Model Context Protocol server.

- `basal-serve ... --mcp on`: the server also answers MCP (streamable HTTP) at /mcp, on its own port and through the
  same queue as /v1/systemone.
- `basal-mcp --url http://host:8000`: a stdio bridge to a running server, for agent hosts that start MCP servers as
  local processes.

Tools: `decide` (one question) and `decide_many` (several questions over one state, every question type of the HTTP
API). Both are read-only. Needs the optional extra: `uv pip install "basal[mcp]"`."""
import argparse
from typing import Annotated, Any, Literal

INSTALL = 'MCP needs the mcp package: uv pip install "basal[mcp]"'

HOW = """basal ({model}) answers typed questions about a state in one forward pass, without generating text, and \
returns calibrated probabilities. Use it for routing, yes/no checks, scores and label sets when you want a fast, \
repeatable answer with a confidence you can act on.

How to ask:
- Put every fact and rule the decision needs in `state`: the model decides only from what it reads.
- One condition per yes/no question; for several labels use type "multi".
- Describe what each option means, not only its name; offer a "cannot be determined" option when the state may not \
contain the answer.
- Ask in the language of the state (Polish or English); compute numbers and dates yourself and ask the decision on top.
- Several questions about the same state: one `decide_many` call is faster than separate calls and gives the same \
answers.

Using the answer: `confidence` is the calibrated probability of the returned answer. Below the confidence your task \
needs, check the case yourself or ask the user. 2-255 options per question (basal-1.x models: 2-10)."""

def available():
    """None when the mcp package imports, else the install hint (checked before a model is loaded)."""
    try:
        import mcp.server.mcpserver  # noqa: F401
    except ImportError:
        return INSTALL
    return None


def build(decide, model="basal"):
    """decide: async request body -> response (Server.decide, or `remote` for the bridge). -> the MCP server."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
    from pydantic import Field  # comes with mcp

    State = Annotated[str | dict[str, Any], Field(description="The text (or a JSON object) the decision is based on, "
                                                              "with every fact and rule it needs.")]
    Lang = Annotated[Literal["pl", "en"] | None, Field(description="The prompt language, when the state text does not "
                                                                   "show it (default: Polish if the state or question "
                                                                   "has Polish diacritics, else English).")]
    from . import __version__

    srv = MCPServer(name="basal", version=__version__, instructions=HOW.format(model=model), log_level="WARNING")
    ro = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    async def ask(body):
        try:
            return await decide(body)
        except Exception as e:  # noqa: BLE001
            raise ToolError(str(e)) from e  # the message of the HTTP API's error, so the agent can fix its call

    @srv.tool(name="decide", annotations=ro)
    async def decide_one(
            state: State,
            question: Annotated[str, Field(description="The question: one condition, phrased for the state.")],
            type: Annotated[Literal["choice", "noul", "score", "multi"],
                            Field(description="choice: one of the options; noul: probability of yes; score: an "
                                              "ordered level; multi: every option that applies.")] = "choice",
            options: Annotated[dict[str, str | None] | list[str] | None,
                               Field(description="choice / multi: key -> what the option means (or a list of keys); "
                                                 "score: the levels, lowest first; noul: optional {\"true\": ..., "
                                                 "\"false\": ...}.")] = None,
            lang: Lang = None) -> dict[str, Any]:
        """Answer one typed question about a state with calibrated probabilities. Returns the answer (choice, noul,
        score or selected), its probabilities and confidence, and the model name."""
        q: dict[str, Any] = {"type": type, "instructions": question}
        if options is not None:
            q["criteria"] = options
        body: dict[str, Any] = {"state": state, "questions": {"answer": q}}
        if lang:
            body["lang"] = lang
        r = await ask(body)
        return {"model": r.get("model"), **r["answers"]["answer"]}

    @srv.tool(annotations=ro)
    async def decide_many(
            state: State,
            questions: Annotated[dict[str, dict[str, Any]],
                                 Field(description="name -> question, as in the basal HTTP API: {\"type\": \"choice\" "
                                                   "| \"noul\" | \"score\" | \"multi\" | \"act\" | \"tool\", "
                                                   "\"instructions\": ..., \"criteria\": ...} plus the type's fields "
                                                   "(choice / noul / score: coverage, max_error; multi: threshold, "
                                                   "min, max, objective; act: costs, max_error; tool: tools, extra, "
                                                   "rank, max_error; any: option_keys).")],
            lang: Lang = None,
            facts: Annotated[Literal["off", "auto"] | None,
                             Field(description="auto: append computed calendar and money facts to a Polish "
                                               "state.")] = None
    ) -> dict[str, Any]:
        """Answer several typed questions about one state in one pass (every question type of the basal HTTP API,
        including cost-aware actions with deferral to a person, and the next tool of an agent). Returns the API
        response: answers by question name, the model name and usage."""
        body: dict[str, Any] = {"state": state, "questions": questions}
        for k, v in (("lang", lang), ("facts", facts)):
            if v:
                body[k] = v
        return await ask(body)

    return srv


def http(server, host):
    """--mcp on: the MCP tools of a running basal server -> (routes for /mcp, the session manager's lifespan).
    Stateless JSON responses: every call is one request. Bound to a loopback host, Host / Origin are checked (DNS
    rebinding)."""
    m = build(getattr(server, "serve", server.decide), getattr(server, "name", None) or "basal")
    app = m.streamable_http_app(host=host, stateless_http=True, json_response=True)
    return app.routes, m.session_manager.run


def remote(url, timeout=60.0, transport=None):
    """decide over HTTP for the bridge: POST <url>/v1/systemone; the server's error message on HTTP errors."""
    import httpx

    client = httpx.AsyncClient(base_url=url.rstrip("/"), timeout=timeout, transport=transport)

    async def decide(body):
        r = await client.post("/v1/systemone", json=body)
        if r.status_code != 200:
            try:
                msg = r.json()["error"]
            except Exception:  # noqa: BLE001
                msg = r.text
            raise ValueError(f"basal server {r.status_code}: {msg}")
        return r.json()

    return decide


def model_name(url, timeout=5.0):
    """The served model's name from GET /v1/models; "basal" when the server does not answer yet."""
    import httpx

    try:
        return httpx.get(url.rstrip("/") + "/v1/models", timeout=timeout).json()["models"][0]["name"]
    except Exception:  # noqa: BLE001
        return "basal"


def main():
    ap = argparse.ArgumentParser(description="basal over MCP (stdio): a bridge to a running basal server for agent "
                                             "hosts that start MCP servers as local processes")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="the basal server (basal-serve)")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    a = ap.parse_args()
    msg = available()
    if msg:
        raise SystemExit(msg)
    build(remote(a.url, a.timeout), model_name(a.url)).run("stdio")


if __name__ == "__main__":
    main()
