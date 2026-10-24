"""Thin tools for agent frameworks over a running basal server's HTTP API (2.5). Each returns the framework's own tool
object that calls /v1/systemone: give it a state and questions (the HTTP API's), or fix a preset (basal/presets.py)
so the agent only passes the state. The frameworks are not dependencies of basal: install the one you use. For MCP
hosts use basal-mcp / --mcp on instead (basal/mcp_server.py).

  from basal.adapters import langchain_tool, llamaindex_tool, crewai_tool
  tool = langchain_tool("http://127.0.0.1:8000")                    # args: state, questions
  triage = crewai_tool("http://127.0.0.1:8000", preset="triage")    # args: state"""
import json

NAME = "basal_decide"
DESCRIPTION = ("Typed decisions about a text (state) with calibrated probabilities, one forward pass, no generated "
               "text. Put every fact the decision needs in the state. questions: {name: {type: choice | noul | score | "
               "multi, instructions: the question, criteria: {key: what the option means}}}.")


def decide_fn(url="http://127.0.0.1:8000", preset=None, lang=None, client=None):
    """The function every adapter wraps: (state, questions=None) -> the answers as a dict (JSON-serialisable)."""
    from .client import Basal
    b = client or Basal(url)

    def basal_decide(state: str, questions: dict | None = None) -> dict:
        if isinstance(questions, str):  # agents often pass JSON text
            questions = json.loads(questions)
        if preset:
            return b.preset(state, preset, lang)
        if not questions:
            raise ValueError("questions are needed (or create the tool with a preset)")
        return b.decide(state, questions)
    return basal_decide


def _describe(preset):
    return f"{DESCRIPTION} This tool asks the '{preset}' preset: pass the state only." if preset else DESCRIPTION


def _need(module, extra):
    raise ImportError(f"this adapter needs {module}: uv pip install {extra}")


def langchain_tool(url="http://127.0.0.1:8000", preset=None, lang=None, client=None):
    """A LangChain StructuredTool (langchain-core)."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        _need("langchain-core", "langchain-core")
    return StructuredTool.from_function(func=decide_fn(url, preset, lang, client), name=NAME,
                                        description=_describe(preset))


def llamaindex_tool(url="http://127.0.0.1:8000", preset=None, lang=None, client=None):
    """A LlamaIndex FunctionTool (llama-index-core)."""
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError:
        _need("llama-index-core", "llama-index-core")
    return FunctionTool.from_defaults(fn=decide_fn(url, preset, lang, client), name=NAME,
                                      description=_describe(preset))


def crewai_tool(url="http://127.0.0.1:8000", preset=None, lang=None, client=None):
    """A CrewAI BaseTool."""
    try:
        from crewai.tools import BaseTool
    except ImportError:
        _need("crewai", "crewai")
    fn = decide_fn(url, preset, lang, client)

    class BasalDecide(BaseTool):
        name: str = NAME
        description: str = _describe(preset)

        def _run(self, state: str, questions: dict | None = None) -> str:
            return json.dumps(fn(state, questions), ensure_ascii=False)
    return BasalDecide()
