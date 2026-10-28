# MCP: basal as a tool for agents

*Experimental.* With the `basal[mcp]` extra, `--mcp on` also serves the decisions over the
[Model Context Protocol](https://modelcontextprotocol.io) at `/mcp` (streamable HTTP, stateless), next to
`/v1/systemone` and through the same queue. An agent host (Claude Code, Claude Desktop, Cursor, an agent SDK) then
calls basal as a tool:

- `decide`: one `choice`, `noul`, `score` or `multi` question about a state (`state`, `question`, `type`, `options`,
  `lang`). Returns that question's answer as in the HTTP API, plus `model`.
- `decide_many`: several questions over one state in one pass, every question type, including `act` with
  `max_error` and `tool` (`state`, `questions` as in the HTTP API, `lang`, `facts`). Returns the HTTP API response.

Both tools are read-only. The server's MCP instructions give the agent the rules of [how to ask](API.md#how-to-ask), and an
invalid request (for example a question with one option) returns a tool error with the server's message.

```bash
uv pip install "basal[fp8,mcp] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --host 127.0.0.1 --port 8000 --mcp on
claude mcp add --transport http basal http://127.0.0.1:8000/mcp     # Claude Code; other hosts take the same URL
```

## Several models

When the MCP server is given a callable that lists its loaded models, it adds a `list_models` tool and an optional
`model` argument to `decide` and `decide_many`. Pass a listed model id to route a decision; omitting it leaves model
selection to the server. `basal-serve --mcp on` remains a single-model server.

For a stdio bridge to a multi-model API server, pass the selected model on the bridge command:

```json
{"mcpServers": {"basal": {"command": "/home/me/basal-env/bin/basal-mcp", "args": ["--url", "http://127.0.0.1:8000", "--model", "Remek/model"]}}}
```

`basal-mcp --model NAME` adds `NAME` to every forwarded request. Without it, `basal-mcp` continues to use the first
model reported by `GET /v1/models` as its displayed model name and leaves request selection unchanged.

For hosts that start MCP servers as local processes, `basal-mcp` is a stdio bridge to a running server (use the full
path of `basal-mcp` in your environment):

```json
{"mcpServers": {"basal": {"command": "/home/me/basal-env/bin/basal-mcp", "args": ["--url", "http://127.0.0.1:8000"]}}}
```

The MCP endpoint has no authentication, like the HTTP API. Bind `--host 127.0.0.1` for agents on the same machine
(the Host and Origin headers are then checked against DNS rebinding), or put the server behind your own gateway.
