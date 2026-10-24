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
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --host 127.0.0.1 --port 8000 --mcp on \
  --api-keys ./api-keys.json
claude mcp add --transport http basal http://127.0.0.1:8000/mcp     # Claude Code; other hosts take the same URL
```

For a streamable-HTTP client, pass the API key in the `Authorization` header when the server uses `--api-keys`:

```json
{
  "mcpServers": {
    "basal": {
      "url": "http://127.0.0.1:8000/mcp",
      "headers": {"Authorization": "Bearer <secret>"}
    }
  }
}
```

`X-API-Key: <secret>` is also accepted. The key file format and `/v1/keys` admin endpoint are documented in
[Serving](SERVING.md#api-access). When basal-serve is managed by basal-ui, its **API server → Connect → MCP** panel
shows the streamable-HTTP URL and header configuration.

For hosts that start MCP servers as local processes, `basal-mcp` is a stdio bridge to a running server (use the full
path of `basal-mcp` in your environment). Its `--api-key` option defaults to `BASAL_API_KEY` and authenticates both
the model lookup and decision requests:

```json
{"mcpServers": {"basal": {"command": "/home/me/basal-env/bin/basal-mcp", "args": ["--url", "http://127.0.0.1:8000", "--api-key", "<secret>"]}}}
```

**basal-ui's API server** serves several models on one port: there `decide` and `decide_many` take a required
`model` argument (a loaded model id) and the `list_models` tool names the loaded models. The bridge sends one model
with every request: add `"--model", "Remek/basal-2.0-mini-MLX-8bit"` to `args` (without it the API server answers
`400 MODEL_REQUIRED`). Details: [API.md](API.md#several-models-behind-basal-ui).

Bind `--host 127.0.0.1` for agents on the same machine (the Host and Origin headers are then checked against DNS
rebinding). To expose the server on another interface, use `--api-keys`; without it, the HTTP API and MCP endpoint are
unauthenticated.
