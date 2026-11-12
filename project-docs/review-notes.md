# Release-readiness review

Reviewed and completed: 12 November 2026

The implementation and release-validation work described in the prior review is complete. The package is ready to publish as `0.1.0`; publishing itself remains an intentionally separate owner action.

## Resolved implementation findings

| Finding | Resolution | Verification |
| --- | --- | --- |
| Streaming exceptions could end an already-started SSE response without an error event. | The SSE generator now converts unexpected exceptions to the documented safe `data: {"error": ...}` event, closes the upstream generator, preserves cancellation, and never sends `[DONE]` after failure. | HTTP regression test covers partial output followed by an unexpected failure. |
| `CSA_MODELS` accepted JSON arrays and invalid aliases, which could fail later as a 500. | Startup now rejects malformed JSON, non-object values, and empty/non-string alias or model values with a clear `CSA_MODELS` error. `CSA_PORT` parsing now also has a clear configuration error. | Configuration regression tests cover each invalid shape. |
| Required request lifecycle fields were not logged. | One privacy-safe completion record now includes request ID, requested model, queue wait, execution duration, total request duration, and outcome. Prompts, tool payloads, credentials, and subprocess environments are not logged. | HTTP regression test asserts the fields and an unexpected-stream outcome. |
| Test lint debt remained. | Imports, unused symbols, encoding call, successive-pair iteration, and formatting have been corrected. | `uv run ruff check src tests` and `uv run ruff format --check src tests` pass. |
| Alternate provider credentials/routing could leak into the CLI child. | The child environment now strips `ANTHROPIC_*`, cloud-provider credential prefixes (`AWS_`, `AZURE_`, `GOOGLE_`, `VERTEX_`), and Claude Code Bedrock, Vertex, Foundry, and related auth-routing switches. | Transport regression test confirms those values are absent while safe environment values remain. |

## Release evidence

- Full test suite: `414 passed` on Python 3.13.0 and Python 3.11.16.
- Lint and format: clean for `src` and `tests`.
- Distribution: `uv build` produced wheel and source distribution successfully.
- Clean installed artifact: the wheel installed into a fresh Python 3.11 environment; the console command and bundled MCP helper imported successfully.
- Local CLI: Claude Code `2.1.296` was authenticated successfully.
- Live local-server validation passed using the authenticated CLI:
  - OpenAI Python SDK synchronous completion;
  - OpenAI Python SDK asynchronous SSE completion;
  - complete OpenAI function-tool result round trip;
  - runnable Python, LangChain, and LangGraph tool-loop examples;
  - LangChain `invoke`, `stream`, `ainvoke`, and `astream`;
  - LangGraph tool execution plus checkpoint continuation in one thread.
- Package-name availability: PyPI returned `404` for `claude-subscription-adapter` on 12 November 2026.

## Release boundaries

The documented compatibility boundaries are deliberate product limits, not unfinished work: sampling and structured-output controls are ignored; requested token limits can raise but not lower the internal cap; required/named tool choice and `parallel_tool_calls=false` are not enforced; strict tool schemas are not guaranteed. Windows is not certified. The package is intentionally not yet published to PyPI.
