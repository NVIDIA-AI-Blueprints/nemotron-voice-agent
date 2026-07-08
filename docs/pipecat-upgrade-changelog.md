# Pipecat Upgrade Changelog

## Summary

- Upgraded server dependency `pipecat-ai[nvidia,silero,runner,webrtc,websocket,openai]` from `1.3.0` to `1.5.0`.
- Refreshed `uv.lock`; the `runner` extra now resolves `pipecat-ai-prebuilt==1.0.3`.
- Upgraded client RTVI packages to the versions listed in the Pipecat `v1.5.0` release notes:
  - `@pipecat-ai/client-js` `1.11.0` to `1.12.0`
  - `@pipecat-ai/client-react` `1.4.0` to `1.7.1`
  - `@pipecat-ai/small-webrtc-transport` `1.10.4` to `1.10.5`
  - `@pipecat-ai/websocket-transport` `1.6.7` to `1.7.0`

## Release Notes Reviewed

- Pipecat `v1.4.0`: https://github.com/pipecat-ai/pipecat/releases/tag/v1.4.0
- Pipecat `v1.5.0`: https://github.com/pipecat-ai/pipecat/releases/tag/v1.5.0
- PyPI latest release marker for `pipecat-ai 1.5.0`: https://pypi.org/project/pipecat-ai/

## Repo Impact

- No server import or constructor migrations were required for the current examples.
- The repo does not import the removed `WorkerParams.loop`, `RealtimeServiceMetadataFrame`, or `RealtimeServiceInfo` symbols.
- The repo does not use `pipecat_flows`, so the `pipecat-ai-flows` fold-in does not require code changes.
- The repo does not use the deprecated `WebsocketServerTransport` aliases directly.
- Existing cascade `on_user_turn_stopped` handlers remain valid; the release-note `content=None` behavior is specific to realtime-service mode.

## Client Notes

`@pipecat-ai/client-react@1.7.1` ships declarations that type `PipecatClientProvider` against an embedded local
`PipecatClient` declaration while hooks refer to `client-js`. `client/tsconfig.json` maps `client-js` to the installed
`@pipecat-ai/client-js` declarations, and `client/src/App.tsx` narrows the unavoidable provider-boundary cast to
`ComponentProps<typeof PipecatClientProvider>["client"]`.

## Validation

- `PYTHONPATH=src uv run python -c "import src.server"`
- `PYTHONPATH=src uv run python -c "from examples.generic import pipeline; from examples.multilingual import pipeline"`
- `PYTHONPATH=src uv run python -c "from examples.omni_assistant import pipeline; from examples.frontend_backend_agent import pipeline; from examples.omni_assistant_subagents import pipeline"`
- `.venv/bin/ruff format --check .`
- `.venv/bin/ruff check .`
- `PYTHONPATH=src .venv/bin/pytest tests/ -x -q`
- `cd client && npm run lint`
- `cd client && npm run build`

## Remaining Runtime Validation

End-to-end WebRTC/WebSocket runtime validation was not run because it requires a configured deployment environment and
service credentials. The static imports, tests, lint, and client build pass under `pipecat-ai==1.5.0`.
