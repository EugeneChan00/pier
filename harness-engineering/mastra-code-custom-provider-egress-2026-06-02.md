# Mastra Code Custom Provider Egress - 2026-06-02

## Symptom

Mastra Code GPT-5.5 xhigh Pier dry runs failed before tool execution on
`prometheus-typed-label-sorting`.

The first custom-provider attempt fixed Mastra's endpoint/model validation but
still failed in Docker filtered egress:

- Mastra used `/v1/chat/completions` with model `gpt-5.5(xhigh)`.
- The agent was in `build` mode.
- No `tool_start` events occurred.
- The API call failed with `connect ENETUNREACH ...:443`.

Expected behavior: Mastra should use CLI Proxy's chat-completions endpoint,
select `gpt-5.5(xhigh)` as the proxy-visible model, keep harness effort at
`thinking_level=xhigh`, and execute tools before any verifier decision.

## Reproduction

Run from the Pier repository with `CLI_PROXY_API_KEY` injected:

```bash
uv run pier run \
  --config jobs/mastra-gpt55-xhigh-prometheus-typed-label-sorting-hostmap/config.json \
  --job-name mastra-gpt55-xhigh-prometheus-typed-label-sorting-proxy-agent
```

The live run used Infisical from the agent-toolkit project:

```bash
infisical run --path=/agents \
  --project-config-dir=/Users/zzmc/container/shared/projects/agent-toolkit \
  -- bash -lc 'cd /Users/zzmc/container/shared/references/bench/pier && uv run pier run --config jobs/mastra-gpt55-xhigh-prometheus-typed-label-sorting-hostmap/config.json --job-name mastra-gpt55-xhigh-prometheus-typed-label-sorting-proxy-agent'
```

## Findings

Mastra Code's built-in `openai/*` route uses OpenAI Responses. Mastra custom
providers use OpenAI-compatible chat completions.

Pier now configures Mastra custom provider settings with:

- provider id: `cli-proxy`
- provider model: `cli-proxy/gpt-5.5(xhigh)`
- proxy-visible model: `gpt-5.5(xhigh)`
- provider URL: `http://127.0.0.1:18766/v1`
- upstream URL: `https://aa.renaissancelab.org/v1`

Docker filtered egress intentionally keeps the main container on an internal
network. Agent commands receive `HTTP_PROXY`/`HTTPS_PROXY`, but Node fetch
clients do not reliably honor those variables. A direct Mastra provider URL to
`https://aa.renaissancelab.org/v1` can therefore fail with `ENETUNREACH`.

The local Mastra shim must forward through Pier's authenticated Squid proxy.
The first CONNECT implementation reached the proxy env but used a Node
`https.request` handoff that still opened a direct socket. A custom one-shot
`https.Agent.createConnection` handoff now binds the upstream request to the
CONNECT tunnel.

## Observability Added

`agent/mastra-code-provider.log` records custom provider settings without
secrets:

```json
{"type":"custom_provider_settings","provider":"cli-proxy","upstream":"https://aa.renaissancelab.org/v1","providerUrl":"http://127.0.0.1:18766/v1","model":"gpt-5.5(xhigh)","providerModel":"cli-proxy/gpt-5.5(xhigh)"}
```

`agent/mastra-code-shim.log` records upstream routing without credentials:

```json
{"type":"upstream_request","viaProxy":true,"targetHost":"aa.renaissancelab.org","path":"/v1/chat/completions","model":"gpt-5.5(xhigh)"}
```

`agent/mastra-code.txt` exposes Mastra stream events, including `tool_start`
and `tool_end`.

## Detector

Run against a Pier trial directory:

```bash
harness-engineering/detect-mastra-code-cli-proxy-custom-provider.py \
  jobs/mastra-gpt55-xhigh-prometheus-typed-label-sorting-proxy-agent/prometheus-typed-label-sorting__AGko8FK \
  --model 'gpt-5.5(xhigh)'
```

Healthy output includes local provider URL, `/v1/chat/completions`,
`viaProxy=true`, and nonzero tool execution counts.

## Verification

Focused tests:

```bash
uv run --with pytest pytest tests/test_mastra_code.py
```

Result: `14 passed`.

Detector result on
`jobs/mastra-gpt55-xhigh-prometheus-typed-label-sorting-proxy-agent/prometheus-typed-label-sorting__AGko8FK`:

- `provider_records=1`
- `upstream_requests=5`
- tool calls include `task_write`, `find_files`, `search_content`, and `view`
- `errors=0`

End-to-end trial result:

- job: `jobs/mastra-gpt55-xhigh-prometheus-typed-label-sorting-proxy-agent`
- trial: `prometheus-typed-label-sorting__AGko8FK`
- reward: `0.0`
- exception: `NonZeroAgentExitCodeError`
- `model.patch`: empty

This is not a completed benchmark task. It proves that the Mastra harness now
routes through custom-provider chat completions and executes tools, but it also
exposes a separate Mastra runtime/exit observability issue: the stream stopped
mid-session after broad search output without a Mastra `error` or `agent_end`
event, and the CLI exited 1.
