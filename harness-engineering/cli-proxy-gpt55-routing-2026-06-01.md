# CLI Proxy GPT-5.5 Routing Suffix Check - 2026-06-01

## Symptom

Pier benchmark setup needed to distinguish GPT-5.5 low and xhigh variants for Claude Code and Mastra Code while still routing through CLI Proxy.

The ambiguous point was whether `low`/`xhigh` belongs in the endpoint model id, for example `gpt-5.5-(xhigh)`, or in a harness/request effort field.

## Reproduction

Run from the Pier repository with `CLI_PROXY_API_KEY` injected:

```bash
harness-engineering/detect-cli-proxy-gpt55-routing.sh
```

Or with Infisical from the agent-toolkit repository:

```bash
infisical run --path /agents -- sh -c 'cd /Users/zzmc/container/shared/references/bench/pier && harness-engineering/detect-cli-proxy-gpt55-routing.sh'
```

## Findings

`/v1/models` advertises the base endpoint model `gpt-5.5`.

CLIProxyAPI supports parenthesized model-name thinking suffixes:

- `gpt-5.5(low)` is accepted.
- `gpt-5.5(xhigh)` is accepted.
- `gpt-5.5(max)` is accepted on the Anthropic-compatible route.
- `gpt-5.5(max)` is rejected on OpenAI Responses because that route supports `low`, `medium`, `high`, and `xhigh`.
- Hyphenated forms such as `gpt-5.5-(low)` and `gpt-5.5-(xhigh)` are rejected.

The live proxy response reports the upstream model as `gpt-5.5` after accepting a parenthesized suffix.

## Harness Gap

Pier previously documented request-level effort and proxy model selection without naming the suffix syntax. That made it easy to confuse:

- CLI model id: `gpt-5.5`
- CLIProxyAPI suffix selector: `gpt-5.5(xhigh)`
- Harness/request effort: Claude Code `reasoning_effort`, Mastra Code `thinking_level`

CLIProxyAPI source gives suffixes priority over body/request effort. Harness effort and model effort are decoupled, so benchmark configs must set both deliberately.

Live Pier dry runs exposed two additional harness gaps:

- Claude Code sends Anthropic `system` payloads that CLI Proxy rejects with `400 {"detail":"System messages are not allowed"}`.
- Mastra Code validates `--model` locally and rejects `openai/gpt-5.5(low)` before reaching CLI Proxy, while CLI Proxy expects the endpoint model without the provider prefix: `gpt-5.5(low)`.

Mastra's post-tool OpenAI Responses follow-up can also fail with `Items are not persisted when store is set to false` after a successful tool write. Pier originally treated that as a trial exception even when the verifier reward was 1.0, and malformed/truncated final error JSON made the detector miss it.

## Mitigation

Pier README examples now use `gpt-5.5(xhigh)` for CLI Proxy model selection and document that `gpt-5.5-(xhigh)` is invalid.

Implemented harness fixes:

- Claude Code starts a local Anthropic-compatible shim for CLI Proxy and moves system text into the first user message before forwarding.
- Claude Code mutates the agent-level env escape hatch after shim setup so `BaseInstalledAgent._exec()` cannot overwrite `ANTHROPIC_BASE_URL` back to the upstream proxy.
- Mastra Code strips parenthesized suffixes before invoking the local CLI validator, then starts a local OpenAI-compatible shim that rewrites outbound request model `gpt-5.5` to `gpt-5.5(low)` or `gpt-5.5(xhigh)`.
- Mastra Code records redacted shim traces such as `{"type":"model_rewrite","from":"gpt-5.5","to":"gpt-5.5(low)"}`.
- Mastra Code suppresses only the post-successful-tool `store=false` follow-up error so Pier can let the verifier decide the result.

Passing live hello-world runs:

| Harness | Model selector | Harness effort | Job | Reward | Exceptions |
| --- | --- | --- | --- | --- | --- |
| Mastra Code | `openai/gpt-5.5(low)` | `thinking_level=xhigh` | `jobs/e2e-fix-mastra-gpt55-low-shim-trace/result.json` | `1.0` | `0` |
| Mastra Code | `openai/gpt-5.5(xhigh)` | `thinking_level=xhigh` | `jobs/e2e-fix-mastra-gpt55-xhigh-shim-trace/result.json` | `1.0` | `0` |
| Claude Code | `gpt-5.5(low)` | `reasoning_effort=max` | `jobs/e2e-fix-claude-gpt55-low-rerun/result.json` | `1.0` | `0` |
| Claude Code | `gpt-5.5(xhigh)` | `reasoning_effort=max` | `jobs/e2e-fix-claude-gpt55-xhigh/result.json` | `1.0` | `0` |

The detector script verifies the live routing behavior before benchmark rollout:

```bash
infisical run --path /agents -- sh -c 'cd /Users/zzmc/container/shared/references/bench/pier && harness-engineering/detect-cli-proxy-gpt55-routing.sh'
```
