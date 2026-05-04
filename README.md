# pier

Pier is a [Harbor](https://www.harborframework.com/docs/tasks)-compatible framework for evaluating coding agents in sandboxed environments. It reads Harbor's task format and runs trials against it.

```bash
pier run -p path/to/task --agent claude-code --env modal
```

## Why pier

Pier is a fork. We wanted a smaller, more opinionated base to build on, starting with one thing Harbor couldn't do: run installed agents (Claude Code, Codex, etc.) in air-gapped tasks (`allow_internet = false`).

When the agent runs *inside* the sandbox, both the install step and the inference call need the network. Pier fixes that by letting agents declare their install scripts and a network allowlist, which environments then honor when setting up the sandbox.

More is planned from here, and pier will keep drifting from Harbor as we go.

## What works today

- **Task format:** Harbor-compatible.
- **Environments:** `docker`, `modal`.
- **Agents:** `nop`, `oracle`, `claude-code`, `codex`, `gemini-cli`, `opencode`, `mini-swe-agent`.
- **Datasets:** local Harbor-format task directories via `-p` / `--path`.

Pier does not currently resolve or download Harbor registry datasets directly. Use Harbor to export a dataset first, then point Pier at the local directory.

If your tasks allow internet, or your agent runs outside the sandbox, Harbor is the broader, more mature choice and you should probably use it.

## Install

```bash
uv tool install pier
# or
pip install pier
```

## Run

```bash
export ANTHROPIC_API_KEY=...
pier run -p path/to/task --agent claude-code --env modal --env-file .env
```

Run a local dataset:

```bash
pier run -p path/to/dataset --agent claude-code --env modal
```

Run a deterministic random subset of a local dataset:

```bash
pier run -p path/to/dataset --n-tasks 10 --sample-seed 0
```

### Agent Runtime Configuration

Pier does not have a shared inference routing abstraction. Use
`agent.model_name` for trial metadata, `agent.env` for runtime environment
variables, and agent-specific `kwargs` for tool config.

For gateway providers, configure the installed tool directly. For example,
Claude Code can use an Anthropic-compatible gateway with:

```yaml
agents:
  - name: claude-code
    model_name: anthropic/claude-opus-4-7
    env:
      ANTHROPIC_AUTH_TOKEN: ${RESPAN_API_KEY}
      ANTHROPIC_BASE_URL: https://endpoint.respan.ai/api/anthropic/
      ANTHROPIC_CUSTOM_HEADERS: "X-Respan-Route-Provider: vertex_ai"
    kwargs:
      reasoning_effort: max
```

OpenCode exposes `opencode_config` for provider entries that Pier does not know
about:

```yaml
agents:
  - name: opencode
    model_name: respan-gemini/gemini-3.1-pro-preview
    env:
      GEMINI_API_KEY: ${RESPAN_API_KEY}
    kwargs:
      opencode_config:
        provider:
          respan-gemini:
            npm: ai-sdk-provider-gemini-cli
            name: Respan Gemini
            options:
              authType: api-key
              baseURL: https://endpoint.respan.ai/api/google/vertexai/
              apiKey: "{env:GEMINI_API_KEY}"
            models:
              gemini-3.1-pro-preview:
                name: Gemini 3.1 Pro Preview
```

Codex exposes `command_model_name`, `config_toml`, and `config_toml_file` kwargs
for CLI-specific configuration. mini-swe-agent exposes `config_yaml` and
`config_file` kwargs for mini-swe-agent config overrides.

To use a Harbor registry dataset, download it with Harbor first. For example, from `~/code/harbor`:

```bash
uv run harbor download swebenchpro -o ~/code/pier/datasets
```

Then run it with Pier:

```bash
cd ~/code/pier
uv run pier run -p datasets/swebenchpro --n-tasks 10 --sample-seed 0
```

See `pier run --help` and `pier job --help` for everything else. Trials land under `jobs/<timestamp>/<trial_id>/`.
