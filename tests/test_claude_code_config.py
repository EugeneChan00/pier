from pathlib import Path

from pier.agents.installed.claude_code import ClaudeCode


def test_claude_code_preserves_explicit_model_aliases_for_proxy(tmp_path: Path):
    agent = ClaudeCode(
        logs_dir=tmp_path,
        model_name="gpt-5.5(xhigh)",
        reasoning_effort="max",
        extra_env={
            "CLI_PROXY_API_KEY": "test-key",
            "ANTHROPIC_BASE_URL": "https://aa.renaissancelab.org",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "gpt-5.5(xhigh)",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5.5(xhigh)",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5.5(xhigh)",
            "CLAUDE_CODE_SUBAGENT_MODEL": "gpt-5.5(xhigh)",
        },
    )

    env = agent._build_runtime_env()

    assert env["ANTHROPIC_BASE_URL"] == "https://aa.renaissancelab.org"
    assert env["ANTHROPIC_API_KEY"] == "test-key"
    assert env["ANTHROPIC_MODEL"] == "gpt-5.5(xhigh)"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "gpt-5.5(xhigh)"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "gpt-5.5(xhigh)"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "gpt-5.5(xhigh)"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "gpt-5.5(xhigh)"
    assert "--effort max" in agent.build_cli_flags()


def test_claude_code_cli_proxy_system_shim_rewrites_runtime_base_url(
    tmp_path: Path,
):
    agent = ClaudeCode(
        logs_dir=tmp_path,
        model_name="gpt-5.5(low)",
        reasoning_effort="max",
        extra_env={
            "CLI_PROXY_API_KEY": "test-key",
            "ANTHROPIC_BASE_URL": "https://aa.renaissancelab.org",
        },
    )

    env = agent._build_runtime_env()

    assert agent._configure_cli_proxy_system_shim_env(env)
    assert env["PIER_CLAUDE_CODE_UPSTREAM_BASE_URL"] == (
        "https://aa.renaissancelab.org"
    )
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18765"
    assert env["ANTHROPIC_MODEL"] == "gpt-5.5(low)"
    assert env["ANTHROPIC_API_KEY"] == "test-key"
    assert agent._extra_env["PIER_CLAUDE_CODE_UPSTREAM_BASE_URL"] == (
        "https://aa.renaissancelab.org"
    )
    assert agent._extra_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18765"


def test_claude_code_cli_proxy_system_shim_can_be_disabled(tmp_path: Path):
    agent = ClaudeCode(
        logs_dir=tmp_path,
        model_name="gpt-5.5(low)",
        reasoning_effort="max",
        extra_env={
            "CLI_PROXY_API_KEY": "test-key",
            "ANTHROPIC_BASE_URL": "https://aa.renaissancelab.org",
            "CLAUDE_CODE_CLI_PROXY_SYSTEM_SHIM": "0",
        },
    )

    env = agent._build_runtime_env()

    assert not agent._configure_cli_proxy_system_shim_env(env)
    assert env["ANTHROPIC_BASE_URL"] == "https://aa.renaissancelab.org"


def test_claude_code_cli_proxy_system_shim_command_uses_node():
    command = ClaudeCode._build_cli_proxy_system_shim_command()

    assert "node /tmp/pier-claude-code-cli-proxy-shim.js" in command
    assert "normalizeClaudeCodePayload" in command
    assert "requestViaProxy" in command


def test_claude_code_install_avoids_debian_nodejs_npm_conflict(tmp_path: Path):
    agent = ClaudeCode(logs_dir=tmp_path)

    root_step = agent.install_spec().steps[0]

    assert "apt-get install -y curl nodejs;" in root_step.run
    assert "apt-get install -y curl nodejs npm" not in root_step.run
