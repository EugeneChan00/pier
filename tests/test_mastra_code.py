from pathlib import Path

from pier.agents.factory import AgentFactory
from pier.agents.installed.mastra_code import MastraCode
from pier.models.agent.name import AgentName


def test_mastra_code_is_registered(tmp_path: Path):
    agent = AgentFactory.create_agent_from_name(
        AgentName.MASTRA_CODE,
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
    )

    assert isinstance(agent, MastraCode)
    assert agent.name() == "mastra-code"


def test_mastra_code_install_spec_uses_headless_cli(tmp_path: Path):
    agent = MastraCode(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5(xhigh)",
        thinking_level="high",
        version="0.15.2",
    )

    spec = agent.install_spec()

    assert spec.agent_name == "mastra-code"
    assert any("npm install -g mastracode@0.15.2" in step.run for step in spec.steps)
    assert spec.verification_command is not None
    assert "mastracode --help" in spec.verification_command
    assert "--thinking-level high" in agent._build_headless_flags()
    assert "--model openai/gpt-5.5" in agent._build_headless_flags()


def test_mastra_code_proxy_suffix_maps_to_runtime_thinking_level(tmp_path: Path):
    agent = MastraCode(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5(low)",
        thinking_level="xhigh",
    )

    assert agent._runtime_model_and_thinking_level() == ("openai/gpt-5.5", "xhigh")
    assert agent._proxy_target_model_name() == "gpt-5.5(low)"
    assert "--thinking-level xhigh" in agent._build_headless_flags()
    assert "--model openai/gpt-5.5" in agent._build_headless_flags()


def test_mastra_code_proxy_suffix_configures_model_shim(tmp_path: Path):
    agent = MastraCode(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5(low)",
        thinking_level="xhigh",
        extra_env={
            "CLI_PROXY_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://aa.renaissancelab.org/v1",
        },
    )
    env = agent._build_runtime_env()

    assert agent._configure_cli_proxy_model_shim_env(env)
    assert env["PIER_MASTRA_CODE_PROXY_MODEL"] == "gpt-5.5(low)"
    assert env["PIER_MASTRA_CODE_UPSTREAM_BASE_URL"] == (
        "https://aa.renaissancelab.org/v1"
    )
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:18766/v1"
    assert agent._extra_env["OPENAI_BASE_URL"] == "http://127.0.0.1:18766/v1"


def test_mastra_code_proxy_suffix_configures_custom_provider(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        MastraCode,
        "_resolve_cli_proxy_host_ip",
        staticmethod(lambda host: "203.0.113.10"),
    )
    agent = MastraCode(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5(xhigh)",
        thinking_level="xhigh",
        extra_env={
            "CLI_PROXY_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://aa.renaissancelab.org/v1",
        },
    )
    env = agent._build_runtime_env()

    assert agent._configure_cli_proxy_custom_provider_env(env)
    assert env["PIER_MASTRA_CODE_PROXY_MODEL"] == "gpt-5.5(xhigh)"
    assert env["PIER_MASTRA_CODE_CUSTOM_PROVIDER_BASE_URL"] == (
        "http://127.0.0.1:18766/v1"
    )
    assert env["PIER_MASTRA_CODE_CUSTOM_PROVIDER_ID"] == "cli-proxy"
    assert env["PIER_MASTRA_CODE_CUSTOM_PROVIDER_SETTINGS"] == (
        "/tmp/pier-mastra-code-cli-proxy-settings.json"
    )
    assert env["NO_PROXY"] == "127.0.0.1,localhost"
    assert env["PIER_MASTRA_CODE_UPSTREAM_HOST"] == "aa.renaissancelab.org"
    assert env["PIER_MASTRA_CODE_UPSTREAM_HOST_IP"] == "203.0.113.10"
    assert agent._runtime_model_and_thinking_level(
        use_cli_proxy_custom_provider=True
    ) == ("cli-proxy/gpt-5.5(xhigh)", "xhigh")

    flags = agent._build_headless_flags(use_cli_proxy_custom_provider=True)

    assert "--model 'cli-proxy/gpt-5.5(xhigh)'" in flags
    assert "--settings /tmp/pier-mastra-code-cli-proxy-settings.json" in flags


def test_mastra_code_custom_provider_settings_command_uses_runtime_env():
    command = MastraCode._build_cli_proxy_custom_provider_settings_command()

    assert "node /tmp/pier-mastra-code-cli-proxy-settings.js" in command
    assert "customProviders" in command
    assert "customModelPacks" in command
    assert "globalSettingsPath" in command
    assert "providerUrl" in command
    assert "url: providerUrl" in command
    assert "upstreamHostMapped" in command
    assert "process.env.OPENAI_API_KEY || process.env.CLI_PROXY_API_KEY" in command
    assert "apiKey: process.env" not in command
    assert "custom_provider_settings" in command


def test_mastra_code_proxy_model_shim_command_uses_node():
    command = MastraCode._build_cli_proxy_model_suffix_shim_command()

    assert "~/.nvm/nvm.sh" in command
    assert "node /tmp/pier-mastra-code-model-shim.js" in command
    assert "normalizeMastraCodePayload" in command
    assert "model_rewrite" in command
    assert "proxyUrlFromEnv" in command
    assert "method: 'CONNECT'" in command
    assert "upstreamAgent.createConnection" in command
    assert "agent: false" not in command
    assert "viaProxy" in command


def test_mastra_code_reports_default_and_configured_domains(tmp_path: Path):
    agent = MastraCode(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        extra_env={
            "OPENAI_BASE_URL": "https://endpoint.respan.ai/api/openai/v1",
            "ANTHROPIC_BASE_URL": "https://aa.renaissancelab.org",
        },
    )

    domains = set(agent.network_allowlist().domains)

    assert {"api.openai.com", "api.anthropic.com", "endpoint.respan.ai"} <= domains
    assert "aa.renaissancelab.org" in domains


def test_mastra_code_maps_cli_proxy_key_to_openai_key(tmp_path: Path):
    agent = MastraCode(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        extra_env={"CLI_PROXY_API_KEY": "test-key"},
    )

    env = agent._build_runtime_env()

    assert env["CLI_PROXY_API_KEY"] == "test-key"
    assert env["OPENAI_API_KEY"] == "test-key"


def test_mastra_code_converts_stream_json_to_atif(tmp_path: Path):
    agent = MastraCode(logs_dir=tmp_path, model_name="openai/gpt-5.5")
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I will inspect the repo."}],
            },
        },
        {
            "type": "tool_start",
            "toolCallId": "tool-1",
            "toolName": "read_file",
            "args": {"path": "README.md"},
        },
        {
            "type": "tool_end",
            "toolCallId": "tool-1",
            "result": {"content": "# Pier"},
            "isError": False,
        },
        {
            "type": "usage_update",
            "usage": {
                "promptTokens": 100,
                "completionTokens": 25,
                "totalTokens": 125,
            },
        },
        {"type": "agent_end", "reason": "completed"},
    ]

    trajectory = agent._convert_events_to_trajectory(events, instruction="Fix the tests")

    assert trajectory is not None
    assert trajectory.schema_version == "ATIF-v1.7"
    assert trajectory.agent.name == "mastra-code"
    assert trajectory.steps[0].source == "user"
    assert trajectory.steps[0].message == "Fix the tests"
    assert trajectory.steps[1].source == "agent"
    assert trajectory.steps[1].message == "I will inspect the repo."
    assert trajectory.steps[1].tool_calls is not None
    assert trajectory.steps[1].tool_calls[0].function_name == "read_file"
    assert trajectory.steps[1].observation is not None
    assert trajectory.steps[1].observation.results[0].source_call_id == "tool-1"
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 25
    assert trajectory.final_metrics.total_steps == 2
    assert trajectory.final_metrics.extra is not None
    assert trajectory.final_metrics.extra["finish_reason"] == "completed"


def test_mastra_code_detects_store_false_followup_after_successful_tool():
    assert MastraCode._is_store_false_followup_error(
        [
            {"type": "tool_end", "toolCallId": "tool-1", "isError": False},
            {
                "type": "error",
                "error": {
                    "message": (
                        "Item with id 'fc_123' not found. Items are not "
                        "persisted when `store` is set to false."
                    )
                },
            },
        ]
    )


def test_mastra_code_does_not_suppress_store_false_without_successful_tool():
    assert not MastraCode._is_store_false_followup_error(
        [
            {
                "type": "error",
                "error": {
                    "message": (
                        "Item with id 'fc_123' not found. Items are not "
                        "persisted when `store` is set to false."
                    )
                },
            },
        ]
    )


def test_mastra_code_detects_store_false_followup_from_malformed_raw_output():
    assert MastraCode._is_store_false_followup_error(
        [{"type": "tool_end", "toolCallId": "tool-1", "isError": False}],
        (
            '{"type":"error","error":{"message":"Item with id fc_123 not found. '
            "Items are not persisted when `store` is set to false."
        ),
    )


def test_mastra_code_does_not_suppress_raw_store_false_without_successful_tool():
    assert not MastraCode._is_store_false_followup_error(
        [],
        (
            '{"type":"error","error":{"message":"Item with id fc_123 not found. '
            "Items are not persisted when `store` is set to false."
        ),
    )
