import json
import re
import shlex
import socket
import sqlite3
from typing import Any, Literal
from urllib.parse import urlparse

from pier.agents.installed.base import (
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from pier.agents.network import allowlist_from_urls
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.name import AgentName
from pier.models.agent.network import NetworkAllowlist
from pier.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from pier.models.trial.paths import EnvironmentPaths
from pier.utils.trajectory_metrics import (
    extra_with_context_metrics,
    peak_context_tokens_from_steps,
    populate_context_from_final_metrics,
)
from pier.utils.trajectory_utils import format_trajectory_json


class MastraCode(BaseInstalledAgent):
    """Installed Mastra Code CLI adapter using its headless stream-json mode."""

    SUPPORTS_ATIF: bool = True

    _OUTPUT_FILENAME = "mastra-code.txt"
    _EVENTS_FILENAME = "mastra-code-events.jsonl"
    _EVENT_DB_FILENAME = "mastra-code-events.sqlite"
    _STREAM_SUMMARY_FILENAME = "mastra-code-stream-summary.json"
    _CONSOLE_LOG_FILENAME = "mastra-code-console.log"
    _STREAM_CAPTURE_SCRIPT_PATH = "/tmp/pier-mastra-code-stream-capture.js"
    _MODES = {"build", "plan", "fast"}
    _THINKING_LEVELS = {"off", "low", "medium", "high", "xhigh"}
    _CLI_PROXY_PROVIDER_ID = "cli-proxy"
    _CLI_PROXY_PROVIDER_NAME = "CLI Proxy"
    _CLI_PROXY_SETTINGS_PATH = "/tmp/pier-mastra-code-cli-proxy-settings.json"
    _HOOK_IPC_SCRIPT_PATH = "/tmp/pier-mastra-code-hook-ipc.js"
    _HOOK_IPC_SOCKET_PATH = "/tmp/pier-mastra-code-hooks.sock"
    _HOOK_IPC_READY_PATH = "/tmp/pier-mastra-code-hooks.ready"
    _HOOK_EVENTS_FILENAME = "mastra-code-hooks.jsonl"
    _HOOK_LOG_FILENAME = "mastra-code-hook-ipc.log"
    _MODEL_SUFFIX_SHIM_PORT = 18766
    _PROXY_SUFFIX_RE = re.compile(
        r"^(?P<base>.+)\((?P<level>low|medium|high|xhigh)\)$"
    )
    _NODE_VERSION_CHECK_COMMAND = (
        "node -e 'const v=process.versions.node.split(\".\").map(Number); "
        "if (v[0] < 22 || (v[0] === 22 && v[1] < 13)) { "
        "console.error(\"Node.js 22.13.0+ required, found \" + "
        "process.versions.node); process.exit(1); }'"
    )
    _DEFAULT_DOMAINS = [
        "api.openai.com",
        "api.anthropic.com",
        ".googleapis.com",
        "gateway-api.mastra.ai",
    ]
    _URL_ENV_KEYS = [
        "ANTHROPIC_BASE_URL",
        "GEMINI_API_BASE",
        "GOOGLE_GEMINI_BASE_URL",
        "MASTRA_GATEWAY_BASE_URL",
        "MASTRA_GATEWAY_URL",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "PIER_MASTRA_CODE_UPSTREAM_BASE_URL",
    ]
    _RUNTIME_ENV_KEYS = [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLI_PROXY_API_KEY",
        "GEMINI_API_BASE",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "MASTRA_API_KEY",
        "MASTRA_DB_AUTH_TOKEN",
        "MASTRA_DB_PATH",
        "MASTRA_DB_URL",
        "MASTRA_OBSERVABILITY_DB_PATH",
        "MASTRA_GATEWAY_BASE_URL",
        "MASTRA_GATEWAY_URL",
        "MASTRA_RESOURCE_ID",
        "MASTRA_STORAGE_BACKEND",
        "MASTRA_TELEMETRY_DISABLED",
        "MASTRA_USER_ID",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_API_KEY",
        "MASTRA_CODE_CLI_PROXY_CUSTOM_PROVIDER",
        "PIER_MASTRA_CODE_OM_OBSERVATION_THRESHOLD",
        "PIER_MASTRA_CODE_OM_REFLECTION_THRESHOLD",
        "PIER_MASTRA_CODE_OBSERVER_MODEL",
        "PIER_MASTRA_CODE_CUSTOM_PROVIDER_BASE_URL",
        "PIER_MASTRA_CODE_UPSTREAM_BASE_URL",
    ]

    def __init__(
        self,
        *args: Any,
        mode: Literal["build", "plan", "fast"] = "build",
        thinking_level: Literal["off", "low", "medium", "high", "xhigh"] = "high",
        timeout: int | None = None,
        settings: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if mode not in self._MODES:
            raise ValueError(
                f"Invalid Mastra Code mode '{mode}'. Valid values: {sorted(self._MODES)}"
            )
        if thinking_level not in self._THINKING_LEVELS:
            raise ValueError(
                "Invalid Mastra Code thinking_level "
                f"'{thinking_level}'. Valid values: {sorted(self._THINKING_LEVELS)}"
            )
        if timeout is not None and timeout <= 0:
            raise ValueError("Mastra Code timeout must be positive")

        self._mode = mode
        self._thinking_level = thinking_level
        self._timeout = timeout
        self._settings = settings

    @staticmethod
    def name() -> str:
        return AgentName.MASTRA_CODE.value

    @staticmethod
    def _nvm_prefix() -> str:
        return 'if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; '

    @classmethod
    def _node_version_check_command(cls) -> str:
        return cls._NODE_VERSION_CHECK_COMMAND

    def get_version_command(self) -> str | None:
        return (
            self._nvm_prefix()
            + self._node_version_check_command()
            + " && mastracode --help >/dev/null && mastracode --version"
        )

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().splitlines()[-1].strip() if stdout.strip() else ""

    def install_spec(self) -> AgentInstallSpec:
        version_spec = f"@{self._version}" if self._version else "@latest"
        root_run = (
            "if command -v apk &> /dev/null; then"
            "  apk add --no-cache curl bash nodejs npm;"
            " elif command -v apt-get &> /dev/null; then"
            "  apt-get update && apt-get install -y curl;"
            " elif command -v yum &> /dev/null; then"
            "  yum install -y curl;"
            " else"
            '  echo "Warning: No known package manager found, assuming curl is available" >&2;'
            " fi"
        )
        agent_run = (
            "set -euo pipefail; "
            "if command -v apk &> /dev/null; then"
            f"  {self._node_version_check_command()} && "
            f"npm install -g mastracode{version_spec};"
            " else"
            "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
            'export NVM_DIR="$HOME/.nvm" && '
            '\\. "$NVM_DIR/nvm.sh" || true && '
            "command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
            "nvm install 22 && "
            f"{self._node_version_check_command()} && "
            f"npm install -g mastracode{version_spec};"
            " fi && "
            f"{self._node_version_check_command()} && "
            "mastracode --help >/dev/null"
        )

        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=root_run,
                ),
                InstallStep(user="agent", run=agent_run),
            ],
            verification_command=(
                self._nvm_prefix()
                + self._node_version_check_command()
                + " && mastracode --help >/dev/null"
            ),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        values = [self._get_env(key) for key in self._URL_ENV_KEYS]
        return allowlist_from_urls(values, default_domains=self._DEFAULT_DOMAINS)

    def _build_runtime_env(self) -> dict[str, str]:
        env = self.build_process_env(
            {key: self._get_env(key) for key in self._RUNTIME_ENV_KEYS}
        )
        if "OPENAI_API_KEY" not in env and "CLI_PROXY_API_KEY" in env:
            env["OPENAI_API_KEY"] = env["CLI_PROXY_API_KEY"]
        env = {key: value for key, value in env.items() if value}
        self._configure_benchmark_runtime_env(env)
        return env

    def _run_resource_id(self) -> str:
        name = (
            self.logs_dir.parent.name
            if self.logs_dir.name == "agent"
            else self.logs_dir.name
        )
        safe = re.sub(r"[^A-Za-z0-9_.:-]+", "-", name).strip("-")
        return f"pier-{safe or 'mastra-code'}"[:128]

    def _configure_benchmark_runtime_env(self, env: dict[str, str]) -> None:
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        resource_id = self._run_resource_id()
        db_path = env.setdefault("MASTRA_DB_PATH", f"{agent_dir}/mastra-code.db")
        env.setdefault("MASTRA_STORAGE_BACKEND", "libsql")
        env.setdefault("MASTRA_DB_URL", f"file:{db_path}")
        env.setdefault(
            "MASTRA_OBSERVABILITY_DB_PATH",
            f"{agent_dir}/mastra-code-observability.duckdb",
        )
        env.setdefault("MASTRA_RESOURCE_ID", resource_id)
        env.setdefault("MASTRA_USER_ID", "pier-agent")
        env.setdefault("MASTRA_TELEMETRY_DISABLED", "1")

    def _runtime_model_and_thinking_level(
        self, use_cli_proxy_custom_provider: bool = False
    ) -> tuple[str | None, str]:
        if not self.model_name:
            return None, self._thinking_level

        match = self._PROXY_SUFFIX_RE.match(self.model_name)
        if not match:
            return self.model_name, self._thinking_level

        target_model = self._proxy_target_model_name()
        if use_cli_proxy_custom_provider and target_model:
            return f"{self._CLI_PROXY_PROVIDER_ID}/{target_model}", self._thinking_level

        # Mastra validates --model locally and rejects CLIProxyAPI's
        # parenthesized suffix syntax. Preserve Pier's result label and keep the
        # configured request effort decoupled; the local shim restores the
        # suffix in the outbound OpenAI-compatible request.
        return match.group("base"), self._thinking_level

    def _proxy_target_model_name(self) -> str | None:
        if not self.model_name:
            return None
        match = self._PROXY_SUFFIX_RE.match(self.model_name)
        if not match:
            return None

        base = match.group("base")
        if "/" in base:
            base = base.split("/", 1)[1]
        return f"{base}({match.group('level')})"

    @staticmethod
    def _is_truthy_env(value: str | None) -> bool:
        return (value or "").strip().lower() in {"1", "true", "yes", "on"}

    def _should_use_cli_proxy_model_shim(self, env: dict[str, str]) -> bool:
        forced = self._get_env("MASTRA_CODE_CLI_PROXY_MODEL_SHIM")
        if forced is not None:
            return self._is_truthy_env(forced)

        if not self._proxy_target_model_name():
            return False

        base_url = (
            env.get("OPENAI_BASE_URL")
            or env.get("OPENAI_API_BASE")
            or self._get_env("PIER_MASTRA_CODE_UPSTREAM_BASE_URL")
        )
        if not base_url:
            return False

        parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
        return parsed.hostname == "aa.renaissancelab.org"

    def _should_use_cli_proxy_custom_provider(self, env: dict[str, str]) -> bool:
        forced = self._get_env("MASTRA_CODE_CLI_PROXY_CUSTOM_PROVIDER")
        if forced is not None:
            return self._is_truthy_env(forced)

        if self._settings:
            return False
        if not self._proxy_target_model_name():
            return False

        base_url = (
            env.get("OPENAI_BASE_URL")
            or env.get("OPENAI_API_BASE")
            or self._get_env("PIER_MASTRA_CODE_UPSTREAM_BASE_URL")
        )
        if not base_url:
            return False

        parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
        return parsed.hostname == "aa.renaissancelab.org"

    def _configure_cli_proxy_custom_provider_env(self, env: dict[str, str]) -> bool:
        target_model = self._proxy_target_model_name()
        if not target_model or not self._should_use_cli_proxy_custom_provider(env):
            return False

        upstream = (
            self._get_env("PIER_MASTRA_CODE_UPSTREAM_BASE_URL")
            or env.get("OPENAI_BASE_URL")
            or env.get("OPENAI_API_BASE")
        )
        if not upstream:
            return False

        local_base_url = f"http://127.0.0.1:{self._MODEL_SUFFIX_SHIM_PORT}/v1"
        parsed = urlparse(upstream if "://" in upstream else f"https://{upstream}")
        upstream_host = parsed.hostname or ""
        updates = {
            "PIER_MASTRA_CODE_UPSTREAM_BASE_URL": upstream,
            "PIER_MASTRA_CODE_PROXY_MODEL": target_model,
            "PIER_MASTRA_CODE_CUSTOM_PROVIDER_BASE_URL": local_base_url,
            "PIER_MASTRA_CODE_CUSTOM_PROVIDER_ID": self._CLI_PROXY_PROVIDER_ID,
            "PIER_MASTRA_CODE_CUSTOM_PROVIDER_NAME": self._CLI_PROXY_PROVIDER_NAME,
            "PIER_MASTRA_CODE_CUSTOM_PROVIDER_SETTINGS": self._CLI_PROXY_SETTINGS_PATH,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
        if upstream_host:
            updates["PIER_MASTRA_CODE_UPSTREAM_HOST"] = upstream_host
            if upstream_ip := self._resolve_cli_proxy_host_ip(upstream_host):
                updates["PIER_MASTRA_CODE_UPSTREAM_HOST_IP"] = upstream_ip
        env.update(updates)
        self._extra_env.update(updates)
        return True

    @staticmethod
    def _resolve_cli_proxy_host_ip(host: str) -> str | None:
        for family in (socket.AF_INET, socket.AF_UNSPEC):
            try:
                infos = socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)
            except OSError:
                continue
            for info in infos:
                address = info[4][0]
                if address:
                    return address
        return None

    def _configure_cli_proxy_model_shim_env(self, env: dict[str, str]) -> bool:
        target_model = self._proxy_target_model_name()
        if not target_model or not self._should_use_cli_proxy_model_shim(env):
            return False

        upstream = (
            self._get_env("PIER_MASTRA_CODE_UPSTREAM_BASE_URL")
            or env.get("OPENAI_BASE_URL")
            or env.get("OPENAI_API_BASE")
        )
        if not upstream:
            return False

        local_base_url = f"http://127.0.0.1:{self._MODEL_SUFFIX_SHIM_PORT}/v1"
        updates = {
            "PIER_MASTRA_CODE_UPSTREAM_BASE_URL": upstream,
            "PIER_MASTRA_CODE_PROXY_MODEL": target_model,
            "OPENAI_BASE_URL": local_base_url,
            "OPENAI_API_BASE": local_base_url,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
        env.update(updates)
        self._extra_env.update(updates)
        return True

    @classmethod
    def _build_cli_proxy_custom_provider_settings_command(cls) -> str:
        log_path = (EnvironmentPaths.agent_dir / "mastra-code-provider.log").as_posix()
        settings_path = cls._CLI_PROXY_SETTINGS_PATH
        script_path = "/tmp/pier-mastra-code-cli-proxy-settings.js"
        script = r"""
const fs = require('fs');
const os = require('os');
const path = require('path');

const settingsPath = process.env.PIER_MASTRA_CODE_CUSTOM_PROVIDER_SETTINGS || '/tmp/pier-mastra-code-cli-proxy-settings.json';
const upstream = process.env.PIER_MASTRA_CODE_UPSTREAM_BASE_URL || process.env.OPENAI_BASE_URL || process.env.OPENAI_API_BASE || 'https://aa.renaissancelab.org/v1';
const providerUrl = process.env.PIER_MASTRA_CODE_CUSTOM_PROVIDER_BASE_URL || upstream;
const providerName = process.env.PIER_MASTRA_CODE_CUSTOM_PROVIDER_NAME || 'CLI Proxy';
const providerId = process.env.PIER_MASTRA_CODE_CUSTOM_PROVIDER_ID || 'cli-proxy';
const model = process.env.PIER_MASTRA_CODE_PROXY_MODEL || '';
const providerModel = `${providerId}/${model}`;
const observerModel = process.env.PIER_MASTRA_CODE_OBSERVER_MODEL || 'gpt-5.4-mini';
const providerObserverModel = `${providerId}/${observerModel}`;
const omObservationThreshold = Number.parseInt(process.env.PIER_MASTRA_CODE_OM_OBSERVATION_THRESHOLD || '1000000000', 10);
const omReflectionThreshold = Number.parseInt(process.env.PIER_MASTRA_CODE_OM_REFLECTION_THRESHOLD || '1000000000', 10);
const apiKey = process.env.OPENAI_API_KEY || process.env.CLI_PROXY_API_KEY || '';
const upstreamHost = process.env.PIER_MASTRA_CODE_UPSTREAM_HOST || '';
const upstreamHostIp = process.env.PIER_MASTRA_CODE_UPSTREAM_HOST_IP || '';

function getGlobalSettingsPath() {
  const platform = os.platform();
  let baseDir;
  if (platform === 'darwin') {
    baseDir = path.join(os.homedir(), 'Library', 'Application Support');
  } else if (platform === 'win32') {
    baseDir = process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming');
  } else {
    baseDir = process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share');
  }
  return path.join(baseDir, 'mastracode', 'settings.json');
}

if (!model) {
  throw new Error('PIER_MASTRA_CODE_PROXY_MODEL is required');
}
if (!apiKey) {
  throw new Error('OPENAI_API_KEY or CLI_PROXY_API_KEY is required');
}

let upstreamHostMapped = false;
if (upstreamHost && upstreamHostIp) {
  try {
    fs.appendFileSync('/etc/hosts', `\n${upstreamHostIp} ${upstreamHost}\n`, 'utf8');
    upstreamHostMapped = true;
  } catch {}
}

const settings = {
  customProviders: [
    {
      name: providerName,
      url: providerUrl,
      apiKey,
      models: [...new Set([model, observerModel])],
    },
  ],
  customModelPacks: [
    {
      name: 'pier-cli-proxy',
      models: {
        build: providerModel,
        plan: providerModel,
        fast: providerModel,
      },
      createdAt: new Date().toISOString(),
    },
  ],
  models: {
    activeModelPackId: 'custom:pier-cli-proxy',
    activeOmPackId: 'custom',
    omModelOverride: providerObserverModel,
    observerModelOverride: providerObserverModel,
    reflectorModelOverride: providerModel,
    omObservationThreshold,
    omReflectionThreshold,
    modeDefaults: {
      build: providerModel,
      plan: providerModel,
      fast: providerModel,
    },
  },
  preferences: {
    yolo: true,
    thinkingLevel: 'xhigh',
  },
};

const serialized = JSON.stringify(settings, null, 2);
fs.mkdirSync(path.dirname(settingsPath), { recursive: true });
fs.writeFileSync(settingsPath, serialized, 'utf8');
const globalSettingsPath = getGlobalSettingsPath();
fs.mkdirSync(path.dirname(globalSettingsPath), { recursive: true });
fs.writeFileSync(globalSettingsPath, serialized, 'utf8');
console.error(JSON.stringify({
  type: 'custom_provider_settings',
  provider: providerId,
  upstream,
  providerUrl,
  model,
  providerModel,
  observerModel,
  providerObserverModel,
  omObservationThreshold,
  omReflectionThreshold,
  settingsPath,
  globalSettingsPath,
  upstreamHost,
  upstreamHostMapped,
}));
"""
        escaped_script = script.strip() + "\n"
        return (
            f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()} && "
            f"cat > {script_path} <<'PIER_MASTRA_CODE_CUSTOM_PROVIDER_SETTINGS'\n"
            f"{escaped_script}"
            "PIER_MASTRA_CODE_CUSTOM_PROVIDER_SETTINGS\n"
            f"{cls._nvm_prefix()}"
            f"node {script_path} > {log_path} 2>&1 && "
            f"test -s {settings_path}"
        )

    @classmethod
    def _build_cli_proxy_model_suffix_shim_command(cls) -> str:
        log_path = (EnvironmentPaths.agent_dir / "mastra-code-shim.log").as_posix()
        ready_path = "/tmp/pier-mastra-code-model-shim.ready"
        script_path = "/tmp/pier-mastra-code-model-shim.js"
        script = rf"""
const http = require('http');
const https = require('https');
const tls = require('tls');
const fs = require('fs');
const {{ URL }} = require('url');

const upstreamBase = new URL(process.env.PIER_MASTRA_CODE_UPSTREAM_BASE_URL || 'https://aa.renaissancelab.org/v1');
const proxyModel = process.env.PIER_MASTRA_CODE_PROXY_MODEL || '';
const apiKey = process.env.OPENAI_API_KEY || process.env.CLI_PROXY_API_KEY || '';
const port = Number(process.env.PIER_MASTRA_CODE_SHIM_PORT || '{cls._MODEL_SUFFIX_SHIM_PORT}');
const readyFile = process.env.PIER_MASTRA_CODE_SHIM_READY_FILE || '/tmp/pier-mastra-code-model-shim.ready';

function normalizeMastraCodePayload(body) {{
  if (proxyModel && body && typeof body === 'object' && typeof body.model === 'string') {{
    const originalModel = body.model;
    body.model = proxyModel;
    console.error(JSON.stringify({{ type: 'model_rewrite', from: originalModel, to: proxyModel }}));
  }}
  return body;
}}

function upstreamUrlFor(reqUrl) {{
  const incoming = new URL(reqUrl, 'http://127.0.0.1');
  const basePath = upstreamBase.pathname.replace(/\/+$/, '');
  const target = new URL(upstreamBase.toString());
  if (basePath && incoming.pathname.startsWith(`${{basePath}}/`)) {{
    target.pathname = incoming.pathname;
  }} else {{
    target.pathname = `${{basePath}}${{incoming.pathname}}`.replace(/\/{{2,}}/g, '/');
  }}
  target.search = incoming.search;
  return target;
}}

function proxyUrlFromEnv() {{
  const raw = process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy || '';
  if (!raw) return null;
  try {{
    return new URL(raw);
  }} catch (error) {{
    console.error(JSON.stringify({{ type: 'invalid_proxy_url', message: error.message }}));
    return null;
  }}
}}

function proxyAuthHeader(proxyUrl) {{
  const username = decodeURIComponent(proxyUrl.username || '');
  const password = decodeURIComponent(proxyUrl.password || '');
  if (!username && !password) return null;
  return `Basic ${{Buffer.from(`${{username}}:${{password}}`).toString('base64')}}`;
}}

function finishWithError(res, error) {{
  if (!res.headersSent) {{
    res.writeHead(502, {{ 'Content-Type': 'application/json' }});
    res.end(JSON.stringify({{ error: error.message }}));
    return;
  }}
  res.destroy(error);
}}

function sendDirect(target, headers, rawBody, res) {{
  const transport = target.protocol === 'http:' ? http : https;
  const upstreamReq = transport.request(target, {{ method: 'POST', headers }}, (upstreamRes) => {{
    res.writeHead(upstreamRes.statusCode || 502, upstreamRes.headers);
    upstreamRes.pipe(res);
  }});
  upstreamReq.on('error', (error) => finishWithError(res, error));
  upstreamReq.write(rawBody);
  upstreamReq.end();
}}

function sendViaHttpProxy(target, headers, rawBody, res, proxyUrl) {{
  const proxyTransport = proxyUrl.protocol === 'https:' ? https : http;
  const proxyPort = proxyUrl.port || (proxyUrl.protocol === 'https:' ? '443' : '80');
  const connectHeaders = {{}};
  const auth = proxyAuthHeader(proxyUrl);
  if (auth) connectHeaders['Proxy-Authorization'] = auth;

  const connectReq = proxyTransport.request({{
    host: proxyUrl.hostname,
    port: proxyPort,
    method: 'CONNECT',
    path: `${{target.hostname}}:${{target.port || '443'}}`,
    headers: connectHeaders,
  }});

  connectReq.on('connect', (connectRes, socket, head) => {{
    if (connectRes.statusCode !== 200) {{
      if (!res.headersSent) {{
        res.writeHead(connectRes.statusCode || 502, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ error: `proxy CONNECT failed: ${{connectRes.statusCode || 0}}` }}));
      }}
      socket.destroy();
      return;
    }}

    if (head && head.length) socket.unshift(head);
    const tlsSocket = tls.connect({{ socket, servername: target.hostname }}, () => {{
      const upstreamAgent = new https.Agent({{ keepAlive: false }});
      upstreamAgent.createConnection = () => tlsSocket;
      const upstreamReq = https.request({{
        protocol: 'https:',
        host: target.hostname,
        port: target.port || 443,
        path: `${{target.pathname}}${{target.search}}`,
        method: 'POST',
        headers,
        agent: upstreamAgent,
      }}, (upstreamRes) => {{
        res.writeHead(upstreamRes.statusCode || 502, upstreamRes.headers);
        upstreamRes.pipe(res);
      }});
      upstreamReq.on('error', (error) => finishWithError(res, error));
      upstreamReq.write(rawBody);
      upstreamReq.end();
    }});
    tlsSocket.on('error', (error) => finishWithError(res, error));
  }});
  connectReq.on('error', (error) => finishWithError(res, error));
  connectReq.end();
}}

function sendUpstream(target, headers, rawBody, res) {{
  const proxyUrl = target.protocol === 'https:' ? proxyUrlFromEnv() : null;
  console.error(JSON.stringify({{
    type: 'upstream_request',
    viaProxy: Boolean(proxyUrl),
    targetHost: target.hostname,
    path: target.pathname,
    model: proxyModel || undefined,
  }}));
  if (proxyUrl) {{
    sendViaHttpProxy(target, headers, rawBody, res, proxyUrl);
    return;
  }}
  sendDirect(target, headers, rawBody, res);
}}

const server = http.createServer((req, res) => {{
  if (req.method !== 'POST') {{
    res.writeHead(404, {{ 'Content-Type': 'application/json' }});
    res.end(JSON.stringify({{ error: 'not found' }}));
    return;
  }}

  const chunks = [];
  req.on('data', (chunk) => chunks.push(chunk));
  req.on('end', () => {{
    let rawBody = Buffer.concat(chunks).toString('utf8');
    try {{
      const parsed = JSON.parse(rawBody || '{{}}');
      rawBody = JSON.stringify(normalizeMastraCodePayload(parsed));
    }} catch (error) {{
      res.writeHead(400, {{ 'Content-Type': 'application/json' }});
      res.end(JSON.stringify({{ error: `invalid json: ${{error.message}}` }}));
      return;
    }}

    const target = upstreamUrlFor(req.url);
    const headers = {{ ...req.headers }};
    delete headers.host;
    delete headers['content-length'];
    headers['content-type'] = 'application/json';
    headers['content-length'] = Buffer.byteLength(rawBody);
    headers['authorization'] = `Bearer ${{apiKey}}`;
    headers['x-api-key'] = apiKey;

    sendUpstream(target, headers, rawBody, res);
  }});
}});

server.listen(port, '127.0.0.1', () => {{
  fs.writeFileSync(readyFile, String(process.pid));
}});
"""
        escaped_script = script.strip() + "\n"
        return (
            f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()} && "
            f"cat > {script_path} <<'PIER_MASTRA_CODE_MODEL_SHIM'\n"
            f"{escaped_script}"
            "PIER_MASTRA_CODE_MODEL_SHIM\n"
            f"rm -f {ready_path} && "
            f"{cls._nvm_prefix()}"
            f"PIER_MASTRA_CODE_SHIM_READY_FILE={ready_path} "
            f"node {script_path} > {log_path} 2>&1 & "
            f"for i in $(seq 1 50); do "
            f"  [ -f {ready_path} ] && exit 0; "
            "  sleep 0.1; "
            "done; "
            f"cat {log_path} >&2 || true; "
            "exit 1"
        )

    def _build_headless_flags(
        self, use_cli_proxy_custom_provider: bool = False
    ) -> str:
        runtime_model, thinking_level = self._runtime_model_and_thinking_level(
            use_cli_proxy_custom_provider=use_cli_proxy_custom_provider
        )
        parts = ["--output-format", "stream-json", "--thinking-level", thinking_level]
        if runtime_model:
            parts.extend(["--model", runtime_model])
        else:
            parts.extend(["--mode", self._mode])
        if self._timeout is not None:
            parts.extend(["--timeout", str(self._timeout)])
        parts.extend(["--resource-id", self._run_resource_id()])
        parts.extend(["--title", self._run_resource_id()])
        settings = (
            self._CLI_PROXY_SETTINGS_PATH
            if use_cli_proxy_custom_provider
            else self._settings
        )
        if settings:
            parts.extend(["--settings", settings])
        return " ".join(shlex.quote(part) for part in parts)

    @classmethod
    def _build_hook_ipc_observer_command(cls) -> str:
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        events_path = f"{agent_dir}/{cls._HOOK_EVENTS_FILENAME}"
        log_path = f"{agent_dir}/{cls._HOOK_LOG_FILENAME}"
        script_path = cls._HOOK_IPC_SCRIPT_PATH
        socket_path = cls._HOOK_IPC_SOCKET_PATH
        ready_path = cls._HOOK_IPC_READY_PATH
        script = r"""
const fs = require('fs');
const net = require('net');

const mode = process.argv[2] || '';
const socketPath = process.env.PIER_MASTRA_CODE_HOOK_SOCKET || '/tmp/pier-mastra-code-hooks.sock';
const eventsPath = process.env.PIER_MASTRA_CODE_HOOK_EVENTS || '/logs/agent/mastra-code-hooks.jsonl';
const readyPath = process.env.PIER_MASTRA_CODE_HOOK_READY || '/tmp/pier-mastra-code-hooks.ready';

function append(record) {
  fs.mkdirSync(require('path').dirname(eventsPath), { recursive: true });
  fs.appendFileSync(eventsPath, JSON.stringify(record) + '\n', 'utf8');
}

function readStdin() {
  return new Promise((resolve) => {
    const chunks = [];
    process.stdin.on('data', (chunk) => chunks.push(chunk));
    process.stdin.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
  });
}

async function emit() {
  const raw = await readStdin();
  let payload;
  try {
    payload = raw.trim() ? JSON.parse(raw) : {};
  } catch (error) {
    payload = { parseError: error.message, raw };
  }
  const record = {
    type: 'mastra_hook',
    observedAt: new Date().toISOString(),
    pid: process.pid,
    payload,
  };
  const client = net.createConnection(socketPath);
  client.on('error', () => {
    append({ ...record, fallback: true });
    process.exit(0);
  });
  client.on('connect', () => {
    client.end(JSON.stringify(record) + '\n');
  });
  client.on('close', () => process.exit(0));
}

function listen() {
  try { fs.unlinkSync(socketPath); } catch {}
  try { fs.unlinkSync(readyPath); } catch {}
  fs.mkdirSync(require('path').dirname(eventsPath), { recursive: true });
  fs.writeFileSync(eventsPath, '', { flag: 'a' });
  const server = net.createServer((socket) => {
    let buffer = '';
    socket.on('data', (chunk) => buffer += chunk.toString('utf8'));
    socket.on('end', () => {
      for (const line of buffer.split('\n')) {
        if (!line.trim()) continue;
        try {
          append(JSON.parse(line));
        } catch (error) {
          append({ type: 'mastra_hook_parse_error', observedAt: new Date().toISOString(), error: error.message, raw: line });
        }
      }
    });
  });
  server.listen(socketPath, () => {
    fs.writeFileSync(readyPath, String(process.pid));
  });
}

if (mode === 'emit') emit();
else if (mode === 'listen') listen();
else {
  process.stderr.write('usage: node pier-mastra-code-hook-ipc.js <listen|emit>\n');
  process.exit(2);
}
"""
        hook_command = f"{cls._nvm_prefix()}node {script_path} emit"
        hooks = {
            event: [
                {
                    "type": "command",
                    "command": hook_command,
                    "timeout": 5000,
                    "description": "Pier Mastra Code hook event capture",
                }
            ]
            for event in [
                "PreToolUse",
                "PostToolUse",
                "Stop",
                "UserPromptSubmit",
                "SessionStart",
                "SessionEnd",
                "Notification",
            ]
        }
        hooks_json = json.dumps(hooks, indent=2)
        escaped_script = script.strip() + "\n"
        return (
            f"HOOK_CONFIG_DIR=\"$({cls._nvm_prefix()}"
            "node -e 'process.stdout.write(require(\"os\").homedir()+\"/.mastracode\")')\" && "
            "PROJECT_ROOT=\"$(git rev-parse --show-toplevel 2>/dev/null || pwd)\" && "
            "PROJECT_HOOK_CONFIG_DIR=\"$PROJECT_ROOT/.mastracode\" && "
            f"mkdir -p {agent_dir} \"$HOOK_CONFIG_DIR\" \"$PROJECT_HOOK_CONFIG_DIR\" && "
            f"cat > {script_path} <<'PIER_MASTRA_CODE_HOOK_IPC'\n"
            f"{escaped_script}"
            "PIER_MASTRA_CODE_HOOK_IPC\n"
            f"cat > \"$HOOK_CONFIG_DIR/hooks.json\" <<'PIER_MASTRA_CODE_HOOKS'\n"
            f"{hooks_json}\n"
            "PIER_MASTRA_CODE_HOOKS\n"
            "cp \"$HOOK_CONFIG_DIR/hooks.json\" \"$PROJECT_HOOK_CONFIG_DIR/hooks.json\" && "
            "printf '{\"global\":\"%s\",\"project\":\"%s\"}\\n' "
            "\"$HOOK_CONFIG_DIR/hooks.json\" "
            f"\"$PROJECT_HOOK_CONFIG_DIR/hooks.json\" > {agent_dir}/mastra-code-hook-config.json && "
            f"rm -f {socket_path} {ready_path} && "
            f"{cls._nvm_prefix()}"
            f"PIER_MASTRA_CODE_HOOK_SOCKET={socket_path} "
            f"PIER_MASTRA_CODE_HOOK_EVENTS={events_path} "
            f"PIER_MASTRA_CODE_HOOK_READY={ready_path} "
            f"node {script_path} listen > {log_path} 2>&1 & "
            f"for i in $(seq 1 50); do "
            f"  [ -S {socket_path} ] && [ -f {ready_path} ] && exit 0; "
            "  sleep 0.1; "
            "done; "
            f"cat {log_path} >&2 || true; "
            "exit 1"
        )

    @classmethod
    def _build_stream_capture_command(cls) -> str:
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        script_path = cls._STREAM_CAPTURE_SCRIPT_PATH
        events_path = f"{agent_dir}/{cls._EVENTS_FILENAME}"
        console_path = f"{agent_dir}/{cls._CONSOLE_LOG_FILENAME}"
        summary_path = f"{agent_dir}/{cls._STREAM_SUMMARY_FILENAME}"
        script = r"""
const fs = require('fs');
const path = require('path');
const { StringDecoder } = require('string_decoder');

const eventsPath = process.env.PIER_MASTRA_CODE_EVENTS || '/logs/agent/mastra-code-events.jsonl';
const consolePath = process.env.PIER_MASTRA_CODE_CONSOLE || '/logs/agent/mastra-code-console.log';
const summaryPath = process.env.PIER_MASTRA_CODE_STREAM_SUMMARY || '/logs/agent/mastra-code-stream-summary.json';
const consoleByteLimit = Number.parseInt(process.env.PIER_MASTRA_CODE_CONSOLE_BYTES || '2097152', 10);
const maxLineChars = Number.parseInt(process.env.PIER_MASTRA_CODE_MAX_LINE_CHARS || '2097152', 10);
const maxStringChars = Number.parseInt(process.env.PIER_MASTRA_CODE_MAX_STRING_CHARS || '120000', 10);
const maxObjectKeys = Number.parseInt(process.env.PIER_MASTRA_CODE_MAX_OBJECT_KEYS || '80', 10);
const maxArrayItems = Number.parseInt(process.env.PIER_MASTRA_CODE_MAX_ARRAY_ITEMS || '80', 10);
const maxDepth = Number.parseInt(process.env.PIER_MASTRA_CODE_MAX_DEPTH || '8', 10);

const keepEventTypes = new Set([
  'agent_start',
  'agent_end',
  'error',
  'message_end',
  'om_buffering_failed',
  'om_observation_end',
  'om_observation_start',
  'om_status',
  'thread_created',
  'tool_end',
  'tool_start',
  'usage_update',
  'warning',
]);

const skippedEventTypes = new Set([
  'display_state_changed',
  'message_delta',
  'message_start',
  'message_update',
  'thinking_delta',
  'thinking_update',
  'tool_input_delta',
  'tool_input_end',
  'tool_input_start',
  'tool_input_update',
]);

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

ensureDir(eventsPath);
ensureDir(consolePath);
ensureDir(summaryPath);
fs.writeFileSync(eventsPath, '', 'utf8');
fs.writeFileSync(consolePath, '', 'utf8');

const eventsStream = fs.createWriteStream(eventsPath, { flags: 'a' });
const consoleStream = fs.createWriteStream(consolePath, { flags: 'a' });
const decoder = new StringDecoder('utf8');
const startedAt = new Date().toISOString();
const eventCounts = {};
const keptEventCounts = {};
const skippedEventCounts = {};
let buffer = '';
let droppingOversizedLine = false;
let consoleBytes = 0;
let totalLines = 0;
let jsonLines = 0;
let keptEvents = 0;
let skippedEvents = 0;
let nonJsonLines = 0;
let invalidJsonLines = 0;
let oversizedLines = 0;
let finished = false;

function count(counter, key) {
  const normalized = key || 'unknown';
  counter[normalized] = (counter[normalized] || 0) + 1;
}

function appendConsole(text) {
  if (consoleBytes >= consoleByteLimit) return;
  const line = Buffer.from(`${text}\n`, 'utf8');
  const remaining = consoleByteLimit - consoleBytes;
  if (line.length <= remaining) {
    consoleStream.write(line);
    consoleBytes += line.length;
    return;
  }
  consoleStream.write(line.subarray(0, remaining));
  consoleBytes += remaining;
}

function truncateString(value) {
  if (value.length <= maxStringChars) return value;
  return `${value.slice(0, maxStringChars)}...[truncated ${value.length - maxStringChars} chars]`;
}

function compactValue(value, depth = 0) {
  if (value === null || value === undefined) return value;
  if (typeof value === 'string') return truncateString(value);
  if (typeof value !== 'object') return value;
  if (depth >= maxDepth) {
    return Array.isArray(value) ? `[array:${value.length}]` : '[object]';
  }
  if (Array.isArray(value)) {
    const compact = value.slice(0, maxArrayItems).map((item) => compactValue(item, depth + 1));
    if (value.length > maxArrayItems) {
      compact.push(`[truncated ${value.length - maxArrayItems} items]`);
    }
    return compact;
  }

  const compact = {};
  const entries = Object.entries(value);
  for (const [key, item] of entries.slice(0, maxObjectKeys)) {
    compact[key] = compactValue(item, depth + 1);
  }
  if (entries.length > maxObjectKeys) {
    compact.__truncatedKeys = entries.length - maxObjectKeys;
  }
  return compact;
}

function shouldKeepEvent(event, eventType) {
  if (keepEventTypes.has(eventType)) return true;
  if (skippedEventTypes.has(eventType)) return false;
  if (event.error || event.isError || String(eventType).includes('error')) return true;
  return false;
}

function mirrorEvent(event, compact) {
  const eventType = String(event.type || 'unknown');
  if (eventType === 'error' || eventType === 'agent_end' || event.isError) {
    process.stdout.write(`${JSON.stringify(compact)}\n`);
  }
}

function processLine(line) {
  totalLines += 1;
  const trimmed = line.trim();
  if (!trimmed) return;
  if (!trimmed.startsWith('{')) {
    nonJsonLines += 1;
    appendConsole(trimmed);
    return;
  }

  let event;
  try {
    event = JSON.parse(trimmed);
  } catch (error) {
    invalidJsonLines += 1;
    appendConsole(JSON.stringify({ type: 'invalid_json_line', error: error.message, line: truncateString(trimmed) }));
    return;
  }
  if (!event || typeof event !== 'object' || Array.isArray(event)) {
    nonJsonLines += 1;
    return;
  }

  jsonLines += 1;
  const eventType = String(event.type || 'unknown');
  count(eventCounts, eventType);
  if (!shouldKeepEvent(event, eventType)) {
    skippedEvents += 1;
    count(skippedEventCounts, eventType);
    return;
  }

  const compact = compactValue(event);
  keptEvents += 1;
  count(keptEventCounts, eventType);
  eventsStream.write(`${JSON.stringify(compact)}\n`);
  mirrorEvent(event, compact);
}

function handleText(text) {
  let start = 0;
  while (start < text.length) {
    const newline = text.indexOf('\n', start);
    if (newline === -1) break;
    const segment = text.slice(start, newline).replace(/\r$/, '');
    if (droppingOversizedLine) {
      droppingOversizedLine = false;
    } else {
      buffer += segment;
      processLine(buffer);
    }
    buffer = '';
    start = newline + 1;
  }

  if (start >= text.length) return;
  if (droppingOversizedLine) return;
  buffer += text.slice(start);
  if (buffer.length > maxLineChars) {
    oversizedLines += 1;
    appendConsole(JSON.stringify({
      type: 'oversized_stream_line_dropped',
      maxLineChars,
      observedChars: buffer.length,
    }));
    buffer = '';
    droppingOversizedLine = true;
  }
}

function writeSummary() {
  const summary = {
    type: 'mastra_code_stream_summary',
    startedAt,
    finishedAt: new Date().toISOString(),
    eventsPath,
    consolePath,
    totalLines,
    jsonLines,
    keptEvents,
    skippedEvents,
    nonJsonLines,
    invalidJsonLines,
    oversizedLines,
    consoleBytes,
    eventCounts,
    keptEventCounts,
    skippedEventCounts,
  };
  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  process.stdout.write(`${JSON.stringify(summary)}\n`);
}

function finish() {
  if (finished) return;
  finished = true;
  if (!droppingOversizedLine && buffer) {
    processLine(buffer);
  }
  buffer = '';
  writeSummary();
  eventsStream.end();
  consoleStream.end();
}

process.stdin.on('data', (chunk) => handleText(decoder.write(chunk)));
process.stdin.on('end', () => {
  const tail = decoder.end();
  if (tail) handleText(tail);
  finish();
});
process.stdin.on('error', (error) => {
  appendConsole(JSON.stringify({ type: 'stdin_error', error: error.message }));
  process.exitCode = 1;
  finish();
});
process.on('uncaughtException', (error) => {
  appendConsole(JSON.stringify({ type: 'stream_capture_uncaught_exception', error: error.message, stack: error.stack }));
  process.exitCode = 1;
  finish();
});
"""
        escaped_script = script.strip() + "\n"
        return (
            f"mkdir -p {agent_dir} && "
            f"cat > {script_path} <<'PIER_MASTRA_CODE_STREAM_CAPTURE'\n"
            f"{escaped_script}"
            "PIER_MASTRA_CODE_STREAM_CAPTURE\n"
            f"{cls._nvm_prefix()}"
            f"PIER_MASTRA_CODE_EVENTS={events_path} "
            f"PIER_MASTRA_CODE_CONSOLE={console_path} "
            f"PIER_MASTRA_CODE_STREAM_SUMMARY={summary_path} "
            f"node -c {script_path} && "
            f": > {events_path} && : > {console_path}"
        )

    @classmethod
    def _stream_capture_invocation(cls) -> str:
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        return (
            f"PIER_MASTRA_CODE_EVENTS={agent_dir}/{cls._EVENTS_FILENAME} "
            f"PIER_MASTRA_CODE_CONSOLE={agent_dir}/{cls._CONSOLE_LOG_FILENAME} "
            f"PIER_MASTRA_CODE_STREAM_SUMMARY={agent_dir}/{cls._STREAM_SUMMARY_FILENAME} "
            f"node {cls._STREAM_CAPTURE_SCRIPT_PATH}"
        )

    @staticmethod
    def _has_successful_tool_end(events: list[dict[str, Any]]) -> bool:
        for event in events:
            if event.get("type") == "tool_end" and event.get("isError") is False:
                return True
        return False

    @classmethod
    def _text_has_store_false_error(cls, text: str) -> bool:
        message = text.lower()
        return (
            "items are not persisted" in message
            and "store" in message
            and "false" in message
        )

    @classmethod
    def _is_store_false_followup_error(
        cls, events: list[dict[str, Any]], raw_output: str = ""
    ) -> bool:
        if not cls._has_successful_tool_end(events):
            return False

        for event in events:
            if event.get("type") != "error":
                continue

            if cls._text_has_store_false_error(cls._stringify(event.get("error"))):
                return True

        return bool(raw_output) and cls._text_has_store_false_error(raw_output)

    def _stream_event_path(self):
        for filename in [self._EVENTS_FILENAME, self._OUTPUT_FILENAME]:
            path = self.logs_dir / filename
            try:
                if path.stat().st_size > 0:
                    return path
            except OSError:
                continue
        return None

    def _iter_stdout_lines(self):
        output_path = self._stream_event_path()
        if output_path is None:
            return

        try:
            with output_path.open(encoding="utf-8", errors="replace") as stream:
                yield from stream
        except OSError as exc:
            self.logger.debug(f"Failed to read Mastra Code stream {output_path}: {exc}")

    @staticmethod
    def _read_text_tail(path, max_bytes: int = 1_048_576) -> str:
        try:
            size = path.stat().st_size
            with path.open("rb") as stream:
                if size > max_bytes:
                    stream.seek(size - max_bytes)
                return stream.read(max_bytes).decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _read_stdout(self) -> str:
        chunks: list[str] = []
        for filename in [
            self._CONSOLE_LOG_FILENAME,
            self._EVENTS_FILENAME,
            self._OUTPUT_FILENAME,
        ]:
            path = self.logs_dir / filename
            if path.exists():
                tail = self._read_text_tail(path)
                if tail:
                    chunks.append(tail)
        return "\n".join(chunks)

    def _parse_stdout(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in self._iter_stdout_lines() or []:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def _write_events_sqlite(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return

        db_path = self.logs_dir / self._EVENT_DB_FILENAME
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mastra_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute("DELETE FROM mastra_events")
                conn.executemany(
                    """
                    INSERT INTO mastra_events (event_type, payload_json)
                    VALUES (?, ?)
                    """,
                    [
                        (
                            str(event.get("type") or "unknown"),
                            json.dumps(event, ensure_ascii=False),
                        )
                        for event in events
                    ],
                )
                conn.execute(
                    """
                    CREATE VIEW IF NOT EXISTS mastra_event_counts AS
                    SELECT event_type, COUNT(*) AS count
                    FROM mastra_events
                    GROUP BY event_type
                    """
                )
        except (OSError, sqlite3.Error) as exc:
            self.logger.debug(f"Failed to write Mastra Code SQLite events: {exc}")

    @classmethod
    def _extract_message_text(cls, message: Any) -> str:
        if isinstance(message, str):
            return message
        if not isinstance(message, dict):
            return cls._stringify(message)

        content = message.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return cls._stringify(content) if content is not None else ""

        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("text") is not None:
                    parts.append(cls._stringify(item.get("text")))
            elif item is not None:
                parts.append(cls._stringify(item))
        return "\n\n".join(part.strip() for part in parts if part and part.strip())

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    @staticmethod
    def _arguments(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value is None:
            return {}
        return {"value": value}

    def _convert_events_to_trajectory(
        self, events: list[dict[str, Any]], instruction: str
    ) -> Trajectory | None:
        if not events:
            return None

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        observations: list[ObservationResult] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        session_id: str | None = None

        for event in events:
            if not session_id:
                value = event.get("sessionId") or event.get("session_id")
                if isinstance(value, str) and value:
                    session_id = value

            event_type = event.get("type")

            if event_type == "message_end":
                text = self._extract_message_text(event.get("message"))
                if text:
                    text_parts.append(text)
                continue

            if event_type == "tool_start":
                call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "")
                if not call_id:
                    call_id = f"tool-{len(tool_calls) + 1}"
                tool_calls.append(
                    ToolCall(
                        tool_call_id=call_id,
                        function_name=str(
                            event.get("toolName") or event.get("tool_name") or ""
                        ),
                        arguments=self._arguments(event.get("args")),
                    )
                )
                continue

            if event_type == "tool_end":
                call_id = event.get("toolCallId") or event.get("tool_call_id")
                result_extra: dict[str, Any] = {}
                if event.get("isError") is not None:
                    result_extra["is_error"] = event.get("isError")
                observations.append(
                    ObservationResult(
                        source_call_id=str(call_id) if call_id is not None else None,
                        content=self._stringify(event.get("result")),
                        extra=result_extra or None,
                    )
                )
                continue

            if event_type == "usage_update" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
                continue

            if event_type == "agent_end":
                reason = event.get("reason")
                finish_reason = str(reason) if reason is not None else None

        steps: list[Step] = [
            Step(step_id=1, source="user", message=instruction),
            Step(
                step_id=2,
                source="agent",
                message="\n\n".join(text_parts),
                model_name=self.model_name,
                tool_calls=tool_calls or None,
                observation=Observation(results=observations)
                if observations
                else None,
                metrics=self._usage_to_metrics(usage),
                llm_call_count=1,
            ),
        ]

        final_extra: dict[str, Any] = {}
        if finish_reason:
            final_extra["finish_reason"] = finish_reason
        if usage.get("totalTokens") is not None:
            final_extra["total_tokens"] = usage["totalTokens"]

        final_metrics = FinalMetrics(
            total_prompt_tokens=usage.get("promptTokens"),
            total_completion_tokens=usage.get("completionTokens"),
            total_steps=len(steps),
            extra=extra_with_context_metrics(
                final_extra or None,
                peak_context_tokens=peak_context_tokens_from_steps(steps),
                summarization_count=None,
            ),
        )

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id or "unknown",
            agent=Agent(
                name=AgentName.MASTRA_CODE.value,
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=final_metrics,
        )

    @staticmethod
    def _usage_to_metrics(usage: dict[str, Any]) -> Metrics | None:
        if not usage:
            return None
        prompt_tokens = usage.get("promptTokens")
        completion_tokens = usage.get("completionTokens")
        if prompt_tokens is None and completion_tokens is None:
            return None
        return Metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            extra={
                key: value
                for key, value in usage.items()
                if key not in {"promptTokens", "completionTokens"}
            }
            or None,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        events = self._parse_stdout()
        if not events:
            return
        self._write_events_sqlite(events)

        try:
            trajectory = self._convert_events_to_trajectory(events, instruction="")
        except Exception:
            self.logger.exception("Failed to convert Mastra Code events to trajectory")
            return

        if not trajectory:
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            trajectory_path.write_text(
                format_trajectory_json(trajectory.to_json_dict()),
                encoding="utf-8",
            )
            self.logger.debug(f"Wrote Mastra Code trajectory to {trajectory_path}")
        except OSError as exc:
            self.logger.debug(
                f"Failed to write trajectory file {trajectory_path}: {exc}"
            )

        if trajectory.final_metrics:
            populate_context_from_final_metrics(context, trajectory.final_metrics)

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)
        env = self._build_runtime_env()
        use_cli_proxy_custom_provider = self._configure_cli_proxy_custom_provider_env(env)
        use_cli_proxy_model_shim = (
            False
            if use_cli_proxy_custom_provider
            else self._configure_cli_proxy_model_shim_env(env)
        )
        flags = self._build_headless_flags(
            use_cli_proxy_custom_provider=use_cli_proxy_custom_provider
        )

        command = (
            f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()} && "
            f"{self._nvm_prefix()}"
            f"printf '%s' {escaped_instruction} | "
            f"mastracode --prompt - {flags} "
            f"2>&1 | {self._stream_capture_invocation()}"
        )

        try:
            if use_cli_proxy_custom_provider:
                await self.exec_as_agent(
                    environment,
                    command=self._build_cli_proxy_custom_provider_settings_command(),
                    env=env,
                    timeout_sec=self._timeout,
                )
                await self.exec_as_agent(
                    environment,
                    command=self._build_cli_proxy_model_suffix_shim_command(),
                    env=env,
                    timeout_sec=self._timeout,
                )
            elif use_cli_proxy_model_shim:
                await self.exec_as_agent(
                    environment,
                    command=self._build_cli_proxy_model_suffix_shim_command(),
                    env=env,
                    timeout_sec=self._timeout,
                )
            await self.exec_as_agent(
                environment,
                command=self._build_hook_ipc_observer_command(),
                env=env,
                timeout_sec=self._timeout,
            )
            await self.exec_as_agent(
                environment,
                command=self._build_stream_capture_command(),
                env=env,
                timeout_sec=self._timeout,
            )
            await self.exec_as_agent(
                environment,
                command=command,
                env=env,
                timeout_sec=self._timeout,
            )
            self._write_events_sqlite(self._parse_stdout())
        except NonZeroAgentExitCodeError:
            events = self._parse_stdout()
            self._write_events_sqlite(events)
            if not self._is_store_false_followup_error(events, self._read_stdout()):
                raise
            self.logger.warning(
                "Mastra Code hit a post-tool OpenAI Responses store=false "
                "follow-up error; continuing so Pier can verify task state."
            )
