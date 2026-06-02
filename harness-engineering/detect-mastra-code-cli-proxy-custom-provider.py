#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def tool_summary(records: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str]]:
    starts: Counter[str] = Counter()
    ends: Counter[str] = Counter()
    for record in records:
        if record.get("type") == "tool_start":
            starts[str(record.get("toolName") or "?")] += 1
        if record.get("type") == "tool_end":
            ends[str(record.get("toolName") or "?")] += 1
        message = record.get("message") or {}
        if not isinstance(message, dict):
            continue
        for item in message.get("content") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_call":
                starts[str(item.get("name") or "?")] += 1
            if item.get("type") == "tool_result":
                ends[str(item.get("name") or "?")] += 1
    return starts, ends


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Mastra Code CLI Proxy custom-provider observability for a "
            "Pier trial directory."
        )
    )
    parser.add_argument("trial_dir", type=Path, help="Pier trial directory")
    parser.add_argument(
        "--model",
        default=None,
        help="Expected proxy-visible model, e.g. gpt-5.5(xhigh)",
    )
    parser.add_argument(
        "--require-tool-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require at least one Mastra tool_start/tool_call event",
    )
    args = parser.parse_args()

    agent_dir = args.trial_dir / "agent"
    provider_records = load_jsonl(agent_dir / "mastra-code-provider.log")
    shim_records = load_jsonl(agent_dir / "mastra-code-shim.log")
    mastra_records = load_jsonl(agent_dir / "mastra-code.txt")

    failures: list[str] = []
    provider = next(
        (r for r in provider_records if r.get("type") == "custom_provider_settings"),
        None,
    )
    if not provider:
        failures.append("missing custom_provider_settings record")
    else:
        provider_url = str(provider.get("providerUrl") or "")
        model = str(provider.get("model") or "")
        provider_model = str(provider.get("providerModel") or "")
        if not provider_url.startswith("http://127.0.0.1:"):
            failures.append(f"providerUrl is not local: {provider_url!r}")
        if args.model and model != args.model:
            failures.append(f"model={model!r}, expected {args.model!r}")
        if model and provider_model != f"cli-proxy/{model}":
            failures.append(
                f"providerModel={provider_model!r}, expected cli-proxy/{model}"
            )

    upstream_requests = [
        r for r in shim_records if r.get("type") == "upstream_request"
    ]
    if not upstream_requests:
        failures.append("missing shim upstream_request record")
    for record in upstream_requests:
        if record.get("path") != "/v1/chat/completions":
            failures.append(f"unexpected upstream path: {record.get('path')!r}")
        if record.get("viaProxy") is not True:
            failures.append("upstream request did not use proxy")
        if args.model and record.get("model") != args.model:
            failures.append(
                f"shim model={record.get('model')!r}, expected {args.model!r}"
            )

    starts, ends = tool_summary(mastra_records)
    if args.require_tool_start and not starts:
        failures.append("missing Mastra tool_start/tool_call evidence")

    errors = [r.get("error") for r in mastra_records if r.get("type") == "error"]
    print(f"trial={args.trial_dir}")
    print(f"provider_records={len(provider_records)}")
    print(f"upstream_requests={len(upstream_requests)}")
    print(f"tool_starts={dict(starts)}")
    print(f"tool_ends={dict(ends)}")
    print(f"errors={len(errors)}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1

    print("mastra custom-provider routing check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
