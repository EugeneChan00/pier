#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: detect-cli-proxy-gpt55-routing.sh [--base-url URL]

Checks CLI Proxy GPT-5.5 model routing for Pier harness runs.

Requires:
  CLI_PROXY_API_KEY in the environment
  curl and jq on PATH

Healthy result:
  - /v1/models advertises base model gpt-5.5
  - gpt-5.5(low) and gpt-5.5(xhigh) work on OpenAI Responses
  - gpt-5.5(max) is rejected on OpenAI Responses
  - gpt-5.5(low), gpt-5.5(xhigh), and gpt-5.5(max) work on Anthropic Messages
  - hyphenated gpt-5.5-(low/xhigh) forms are rejected
USAGE
}

base_url="https://aa.renaissancelab.org"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      base_url="${2:?--base-url requires a value}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CLI_PROXY_API_KEY:-}" ]]; then
  echo "CLI_PROXY_API_KEY is not set" >&2
  exit 2
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 2
fi

failures=0

request_status() {
  local url="$1"
  local payload="$2"
  shift 2
  local body status
  body=$(curl -sS -w $'\n__STATUS__:%{http_code}' "$@" -H "Content-Type: application/json" -d "$payload" "$url")
  status="${body##*__STATUS__:}"
  printf "%s" "$status"
}

expect_status() {
  local label="$1"
  local actual="$2"
  local expected="$3"
  if [[ "$actual" == "$expected" ]]; then
    printf "PASS %s status=%s\n" "$label" "$actual"
  else
    printf "FAIL %s status=%s expected=%s\n" "$label" "$actual" "$expected"
    failures=$((failures + 1))
  fi
}

model_count=$(curl -sS -H "Authorization: Bearer ${CLI_PROXY_API_KEY}" "${base_url%/}/v1/models" \
  | jq '[.data[]?.id | select(. == "gpt-5.5")] | length')
if [[ "$model_count" == "1" ]]; then
  echo "PASS models advertises gpt-5.5"
else
  echo "FAIL models did not advertise gpt-5.5"
  failures=$((failures + 1))
fi

for model in "gpt-5.5(low)" "gpt-5.5(xhigh)"; do
  payload=$(jq -cn --arg model "$model" '{model:$model,input:"Return exactly: ok",max_output_tokens:16}')
  status=$(request_status "${base_url%/}/v1/responses" "$payload" -H "Authorization: Bearer ${CLI_PROXY_API_KEY}")
  expect_status "openai ${model}" "$status" "200"
done

payload=$(jq -cn '{model:"gpt-5.5(max)",input:"Return exactly: ok",max_output_tokens:16}')
status=$(request_status "${base_url%/}/v1/responses" "$payload" -H "Authorization: Bearer ${CLI_PROXY_API_KEY}")
expect_status "openai gpt-5.5(max)" "$status" "400"

for model in "gpt-5.5-(low)" "gpt-5.5-(xhigh)"; do
  payload=$(jq -cn --arg model "$model" '{model:$model,input:"Return exactly: ok",max_output_tokens:16}')
  status=$(request_status "${base_url%/}/v1/responses" "$payload" -H "Authorization: Bearer ${CLI_PROXY_API_KEY}")
  expect_status "openai ${model}" "$status" "502"
done

for model in "gpt-5.5(low)" "gpt-5.5(xhigh)" "gpt-5.5(max)"; do
  payload=$(jq -cn --arg model "$model" '{model:$model,max_tokens:16,messages:[{role:"user",content:"Return exactly: ok"}]}')
  status=$(request_status "${base_url%/}/v1/messages?beta=true" "$payload" \
    -H "x-api-key: ${CLI_PROXY_API_KEY}" \
    -H "anthropic-version: 2023-06-01")
  expect_status "anthropic ${model}" "$status" "200"
done

for model in "gpt-5.5-(low)" "gpt-5.5-(xhigh)"; do
  payload=$(jq -cn --arg model "$model" '{model:$model,max_tokens:16,messages:[{role:"user",content:"Return exactly: ok"}]}')
  status=$(request_status "${base_url%/}/v1/messages?beta=true" "$payload" \
    -H "x-api-key: ${CLI_PROXY_API_KEY}" \
    -H "anthropic-version: 2023-06-01")
  expect_status "anthropic ${model}" "$status" "502"
done

if [[ "$failures" -gt 0 ]]; then
  echo "cli proxy GPT-5.5 routing check failed: ${failures} failure(s)"
  exit 1
fi

echo "cli proxy GPT-5.5 routing check passed"
