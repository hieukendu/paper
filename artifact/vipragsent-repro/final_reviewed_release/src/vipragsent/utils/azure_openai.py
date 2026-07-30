from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class AzureOpenAIError(RuntimeError):
    """Raised when an Azure OpenAI request cannot be completed."""


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str


def azure_configured(kind: str = "label") -> bool:
    try:
        config_from_env(kind)
    except AzureOpenAIError:
        return False
    return True


def config_from_env(kind: str = "label") -> AzureOpenAIConfig:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
    deployment = deployment_from_env(kind)
    missing = []
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not api_version:
        missing.append("AZURE_OPENAI_API_VERSION")
    if not deployment:
        missing.append(deployment_env_name(kind))
    if missing:
        raise AzureOpenAIError("Missing Azure OpenAI env var(s): " + ", ".join(missing))
    return AzureOpenAIConfig(endpoint=endpoint, api_key=api_key, api_version=api_version, deployment=deployment)


def deployment_env_name(kind: str) -> str:
    if kind == "rationale":
        return "AZURE_OPENAI_DEPLOYMENT_RATIONALE"
    return "AZURE_OPENAI_DEPLOYMENT_LABEL"


def deployment_from_env(kind: str) -> str:
    preferred = os.getenv(deployment_env_name(kind), "").strip()
    if preferred:
        return preferred
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()


def chat_completion(
    messages: list[dict[str, str]],
    *,
    kind: str = "label",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 500,
    json_mode: bool = False,
) -> dict[str, Any]:
    config = config_from_env(kind)
    deployment = urllib.parse.quote(config.deployment, safe="")
    url = (
        f"{config.endpoint}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={urllib.parse.quote(config.api_version)}"
    )
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": config.api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AzureOpenAIError(f"Azure OpenAI HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise AzureOpenAIError(f"Azure OpenAI request failed: {exc.reason}") from exc


def first_message_content(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise AzureOpenAIError("Azure OpenAI response did not contain choices[0].message.content") from exc
