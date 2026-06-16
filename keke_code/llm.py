"""OpenAI-compatible chat client used by the agent.

核心原理：CLI Agent 通常把“模型供应商”隔离在单独模块中。这样上层 Agent Loop
只依赖 chat(messages) 这个抽象，底层可以切换 OpenAI、Claude、私有网关或本地模型。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


class LLMError(RuntimeError):
    """Raised when the remote model cannot be reached or returns invalid data."""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class OpenAICompatibleClient:
    """Tiny stdlib client for /chat/completions APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("KEKE_CODE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("KEKE_CODE_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("KEKE_CODE_MODEL") or "gpt-4o-mini"
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[ChatMessage]) -> str:
        if not self.api_key:
            raise LLMError("missing API key; set KEKE_CODE_API_KEY or OPENAI_API_KEY")

        payload = {
            "model": self.model,
            "messages": [message.__dict__ for message in messages],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"LLM HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response: {data}") from exc
