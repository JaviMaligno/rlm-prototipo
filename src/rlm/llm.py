from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import AzureOpenAI

from .config import AzureConfig


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[Any]
    raw: Any


class LLMClient:
    def __init__(self, config: AzureConfig) -> None:
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=model or self.config.deployment_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        return LLMResponse(
            content=message.content,
            tool_calls=message.tool_calls or [],
            raw=response,
        )
