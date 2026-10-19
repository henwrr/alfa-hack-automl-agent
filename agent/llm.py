"""Минимальный OpenAI-совместимый клиент (те же переменные окружения, что в baseline.py).

Без зависимостей: urllib. Ответ ждём как JSON в тексте — так надёжнее,
чем native tool calls у разных провайдеров.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, timeout: float = 180, max_total_tokens: int = 300_000):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "")
        self.base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.max_total_tokens = max_total_tokens
        self.used_tokens = 0
        self.calls = 0

    @property
    def available(self) -> bool:
        return bool(self.model) and self.used_tokens < self.max_total_tokens

    def chat(self, messages: list[dict], max_tokens: int = 4000, temperature: float = 0.4, retries: int = 2) -> str:
        if not self.available:
            raise LLMError("LLM недоступна: нет OPENAI_MODEL или исчерпан бюджет токенов")
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(f"{self.base_url}/chat/completions",
                                                 data=json.dumps(body).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                self.calls += 1
                usage = data.get("usage") or {}
                self.used_tokens += int(usage.get("total_tokens") or
                                        (sum(len(m["content"]) for m in messages) + max_tokens) // 3)
                return data["choices"][0]["message"].get("content") or ""
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        raise LLMError(f"LLM не ответила: {type(last).__name__}: {str(last)[:300]}".replace(self.api_key or "\0", "***"))


def extract_json(text: str) -> dict:
    """Достаём JSON-объект из ответа (убираем <think>, ```json и прочий текст)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced + ([text[text.index("{"): text.rindex("}") + 1]] if "{" in text and "}" in text else [])
    for chunk in candidates:
        try:
            value = json.loads(chunk)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("в ответе нет валидного JSON-объекта")
