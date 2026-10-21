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
    def __init__(self, timeout: float | None = None, max_total_tokens: int | None = None):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "")
        self.base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        # Локальная Ollama: нативный /api/chat (можно выключить «thinking» и задать размер контекста).
        self.provider = os.getenv("AUTOML_PROVIDER") or ("ollama" if ":11434" in self.base_url else "openai")
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", 16384))
        try:
            self.extra_body = json.loads(os.getenv("AGENT_EXTRA_BODY", "{}"))
        except json.JSONDecodeError:
            self.extra_body = {}
        self.timeout = timeout or float(os.getenv("AGENT_LLM_TIMEOUT", 900 if self.provider == "ollama" else 240))
        self.max_total_tokens = max_total_tokens or int(os.getenv("AGENT_MAX_TOKENS", 400_000))
        self.used_tokens = 0
        self.calls = 0
        self.last_prompt_tokens = 0

    @property
    def available(self) -> bool:
        return bool(self.model) and self.used_tokens < self.max_total_tokens

    def chat(self, messages: list[dict], max_tokens: int = 4000, temperature: float = 0.4, retries: int = 2,
             schema: dict | None = None) -> str:
        """schema — JSON Schema ответа: Ollama получает её в `format` (жёстко ограничивает вывод),
        OpenAI-совместимый API — в `response_format`, только если AGENT_STRUCTURED=1 (поддержка у
        провайдеров разная, без флага не рискуем получить 400)."""
        if not self.available:
            raise LLMError("LLM недоступна: нет OPENAI_MODEL или исчерпан бюджет токенов")
        if self.provider == "ollama":
            url = self.base_url.removesuffix("/v1") + "/api/chat"
            body = {"model": self.model, "messages": messages, "stream": False, "think": False,
                    "options": {"num_ctx": self.num_ctx, "num_predict": max_tokens, "temperature": temperature}}
            if schema is not None and os.getenv("AGENT_STRUCTURED", "1") != "0":
                body["format"] = schema
        else:
            url = f"{self.base_url}/chat/completions"
            body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                    **self.extra_body}
            if schema is not None and os.getenv("AGENT_STRUCTURED") == "1":
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "proposal", "schema": schema}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(url,
                                                 data=json.dumps(body).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                self.calls += 1
                if self.provider == "ollama":
                    if "error" in data:
                        raise LLMError(f"Ollama: {data['error']}")
                    self.used_tokens += int(data.get("prompt_eval_count", 0)) + int(data.get("eval_count", 0))
                    self.last_prompt_tokens = int(data.get("prompt_eval_count", 0))
                    return data["message"].get("content") or ""
                usage = data.get("usage") or {}
                self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
                self.used_tokens += int(usage.get("total_tokens") or
                                        (sum(len(m["content"]) for m in messages) + max_tokens) // 3)
                return data["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code < 500 and exc.code != 429:
                    break
                time.sleep(2 * (attempt + 1))
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
