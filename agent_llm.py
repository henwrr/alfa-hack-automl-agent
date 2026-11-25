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
        # Платформа может не передать OPENAI_MODEL (по инструкции он необязателен) — тогда без модели агент
        # молча ушёл бы в offline. Есть ключ -> модель организаторов по умолчанию.
        self.model = os.getenv("OPENAI_MODEL") or ("qwen/qwen3.8-27b" if self.api_key else "")
        self.base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        # Локальная Ollama: нативный /api/chat (можно выключить «thinking» и задать размер контекста).
        self.provider = os.getenv("AUTOML_PROVIDER") or ("ollama" if ":11434" in self.base_url else "openai")
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", 16384))
        # Qwen организаторов по умолчанию «думает»: на коротком запросе 295 из 300 токенов ушли в reasoning, ответ
        # пустой; выходные токены в ~30 раз дороже входных. reasoning.enabled=false проверен на прокси — 0 токенов
        # рассуждений, тот же ответ. Переопределяется AGENT_EXTRA_BODY (например '{}' — вернуть рассуждения).
        try:
            self.extra_body = json.loads(os.getenv("AGENT_EXTRA_BODY", '{"reasoning": {"enabled": false}}'))
        except json.JSONDecodeError:
            self.extra_body = {}
        self.max_cost = float(os.getenv("AGENT_MAX_COST", 60))  # ₽ на запуск: дневной бюджет 600 ₽ общий на команду
        self.cost = 0.0
        self.timeout = timeout or float(os.getenv("AGENT_LLM_TIMEOUT", 900 if self.provider == "ollama" else 240))
        self.max_total_tokens = max_total_tokens or int(os.getenv("AGENT_MAX_TOKENS", 400_000))
        self.used_tokens = 0
        self.calls = 0
        self.last_prompt_tokens = 0

    @property
    def available(self) -> bool:
        return bool(self.model) and self.used_tokens < self.max_total_tokens and self.cost < self.max_cost

    def chat(self, messages: list[dict], max_tokens: int = 4000, temperature: float = 0.4, retries: int = 2,
             schema: dict | None = None) -> str:
        """schema — JSON Schema ответа: Ollama получает её в `format` (жёстко ограничивает вывод),
        OpenAI-совместимый API — в `response_format` (прокси организаторов json_schema поддерживает; выключить —
        AGENT_STRUCTURED=0). Если провайдер отвергнет response_format/extra_body (HTTP 400) — повтор без них."""
        if not self.available:
            raise LLMError("LLM недоступна: нет OPENAI_MODEL или исчерпан бюджет токенов/рублей")
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
            if schema is not None and os.getenv("AGENT_STRUCTURED", "1") != "0":
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
                self.cost += float(usage.get("cost") or 0.0)  # прокси организаторов отдаёт стоимость в рублях
                self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
                self.used_tokens += int(usage.get("total_tokens") or
                                        (sum(len(m["content"]) for m in messages) + max_tokens) // 3)
                return data["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code == 400 and self.provider != "ollama" and ("response_format" in body or self.extra_body):
                    # провайдер не знает необязательных полей — убираем их и повторяем
                    body.pop("response_format", None)
                    for k in self.extra_body:
                        body.pop(k, None)
                    self.extra_body = {}
                    continue
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
