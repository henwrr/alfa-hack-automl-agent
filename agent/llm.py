"""Обёртка над OpenAI-совместимым API (Qwen). LLM необязательна: при любых сбоях агент работает без неё."""
import json
import os
import re

from .config import LLM_MAX_CALLS, LLM_TIMEOUT, log, remaining


class LLM:
    def __init__(self):
        self.model = os.environ.get("OPENAI_MODEL")
        self.enabled = bool(self.model and os.environ.get("OPENAI_API_KEY"))
        self.calls = 0
        self.fails_in_row = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.client = None
        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI(timeout=LLM_TIMEOUT, max_retries=1)
            except Exception as e:  # noqa: BLE001
                log("LLM init failed:", e)
                self.enabled = False
        log("LLM enabled:", self.enabled, "model:", self.model)

    def available(self) -> bool:
        return (self.enabled and self.calls < LLM_MAX_CALLS
                and self.fails_in_row < 3 and remaining() > LLM_TIMEOUT + 60)

    def chat(self, system: str, user: str, max_tokens: int = 3000) -> str | None:
        if not self.available():
            return None
        self.calls += 1
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            if getattr(resp, "usage", None):
                self.prompt_tokens += resp.usage.prompt_tokens or 0
                self.completion_tokens += resp.usage.completion_tokens or 0
            text = resp.choices[0].message.content or ""
            self.fails_in_row = 0
            return text
        except Exception as e:  # noqa: BLE001
            self.fails_in_row += 1
            log("LLM call failed:", repr(e)[:300])
            return None

    def chat_json(self, system: str, user: str, max_tokens: int = 3000):
        text = self.chat(system + "\nОтвечай ТОЛЬКО валидным JSON без пояснений и без markdown.",
                         user, max_tokens)
        if text is None:
            return None
        return parse_json(text)


def parse_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None
