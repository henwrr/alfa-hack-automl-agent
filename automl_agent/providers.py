"""Provider adapters exposing a canonical OpenAI-style tool-call response."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from concurrent.futures import Future
from typing import Any

import httpx

from .contracts import ModelResponse


class ProviderError(RuntimeError):
    """A provider failure classified for bounded retry and caller handling."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class _HTTPProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        retries: int,
        extra_body: dict | None,
        *,
        ollama: bool,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if not isinstance(retries, int) or retries < 0:
            raise ValueError("retries must be a non-negative integer")
        if extra_body is not None and not isinstance(extra_body, dict):
            raise TypeError("extra_body must be a dict or None")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.retries = retries
        self.extra_body = dict(extra_body or {})
        self.ollama = ollama
        if ollama:
            if self.base_url.endswith("/v1"):
                self.base_url = self.base_url[:-3]
            self.endpoint = self.base_url + "/api/chat"
        else:
            self.endpoint = self.base_url + "/chat/completions"

    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        timeout: float,
        max_tokens: int,
    ) -> ModelResponse:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise TypeError("messages and tools must be lists")
        coroutine = self._complete_async(messages, tools, float(timeout), max_tokens)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        # The public interface is synchronous. If called from an async host,
        # run the request in a private event loop so we do not nest asyncio.run.
        result: Future[ModelResponse] = Future()

        def run_in_thread() -> None:
            try:
                result.set_result(asyncio.run(coroutine))
            except BaseException as exc:
                result.set_exception(exc)

        thread = threading.Thread(target=run_in_thread, name="automl-provider", daemon=True)
        thread.start()
        try:
            return result.result(timeout=float(timeout) + 0.5)
        except TimeoutError:
            raise ProviderError("provider request timed out", code="timeout", retryable=True) from None

    async def _complete_async(
        self,
        messages: list[dict],
        tools: list[dict],
        timeout: float,
        max_tokens: int,
    ) -> ModelResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = self._payload(messages, tools, max_tokens)
        try:
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(timeout=None) as client:
                    for attempt in range(self.retries + 1):
                        try:
                            response = await client.post(self.endpoint, headers=headers, json=payload)
                        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                                httpx.WriteError, httpx.RemoteProtocolError) as exc:
                            if attempt >= self.retries:
                                raise ProviderError(
                                    "provider transport failed", code="transport_error", retryable=True
                                ) from None
                            await asyncio.sleep(min(0.1 * (2**attempt), 0.8))
                            continue
                        if response.is_success:
                            try:
                                body = response.json()
                            except (ValueError, json.JSONDecodeError):
                                raise ProviderError(
                                    "provider returned invalid JSON", code="invalid_response", retryable=False
                                ) from None
                            return self._normalize(body)
                        error = self._http_error(response.status_code, response.content)
                        if error.retryable and attempt < self.retries:
                            await asyncio.sleep(min(0.1 * (2**attempt), 0.8))
                            continue
                        raise error
        except TimeoutError:
            raise ProviderError("provider request timed out", code="timeout", retryable=True) from None
        except httpx.InvalidURL:
            raise ProviderError("provider URL is invalid", code="invalid_url", retryable=False) from None
        except httpx.RequestError as exc:
            transient = isinstance(
                exc,
                (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError),
            )
            code = "transport_error" if transient else "request_error"
            raise ProviderError("provider transport failed", code=code, retryable=transient) from None
        raise ProviderError("provider request failed", code="unknown", retryable=False)

    def _payload(self, messages: list[dict], tools: list[dict], max_tokens: int) -> dict[str, Any]:
        raise NotImplementedError

    def _normalize(self, body: Any) -> ModelResponse:
        raise NotImplementedError

    @staticmethod
    def _http_error(status: int, body: bytes) -> ProviderError:
        text = body[:8192].decode("utf-8", errors="ignore").lower()
        if status == 401:
            code, message, retryable = "unauthorized", "provider rejected credentials", False
        elif status == 403:
            code, message, retryable = "forbidden", "provider denied the request", False
        elif status == 400 and (
            "context_length_exceeded" in text
            or "maximum context length" in text
            or "context length is" in text
            or "too many tokens" in text
        ):
            code, message, retryable = "context_length_exceeded", "provider context limit exceeded", False
        elif status == 429:
            code, message, retryable = "rate_limit", "provider rate limit", True
        elif status in (408, 425) or status >= 500:
            code, message, retryable = "server_error", "temporary provider error", True
        else:
            code, message, retryable = f"http_{status}", f"provider returned HTTP {status}", False
        return ProviderError(message, code=code, retryable=retryable)


class OpenAICompatibleProvider(_HTTPProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        retries: int = 2,
        extra_body: dict | None = None,
    ) -> None:
        super().__init__(base_url, api_key, model, retries, extra_body, ollama=False)

    def _payload(self, messages: list[dict], tools: list[dict], max_tokens: int) -> dict[str, Any]:
        payload = dict(self.extra_body)
        payload.update({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "parallel_tool_calls": False,
            "max_tokens": max_tokens,
        })
        return payload

    def _normalize(self, body: Any) -> ModelResponse:
        try:
            choice = body["choices"][0]
            message = dict(choice["message"])
            message.setdefault("role", "assistant")
            message["tool_calls"] = _canonical_tool_calls(message.get("tool_calls", []))
            usage = body.get("usage") or {}
            prompt_tokens = _int_or_zero(usage.get("prompt_tokens"))
            completion_tokens = _int_or_zero(usage.get("completion_tokens"))
            total_tokens = _int_or_zero(usage.get("total_tokens"), prompt_tokens + completion_tokens)
            finish_reason = choice.get("finish_reason") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError("provider response shape is invalid", code="invalid_response", retryable=False) from None
        return ModelResponse(
            message=message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=str(finish_reason),
        )


class OllamaProvider(_HTTPProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        retries: int = 2,
        extra_body: dict | None = None,
    ) -> None:
        super().__init__(base_url, api_key, model, retries, extra_body, ollama=True)

    def _payload(self, messages: list[dict], tools: list[dict], max_tokens: int) -> dict[str, Any]:
        extra = dict(self.extra_body)
        extra_options = extra.pop("options", {})
        payload: dict[str, Any] = {
            **extra,
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": False,
            "options": {**(extra_options if isinstance(extra_options, dict) else {}), "num_predict": max_tokens},
        }
        return payload

    def _normalize(self, body: Any) -> ModelResponse:
        try:
            message = dict(body["message"])
            message.setdefault("role", "assistant")
            message["tool_calls"] = _canonical_tool_calls(message.get("tool_calls", []))
            prompt_tokens = _int_or_zero(body.get("prompt_eval_count"))
            completion_tokens = _int_or_zero(body.get("eval_count"))
            total_tokens = prompt_tokens + completion_tokens
            finish_reason = body.get("done_reason") or ("stop" if body.get("done") else "")
        except (KeyError, TypeError, AttributeError):
            raise ProviderError("provider response shape is invalid", code="invalid_response", retryable=False) from None
        return ModelResponse(
            message=message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=str(finish_reason),
        )


def _int_or_zero(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def _canonical_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if tool_calls is None:
        return []
    if not isinstance(tool_calls, list):
        raise ProviderError("provider tool calls are invalid", code="invalid_response", retryable=False)
    normalized = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ProviderError("provider tool call is invalid", code="invalid_response", retryable=False)
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ProviderError("provider tool call is invalid", code="invalid_response", retryable=False)
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                raise ProviderError("provider tool arguments are invalid", code="invalid_response", retryable=False) from None
        normalized.append({
            "id": call.get("id") or f"call_{uuid.uuid4().hex}",
            "type": "function",
            "function": {"name": function["name"], "arguments": arguments},
        })
    return normalized
