# -*- coding: utf-8 -*-
"""Тонкий клиент DeepSeek API (OpenAI-совместимый).

Ключ берётся ТОЛЬКО из переменной окружения DEEPSEEK_API_KEY;
в коде, конфигах и логах ключ не хранится и не выводится.

Документация: https://api-docs.deepseek.com/
"""
import logging
import os
import time

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
TIMEOUT = 120          # секунд на один вызов
MAX_RETRIES = 2        # повторы при 429/5xx/таймауте

log = logging.getLogger("rag1c")

_client = None


def api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Не задана переменная окружения DEEPSEEK_API_KEY. "
            "Windows (постоянно): setx DEEPSEEK_API_KEY \"sk-...\" и открыть "
            "новую консоль; текущая сессия PowerShell: "
            "$env:DEEPSEEK_API_KEY=\"sk-...\".")
    return key


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=api_key(), base_url=BASE_URL, timeout=TIMEOUT)
    return _client


def chat(messages: list[dict], *, model: str = MODEL, temperature: float = 0.0,
         max_tokens: int = 2048) -> dict:
    """Один вызов chat/completions (non-stream, thinking выключен).

    Возвращает dict(text, model, usage{prompt,completion,total}, latency_s).
    """
    t0 = time.monotonic()
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
            dt = time.monotonic() - t0
            u = resp.usage
            usage = {
                "prompt": getattr(u, "prompt_tokens", 0),
                "completion": getattr(u, "completion_tokens", 0),
                "total": getattr(u, "total_tokens", 0),
            }
            log.info("llm %s: tokens in=%s out=%s, %.1fs",
                     model, usage["prompt"], usage["completion"], dt)
            return {"text": (resp.choices[0].message.content or "").strip(),
                    "model": model, "usage": usage, "latency_s": round(dt, 2)}
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_err = e
            log.warning("llm попытка %d не удалась: %s", attempt + 1, e)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"DeepSeek API: запрос не удался после {MAX_RETRIES + 1} попыток: {last_err}")
