# -*- coding: utf-8 -*-
"""RAG-агент: вопрос -> (поиск чанков) -> объединение с вопросом -> LLM.

Режимы:
  plain — вопрос напрямую в LLM, без контекста (эталон для сравнения);
  rag   — top-k чанков из векторного поиска + инструкция отвечать по контексту
          с цитированием источников [N].
"""
import logging

import llm
from search import search

log = logging.getLogger("rag1c")

SYSTEM_PLAIN = (
    "Ты — эксперт-консультант по платформе 1С:Предприятие и экосистеме 1С. "
    "Отвечай на русском языке, по существу, без лишних вступлений."
)

SYSTEM_RAG = (
    "Ты — эксперт-консультант по платформе 1С:Предприятие и экосистеме 1С. "
    "Отвечай на русском языке, по существу, опираясь ТОЛЬКО на приведённый "
    "контекст. Правила:\n"
    "1. Факты бери из контекста; собственные знания — только для связности.\n"
    "2. После ключевых утверждений ставь ссылку на источник в формате [N].\n"
    "3. Если в контексте нет ответа на вопрос — прямо скажи об этом, "
    "не додумывай.\n"
    "4. Не цитируй дословно больше двух предложений подряд."
)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Контекст из чанков + вопрос -> messages для chat API."""
    parts = []
    for i, c in enumerate(chunks, 1):
        section = c["section"] or "без раздела"
        parts.append(f'[{i}] {c["title"]} — {section} ({c["source"]})\n{c["text"]}')
    context = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
    ]


def _sources(chunks: list[dict]) -> list[dict]:
    """Уникальные источники выдачи: source + title + разделы."""
    out, seen = [], set()
    for c in chunks:
        if c["source"] in seen:
            continue
        seen.add(c["source"])
        secs = sorted({x["section"] for x in chunks
                       if x["source"] == c["source"] and x["section"]})
        out.append({"source": c["source"], "title": c["title"], "sections": secs})
    return out


def ask(question: str, mode: str = "rag", strategy: str = "structural",
        backend: str = "faiss", k: int = 5) -> dict:
    """Вопрос -> ответ. mode: 'rag' | 'plain'.

    Возвращает dict(mode, question, answer, sources, chunks, usage, latency_s, model).
    """
    log.info("agent: mode=%s strategy=%s backend=%s k=%s q=%r",
             mode, strategy, backend, k, question[:120])
    if mode == "plain":
        r = llm.chat([
            {"role": "system", "content": SYSTEM_PLAIN},
            {"role": "user", "content": question},
        ])
        return {"mode": mode, "question": question, "answer": r["text"],
                "sources": [], "chunks": [], "usage": r["usage"],
                "latency_s": r["latency_s"], "model": r["model"]}
    if mode != "rag":
        raise ValueError(f"неизвестный режим: {mode!r} (ожидается rag|plain)")
    chunks = search(question, k=k, strategy=strategy, backend=backend)
    r = llm.chat(build_messages(question, chunks))
    return {"mode": mode, "question": question, "answer": r["text"],
            "sources": _sources(chunks), "chunks": chunks,
            "usage": r["usage"], "latency_s": r["latency_s"], "model": r["model"]}
