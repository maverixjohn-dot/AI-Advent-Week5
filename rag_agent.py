# -*- coding: utf-8 -*-
"""RAG-агент: вопрос -> (поиск чанков) -> объединение с вопросом -> LLM.

Режимы:
  plain — вопрос напрямую в LLM, без контекста (эталон для сравнения);
  rag   — top-k чанков из векторного поиска + инструкция отвечать по контексту
          с цитированием источников [N].
"""
import logging

import llm
import rerank

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
        backend: str = "faiss", k: int = 5, pipeline: str = "baseline",
        k_pre: int | None = None, min_sim: float | None = None,
        rerank_engine: str = "llm") -> dict:
    """Вопрос -> ответ. mode: 'rag' | 'plain'; pipeline — режим конвейера
    из rerank.PIPELINES (baseline/filter/rerank/rewrite/full).

    Возвращает dict(mode, question, answer, sources, chunks, usage,
    latency_s, model, pipeline, query_used, stats).
    """
    log.info("agent: mode=%s pipeline=%s strategy=%s backend=%s k=%s q=%r",
             mode, pipeline, strategy, backend, k, question[:120])
    if mode == "plain":
        r = llm.chat([
            {"role": "system", "content": SYSTEM_PLAIN},
            {"role": "user", "content": question},
        ])
        return {"mode": mode, "question": question, "answer": r["text"],
                "sources": [], "chunks": [], "usage": r["usage"],
                "latency_s": r["latency_s"], "model": r["model"],
                "pipeline": None, "query_used": question, "stats": None}
    if mode != "rag":
        raise ValueError(f"неизвестный режим: {mode!r} (ожидается rag|plain)")
    pr = rerank.run_pipeline(
        question, mode=pipeline, strategy=strategy, backend=backend,
        k_pre=k_pre or rerank.K_PRE, k_post=k,
        min_sim=min_sim if min_sim is not None else rerank.MIN_SIM,
        rerank_engine=rerank_engine)
    chunks = pr["chunks"]
    r = llm.chat(build_messages(question, chunks))
    return {"mode": mode, "question": question, "answer": r["text"],
            "sources": _sources(chunks), "chunks": chunks,
            "usage": r["usage"], "latency_s": r["latency_s"],
            "model": r["model"], "pipeline": pipeline,
            "query_used": pr["query_used"], "stats": pr["stats"]}
