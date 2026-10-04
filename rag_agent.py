# -*- coding: utf-8 -*-
"""RAG-агент: вопрос -> (поиск чанков) -> объединение с вопросом -> LLM.

Режимы:
  plain — вопрос напрямую в LLM, без контекста (эталон для сравнения);
  rag   — top-k чанков из векторного поиска + инструкция отвечать по контексту
          с цитированием источников [N].
"""
import logging

import grounded
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


def ask_grounded(question: str, strategy: str = "structural",
                 backend: str = "faiss", k: int = 5, pipeline: str = "full",
                 k_pre: int | None = None, min_sim: float | None = None,
                 gate_sim: float | None = None, rerank_engine: str = "llm",
                 llm_fn=None) -> dict:
    """Вопрос -> строгий структурированный ответ.

    Контракт результата:
      status:   "ok" | "low_relevance" (отсечено gate, LLM не вызывалась)
                | "model_refusal" (gate пропустил, но модель признала
                  контекст недостаточным: «не знаю» + пустые цитаты)
                | "parse_error" | "no_verified_quotes" (нарушения контракта)
      answer:   текст ответа со ссылками [N] (или «не знаю» + просьба
                уточнить вопрос при status="low_relevance");
      sources:  [{chunk_id, source, title, section}] — только чанки,
                реально процитированные моделью;
      quotes:   [{chunk_id, source, section, quote, verified}] — цитаты,
                verified: "strict" (дословная), "soft" (подменена типографика),
                "fail" (не подтверждена чанком);
      gate:     {passed, top_score, gate_sim} — барьер релевантности.

    При gate.passed=False LLM не вызывается: «не знаю» гарантировано
    детерминированно, а не надеждой на системный промпт.

    llm_fn — подмена функции генерации (тесты/прогон без API);
    сигнатура как у llm.chat, по умолчанию llm.chat.
    """
    log.info("agent-grounded: pipeline=%s strategy=%s backend=%s k=%s q=%r",
             pipeline, strategy, backend, k, question[:120])
    ms = min_sim if min_sim is not None else rerank.MIN_SIM
    gate = gate_sim if gate_sim is not None else ms
    pr = rerank.run_pipeline(
        question, mode=pipeline, strategy=strategy, backend=backend,
        k_pre=k_pre or rerank.K_PRE, k_post=k, min_sim=ms,
        rerank_engine=rerank_engine, gate_sim=gate)
    chunks = pr["chunks"]
    g = grounded.gate_relevance(chunks, gate)
    base = {"mode": "rag+grounded", "question": question, "chunks": chunks,
            "pipeline": pipeline, "query_used": pr["query_used"],
            "stats": pr["stats"], "gate": g, "model": None,
            "usage": None, "latency_s": None}
    if not g["passed"]:
        log.info("agent-grounded: gate не пройден (top=%.3f < %.3f), "
                 "LLM не вызывается", g["top_score"], g["gate_sim"])
        return {**base, "status": "low_relevance",
                "answer": grounded.refusal_answer(question, chunks, g),
                "sources": [], "quotes": []}
    call = llm_fn or llm.chat
    r = call(grounded.build_messages(question, chunks),
             max_tokens=2500, response_format={"type": "json_object"})
    parsed = grounded.parse_grounded(r["text"], chunks)
    status = "ok" if parsed["parse_ok"] else "parse_error"
    if parsed["parse_ok"] and not parsed["sources"]:
        if parsed["answer"].startswith("Не знаю"):
            # легитимный отказ модели (второй слой «не знаю»: gate пропустил,
            # но контекст вопрос не покрывает) — источники/цитаты пусты
            # по контракту, это не нарушение
            status = "model_refusal"
        else:
            # содержательный ответ без единой подтверждённой цитаты —
            # нарушение контракта
            status = "no_verified_quotes"
            log.warning("agent-grounded: ни одной подтверждённой цитаты")
    return {**base, "status": status, "answer": parsed["answer"],
            "sources": parsed["sources"], "quotes": parsed["quotes"],
            "usage": r["usage"], "latency_s": r["latency_s"],
            "model": r["model"]}
