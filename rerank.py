# -*- coding: utf-8 -*-
"""Второй этап после поиска: фильтрация, LLM-реранкинг, query rewrite.

Конвейер (режим задаётся параметром mode):
  baseline: search(k_post)
  filter:   search(k_pre) -> порог similarity -> top k_post
  rerank:   search(k_pre) -> порог -> LLM-реранк -> top k_post
  rewrite:  rewrite -> search(k_post)
  full:     rewrite -> search(k_pre) -> порог -> LLM-реранк -> top k_post

LLM-реранкер — listwise: все кандидаты одним вызовом, ответ JSON
{"ranking": [...]}. Калиброванного скора у LLM нет, поэтому отсечение
нерелевантных — на retrieval-стороне (порог косинуса e5), реранкер
только переупорядочивает. Любой сбой LLM-стадии -> fallback на исходный
порядок/запрос: конвейер не должен ломать выдачу.
"""
import json
import logging
import re

import llm
from search import search

log = logging.getLogger("rag1c")

K_PRE = 20          # кандидатов из retrieval до фильтрации
K_POST = 5          # чанков уходит в контекст LLM
MIN_SIM = 0.75      # порог косинусного сходства e5
KEEP_MIN = 1        # минимум чанков в контексте даже при пустой фильтрации

PIPELINES = ("baseline", "filter", "rerank", "rewrite", "full")
RERANK_ENGINES = ("llm", "ce")

# Cross-encoder для движка "ce". Мультиязычный BGE v2-m3 (~2.3 ГБ, CPU).
# Лёгкая альтернатива: "DiTy/cross-encoder-russian-rubert-tiny2" (~50 МБ).
CE_MODEL = "BAAI/bge-reranker-v2-m3"

# ------------------------------------------------------------- порог
def apply_threshold(chunks: list[dict], min_sim: float = MIN_SIM,
                    keep_min: int = KEEP_MIN) -> list[dict]:
    """Отсечение по косинусному скору retrieval; гарантия keep_min чанков."""
    kept = [c for c in chunks if c["score"] >= min_sim]
    if len(kept) < keep_min:
        kept = chunks[:keep_min]
    return kept

# ------------------------------------------------------------- LLM-реранк
_RERANK_SYS = (
    "Ты — реранкер результатов поиска по документации 1С:Предприятие. "
    "Оцени релевантность каждого фрагмента вопросу пользователя. "
    "Верни JSON вида {\"ranking\": [номера фрагментов в порядке убывания "
    "релевантности]}. Включай только релевантные фрагменты; полностью "
    "нерелевантные — не включай. Ничего, кроме JSON, не выводи."
)
_NUMARR = re.compile(r"\[[\d\s,]+\]")


def _parse_ranking(text: str, n: int) -> list[int] | None:
    """Извлекает перестановку индексов 1..n из ответа; None при неудаче."""
    try:
        data = json.loads(text)
        ranking = data.get("ranking")
    except (json.JSONDecodeError, AttributeError):
        m = _NUMARR.search(text)
        if not m:
            return None
        try:
            ranking = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(ranking, list):
        return None
    out = []
    for x in ranking:
        if isinstance(x, int) and 1 <= x <= n and x not in out:
            out.append(x)
    return out or None


def llm_rerank(question: str, chunks: list[dict],
               k_post: int = K_POST) -> tuple[list[dict], dict]:
    """Переупорядочивание кандидатов одним listwise-вызовом LLM.

    Возвращает (top-k_post чанков, info). Fallback — исходный порядок.
    """
    n = len(chunks)
    info = {"candidates": n, "rerank": "llm", "fallback": False}
    if n <= k_post:
        return chunks[:k_post], info
    parts = []
    for i, c in enumerate(chunks, 1):
        section = c["section"] or "без раздела"
        parts.append(f'[{i}] {c["title"]} — {section} ({c["source"]})\n'
                     f'{c["text"]}')
    messages = [
        {"role": "system", "content": _RERANK_SYS},
        {"role": "user", "content":
            f"Вопрос: {question}\n\nФрагменты:\n" + "\n\n".join(parts)},
    ]
    try:
        r = llm.chat(messages, max_tokens=512,
                     response_format={"type": "json_object"})
        ranking = _parse_ranking(r["text"], n)
    except Exception as e:
        ranking = None
        log.warning("rerank: вызов LLM не удался (%s), исходный порядок", e)
    if ranking is None:
        info["fallback"] = True
        log.warning("rerank: не удалось разобрать ranking, исходный порядок")
        return chunks[:k_post], info
    # сначала ранжированные LLM, затем неоценённые в исходном порядке
    ordered = [chunks[i - 1] for i in ranking]
    ordered += [c for j, c in enumerate(chunks, 1) if j not in ranking]
    info["llm_ranked"] = len(ranking)
    return ordered[:k_post], info

# ------------------------------------------------------------- CE-реранк
_ce_model = None


def _get_ce():
    """Ленивая загрузка cross-encoder'а (модель качается один раз)."""
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder
        log.info("загрузка cross-encoder %s …", CE_MODEL)
        _ce_model = CrossEncoder(CE_MODEL, device="cpu")
    return _ce_model


def ce_rerank(question: str, chunks: list[dict],
              k_post: int = K_POST) -> tuple[list[dict], dict]:
    """Cross-encoder: скор каждой пары (вопрос, чанк), сортировка по убыванию.

    Скор (логит) сохраняется в c["ce_score"]. Fallback — исходный порядок.
    """
    n = len(chunks)
    info = {"candidates": n, "rerank": "ce", "fallback": False,
            "ce_model": CE_MODEL}
    if n <= k_post:
        return chunks[:k_post], info
    try:
        scores = _get_ce().predict([[question, c["text"]] for c in chunks])
    except Exception as e:
        log.warning("rerank-ce: %s, исходный порядок", e)
        info["fallback"] = True
        return chunks[:k_post], info
    order = sorted(range(n), key=lambda i: -float(scores[i]))
    ranked = []
    for i in order:
        c = dict(chunks[i])
        c["ce_score"] = round(float(scores[i]), 4)
        ranked.append(c)
    info["ce_top"] = ranked[0]["ce_score"]
    info["ce_cut"] = ranked[k_post - 1]["ce_score"]
    return ranked[:k_post], info

# ------------------------------------------------------------- rewrite
_REWRITE_SYS = (
    "Ты переписываешь вопрос пользователя в точный поисковый запрос для "
    "векторного поиска по русскоязычной документации 1С:Предприятие. "
    "Сохрани смысл вопроса; приведи термины к точной номенклатуре 1С "
    "(имена объектов метаданных, регистров, документов, механизмов); "
    "убери разговорные обороты. Верни только текст запроса, одной строкой, "
    "без пояснений и кавычек."
)


def rewrite_query(question: str) -> tuple[str, bool]:
    """Переписывание вопроса в поисковый запрос. (запрос, ok)"""
    try:
        r = llm.chat([
            {"role": "system", "content": _REWRITE_SYS},
            {"role": "user", "content": question},
        ], max_tokens=256)
        q = r["text"].strip().strip('"').split("\n")[0].strip()
        if q and len(q) >= 3:
            log.info("rewrite: %r -> %r", question[:80], q[:80])
            return q, True
    except Exception as e:
        log.warning("rewrite: вызов LLM не удался (%s), исходный вопрос", e)
    return question, False

# ------------------------------------------------------------- конвейер
def run_pipeline(question: str, mode: str = "baseline",
                 strategy: str = "structural", backend: str = "faiss",
                 k_pre: int = K_PRE, k_post: int = K_POST,
                 min_sim: float = MIN_SIM,
                 rerank_engine: str = "llm",
                 gate_sim: float | None = None) -> dict:
    """Вопрос -> dict(query_used, chunks, stats). mode из PIPELINES,
    rerank_engine из RERANK_ENGINES ('llm' | 'ce').

    gate_sim — ранний барьер релевантности: если топ-скор retrieval ниже
    порога, конвейер завершается ДО стадии реранка (не тратим вызов LLM
    на заведомо слабый контекст); stats["gate_passed"]=False, в chunks
    возвращаются кандидаты retrieval (для диагностики/подсказки темы).
    """
    if mode not in PIPELINES:
        raise ValueError(f"неизвестный режим конвейера: {mode!r} "
                         f"(ожидается {'/'.join(PIPELINES)})")
    if rerank_engine not in RERANK_ENGINES:
        raise ValueError(f"неизвестный реранкер: {rerank_engine!r} "
                         f"(ожидается {'/'.join(RERANK_ENGINES)})")
    stats = {"pipeline": mode, "rewritten": False, "candidates": None,
             "after_threshold": None, "k_pre": k_pre, "k_post": k_post,
             "min_sim": min_sim, "rerank_fallback": False,
             "rerank_engine": rerank_engine, "gate_passed": None}

    query_used = question
    if mode in ("rewrite", "full"):
        query_used, ok = rewrite_query(question)
        stats["rewritten"] = ok

    wide = mode in ("filter", "rerank", "full")
    candidates = search(query_used, k=k_pre if wide else k_post,
                        strategy=strategy, backend=backend)
    if wide:
        stats["candidates"] = len(candidates)
        candidates = apply_threshold(candidates, min_sim)
        stats["after_threshold"] = len(candidates)

    if gate_sim is not None:
        top = max((float(c.get("score", 0.0)) for c in candidates),
                  default=0.0)
        stats["gate_passed"] = bool(candidates) and top >= gate_sim
        if not stats["gate_passed"]:
            stats["final"] = 0
            log.info("pipeline: gate %.3f < %.3f, реранк пропущен",
                     top, gate_sim)
            return {"query_used": query_used, "chunks": candidates[:k_post],
                    "stats": stats}

    if mode in ("rerank", "full"):
        rerank_fn = ce_rerank if rerank_engine == "ce" else llm_rerank
        chunks, info = rerank_fn(question, candidates, k_post)
        stats["rerank_fallback"] = info["fallback"]
    else:
        chunks = candidates[:k_post]

    stats["final"] = len(chunks)
    return {"query_used": query_used, "chunks": chunks, "stats": stats}
