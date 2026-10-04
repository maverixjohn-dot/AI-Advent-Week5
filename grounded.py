# -*- coding: utf-8 -*-
"""Строгий контракт ответа RAG: ответ + источники (source/section/chunk_id)
+ дословные цитаты из чанков, и режим «не знаю» при слабом контексте.

Поток:
  1. gate_relevance(chunks, gate_sim) — детерминированный барьер ДО вызова LLM:
     если топ-скор retrieval ниже порога, модель не вызывается вовсе,
     ассистент обязан ответить «не знаю» и попросить уточнение.
  2. build_messages — промпт, требующий JSON {answer, citations:[{chunk,quote}]}.
  3. parse_grounded — разбор ответа LLM: извлечение JSON, проверка, что каждая
     цитата — ДОСЛОВНЫЙ фрагмент указанного чанка (верификация подстрокой),
     сборка списка источников из реально процитированных чанков.

Источники собираются только из процитированных чанков: ссылка на чанк,
который модель не использовала, — ложная атрибуция.
"""
import json
import logging
import re

log = logging.getLogger("rag1c")

# ------------------------------------------------------------- промпт
SYSTEM_GROUNDED = (
    "Ты — эксперт-консультант по платформе 1С:Предприятие и экосистеме 1С. "
    "Отвечаешь на русском языке СТРОГО на основе приведённого контекста.\n"
    "Формат ответа — ТОЛЬКО валидный JSON (без markdown-ограждений, без "
    "текста вокруг):\n"
    '{\n'
    '  "answer": "текст ответа; после каждого ключевого утверждения — '
    'ссылка на источник в формате [N]",\n'
    '  "citations": [{"chunk": N, "quote": "дословный фрагмент чанка N"}, ...]\n'
    '}\n'
    "Правила:\n"
    "1. Факты — только из контекста. Собственные знания — только для "
    "связности изложения, не как источник утверждений.\n"
    "2. citations обязательны: минимум одна цитата на каждый чанк, "
    "который реально использован в answer.\n"
    "3. quote — ДОСЛОВНЫЙ фрагмент чанка (копируй символ в символ, 1–2 "
    "предложения, до 300 знаков). Перефразирование в quote запрещено: "
    "цитата проверяется автоматически как подстрока чанка.\n"
    "4. Не приводи более 5 цитат; выбирай самые информативные.\n"
    "5. Если в контексте нет ответа на вопрос — честно напиши в answer: "
    "«Не знаю: в предоставленном контексте нет ответа на этот вопрос. "
    "Уточните вопрос.» — и верни пустой массив citations. Додумывать "
    "запрещено."
)


# Промпт для режима чата с памятью: тот же контракт (answer + дословные
# citations), плюс обязательное поле "memory" — снимок памяти задачи.
# Вход дополнен блоками «Память задачи» и «История диалога».
SYSTEM_GROUNDED_MEM = (
    "Ты — эксперт-консультант по платформе 1С:Предприятие и экосистеме 1С, "
    "ведущий многовходовой диалог. Отвечаешь на русском языке СТРОГО на "
    "основе приведённого контекста.\n"
    "Формат ответа — ТОЛЬКО валидный JSON (без markdown-ограждений, без "
    "текста вокруг):\n"
    '{\n'
    '  "answer": "текст ответа; после каждого ключевого утверждения — '
    'ссылка на источник в формате [N]",\n'
    '  "citations": [{"chunk": N, "quote": "дословный фрагмент чанка N"}, ...],\n'
    '  "memory": {"goal": "цель диалога одной фразой",\n'
    '             "clarifications": ["что пользователь уточнил", ...],\n'
    '             "constraints": ["зафиксированные ограничения/условия", ...],\n'
    '             "terms": ["термины и имена объектов", ...]}\n'
    '}\n'
    "Правила:\n"
    "1. Факты — только из контекста. Собственные знания — только для "
    "связности изложения, не как источник утверждений.\n"
    "2. citations обязательны: минимум одна цитата на каждый чанк, "
    "который реально использован в answer.\n"
    "3. quote — ДОСЛОВНЫЙ фрагмент чанка (копируй символ в символ, 1–2 "
    "предложения, до 300 знаков). Перефразирование в quote запрещено: "
    "цитата проверяется автоматически как подстрока чанка.\n"
    "4. Не приводи более 5 цитат; выбирай самые информативные.\n"
    "5. Если в контексте нет ответа на вопрос — честно напиши в answer: "
    "«Не знаю: в предоставленном контексте нет ответа на этот вопрос. "
    "Уточните вопрос.» — и верни пустой массив citations. Додумывать "
    "запрещено. Поле memory при этом всё равно заполняй.\n"
    "6. Учитывай «Память задачи» и «Историю диалога»: вопросы-уточнения "
    "понимай в рамках зафиксированной цели и ограничений.\n"
    "7. memory — полный снимок памяти задачи ПОСЛЕ текущего хода. Переноси "
    "в него всё актуальное из прежней памяти и добавляй новое из текущего "
    "сообщения. Если цель не изменилась — верни её без изменений. Не "
    "отбрасывай ранее зафиксированные ограничения и термины без явной "
    "причины в словах пользователя.\n"
    "8. Каждая запись memory — короткая фраза (до 150 знаков), без "
    "дублирования одного и того же разными словами."
)


def _memory_block(task_state: str, history: str) -> str:
    return (f"Память задачи:\n{task_state}\n\n"
            f"История диалога:\n{history}\n\n")


def build_messages(question: str, chunks: list[dict], history: str = "",
                   task_state: str = "", memory: bool = False) -> list[dict]:
    """Контекст из чанков + вопрос -> messages для chat API (JSON-режим).

    history/task_state — блоки памяти диалога (мини-чат); memory=True
    переключает системный промпт на вариант с полем "memory" в контракте.
    """
    parts = []
    for i, c in enumerate(chunks, 1):
        section = c["section"] or "без раздела"
        parts.append(f'[{i}] {c["title"]} — {section} '
                     f'(source={c["source"]}, chunk_id={c["chunk_id"]})\n'
                     f'{c["text"]}')
    context = "\n\n".join(parts)
    prefix = _memory_block(task_state, history) if memory else ""
    system = SYSTEM_GROUNDED_MEM if memory else SYSTEM_GROUNDED
    return [
        {"role": "system", "content": system},
        {"role": "user", "content":
            f"{prefix}Контекст:\n{context}\n\nВопрос: {question}"},
    ]


# ------------------------------------------------------------- gate
def gate_relevance(chunks: list[dict], gate_sim: float) -> dict:
    """Детерминированный барьер релевантности.

    Возвращает dict(passed, top_score, gate_sim). passed=False — LLM не
    вызывается, ассистент обязан ответить «не знаю».
    """
    top = max((float(c.get("score", 0.0)) for c in chunks), default=0.0)
    return {"passed": bool(chunks) and top >= gate_sim,
            "top_score": round(top, 4), "gate_sim": gate_sim}


def refusal_answer(question: str, chunks: list[dict], gate: dict) -> str:
    """Текст отказа: «не знаю» + явная просьба уточнить вопрос."""
    hint = ""
    if chunks:
        c = chunks[0]
        section = c["section"] or "без раздела"
        hint = (f" Ближайшая по смыслу тема в базе — «{c['title']} — "
                f"{section}» (сходство {gate['top_score']:.2f}), но и она "
                f"не покрывает вопрос.")
    return (
        "Не знаю: в базе знаний нет достаточно релевантной информации по "
        f"этому вопросу (максимальное сходство {gate['top_score']:.2f} ниже "
        f"порога {gate['gate_sim']:.2f}).{hint} Пожалуйста, уточните вопрос: "
        "приведите точные термины 1С, имена объектов метаданных, документов "
        "или механизмов, о которых спрашиваете."
    )


# ------------------------------------------------------------- разбор ответа
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _extract_json(text: str) -> dict | None:
    """JSON-объект из ответа LLM; None при неудаче. Устойчив к
    markdown-ограждениям, тексту вокруг объекта и висячим запятым."""
    t = _FENCE.sub("", text.strip())
    try:
        data = json.loads(t)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    frag = t[start:end + 1]
    for candidate in (frag, _TRAILING_COMMA.sub(r"\1", frag)):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Нормализация для строгой проверки цитаты: схлопывание whitespace."""
    return _WS.sub(" ", s).strip()


def _norm_soft(s: str) -> str:
    """Мягкая нормализация: кавычки/тире/ё — для диагностики «почти дословных»
    цитат (LLM подменил типографику). В verified не засчитывается."""
    t = _norm(s)
    for a, b in (("«", '"'), ("»", '"'), ("“", '"'), ("”", '"'),
                 ("„", '"'), ("—", "-"), ("–", "-"), ("ё", "е"), ("Ё", "Е")):
        t = t.replace(a, b)
    return t


def verify_quote(quote: str, chunk_text: str) -> str:
    """'strict' — дословная подстрока (с точностью до whitespace);
    'soft' — совпала только после замены типографики; 'fail' — не цитата."""
    if not quote or not chunk_text:
        return "fail"
    if _norm(quote) in _norm(chunk_text):
        return "strict"
    if _norm_soft(quote) in _norm_soft(chunk_text):
        return "soft"
    return "fail"


def parse_memory(data: dict) -> dict | None:
    """Поле "memory" из ответа LLM -> dict(goal, clarifications,
    constraints, terms) или None. Невалидные элементы списков отбрасываются;
    дальнейшая защита от потерь — merge_memory() в chat_memory."""
    if not isinstance(data, dict):  # json.loads мог вернуть list/None/...
        return None
    mem = data.get("memory")
    if not isinstance(mem, dict):
        return None
    out = {"goal": str(mem.get("goal") or "").strip()}
    for field in ("clarifications", "constraints", "terms"):
        raw = mem.get(field)
        out[field] = ([x.strip() for x in raw
                       if isinstance(x, str) and x.strip()]
                      if isinstance(raw, list) else [])
    return out


def parse_grounded(text: str, chunks: list[dict]) -> dict:
    """Ответ LLM -> dict(parse_ok, answer, quotes, sources, memory).

    quotes:  [{chunk_id, source, section, quote, verified: strict|soft|fail}]
    sources: [{chunk_id, source, title, section}] — только чанки, из которых
              есть хотя бы одна цитата strict/soft (fail-цитаты источник
              не подтверждают).
    memory:  снимок памяти задачи (только в режиме SYSTEM_GROUNDED_MEM),
              иначе None.
    """
    data = _extract_json(text)
    if data is None:
        log.warning("grounded: не удалось разобрать JSON ответа")
        return {"parse_ok": False, "answer": text.strip(),
                "quotes": [], "sources": [], "memory": None}
    answer = str(data.get("answer") or "").strip()
    raw_cit = data.get("citations")
    if not isinstance(raw_cit, list):
        raw_cit = []
    quotes, sources, seen = [], [], set()
    for item in raw_cit:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("chunk"))
        except (TypeError, ValueError):
            continue
        quote = str(item.get("quote") or "").strip()
        if not (1 <= idx <= len(chunks)) or not quote:
            log.warning("grounded: цитата с недопустимым chunk=%r пропущена",
                        item.get("chunk"))
            continue
        c = chunks[idx - 1]
        verdict = verify_quote(quote, c["text"])
        if verdict == "fail":
            log.warning("grounded: цитата не подтверждена чанком %s: %r…",
                        c["chunk_id"], quote[:60])
        quotes.append({"chunk_id": c["chunk_id"], "source": c["source"],
                       "section": c["section"], "quote": quote,
                       "verified": verdict})
        if verdict != "fail" and c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            sources.append({"chunk_id": c["chunk_id"], "source": c["source"],
                            "title": c["title"], "section": c["section"]})
    return {"parse_ok": True, "answer": answer,
            "quotes": quotes, "sources": sources,
            "memory": parse_memory(data)}
