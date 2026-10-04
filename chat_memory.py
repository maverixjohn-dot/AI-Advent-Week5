# -*- coding: utf-8 -*-
"""Память мини-чата: история диалога + «память задачи» (task state).

Два слоя памяти с разной надёжностью:

  1. История диалога (turns) — детерминированная: каждое сообщение
     пользователя и каждый ответ ассистента (с источниками, цитатами и
     статусом) записываются как есть. Не зависит от LLM, теряться не может.

  2. Память задачи (task_state) — LLM извлекает её в поле "memory"
     grounded-ответа (см. grounded.SYSTEM_GROUNDED_MEM), но применяется она
     через merge_memory() с защитной семантикой:
       - goal: заменяется только НЕПУСТЫМ новым значением — модель не может
         молча стереть цель диалога;
       - clarifications / constraints / terms: объединение по множеству
         (дедупликация без учёта регистра), старые записи не удаляются
         автоматически — модель может только ДОБАВЛЯТЬ;
       - лимит MAX_ITEMS на список: при переполнении отбрасываются новые,
         а не накопленные записи.
     Итог: сбой или галлюцинация памяти у LLM в худшем случае даёт
     «нет новых записей», а не потерю накопленного состояния.

Сессия сохраняется в chats/<session_id>.json (атомарно, через .tmp).
"""
import json
import logging
import os
import re
import time
import uuid

log = logging.getLogger("rag1c")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHATS_DIR = os.path.join(ROOT, "chats")

MAX_ITEMS = 20          # лимит на каждый список памяти задачи
HISTORY_TURNS = 6       # сколько последних ходов подаётся в контекст LLM
MAX_FIELD = 300         # лимит длины одной записи памяти


def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


class TaskState:
    """Цель диалога, уточнения пользователя, зафиксированные ограничения
    и термины. Все поля сериализуемы в JSON."""

    def __init__(self, goal: str = "", clarifications=None, constraints=None,
                 terms=None):
        self.goal = str(goal or "").strip()
        self.clarifications = list(clarifications or [])
        self.constraints = list(constraints or [])
        self.terms = list(terms or [])

    def merge(self, mem: dict) -> dict:
        """Применить memory из ответа LLM. Возвращает dict изменений
        (для лога/UI). Семантика: цель не затирается пустым значением,
        списки только пополняются (union с дедупликацией)."""
        if not isinstance(mem, dict):
            return {}
        changed = {}
        goal = _clip(mem.get("goal") or "", MAX_FIELD)
        if goal and _norm_key(goal) != _norm_key(self.goal):
            changed["goal"] = {"old": self.goal, "new": goal}
            self.goal = goal
        for field in ("clarifications", "constraints", "terms"):
            raw = mem.get(field)
            if not isinstance(raw, list):
                continue
            have = {_norm_key(x) for x in getattr(self, field)}
            added = []
            for item in raw:
                item = _clip(item, MAX_FIELD)
                if item and _norm_key(item) not in have:
                    have.add(_norm_key(item))
                    added.append(item)
            if added:
                lst = getattr(self, field) + added
                overflow = max(0, len(lst) - MAX_ITEMS)
                if overflow:
                    log.warning("memory: список %s переполнен, отброшено %d "
                                "новых записей", field, overflow)
                    lst = lst[:MAX_ITEMS]
                setattr(self, field, lst)
                changed[field] = added
        return changed

    def to_dict(self) -> dict:
        return {"goal": self.goal, "clarifications": self.clarifications,
                "constraints": self.constraints, "terms": self.terms}

    @classmethod
    def from_dict(cls, d: dict) -> "TaskState":
        d = d if isinstance(d, dict) else {}
        return cls(goal=d.get("goal", ""),
                   clarifications=d.get("clarifications"),
                   constraints=d.get("constraints"), terms=d.get("terms"))

    def text(self) -> str:
        """Компактный блок для промпта; пустое состояние — явная пометка."""
        lines = [f"Цель диалога: {self.goal or '(ещё не зафиксирована)'}"]
        for label, field in (("Уточнения пользователя", "clarifications"),
                             ("Зафиксированные ограничения", "constraints"),
                             ("Термины и имена объектов", "terms")):
            items = getattr(self, field)
            lines.append(f"{label}: "
                         + ("; ".join(items) if items else "(нет)"))
        return "\n".join(lines)


class ChatSession:
    """Диалог: история ходов + память задачи + сериализация в chats/."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S_") + \
            uuid.uuid4().hex[:6]
        self.created = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.turns: list[dict] = []
        self.state = TaskState()

    # ------------------------------------------------------------ запись
    def add_user(self, text: str) -> dict:
        turn = {"role": "user", "text": text,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.turns.append(turn)
        return turn

    def add_assistant(self, result: dict) -> dict:
        """Записать ответ ask_grounded: текст, статус, источники, цитаты.
        Хранятся только проверенные цитаты (strict/soft) — fail-цитаты
        в памяти не закрепляются."""
        turn = {
            "role": "assistant", "text": result.get("answer", ""),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": result.get("status"),
            "sources": [{"chunk_id": s["chunk_id"], "source": s["source"],
                         "section": s["section"]}
                        for s in result.get("sources", [])],
            "quotes": [{"chunk_id": q["chunk_id"], "quote": q["quote"],
                        "verified": q["verified"]}
                       for q in result.get("quotes", [])
                       if q.get("verified") in ("strict", "soft")],
        }
        self.turns.append(turn)
        return turn

    def update_memory(self, result: dict) -> dict:
        """Применить memory из ответа (если LLM его вернула)."""
        mem = result.get("memory")
        if not mem:
            return {}
        changed = self.state.merge(mem)
        if changed:
            log.info("memory: обновление состояния: %s",
                     {k: (v if isinstance(v, list) else "замена")
                      for k, v in changed.items()})
        return changed

    # ------------------------------------------------------------ чтение
    def history_text(self, max_turns: int = HISTORY_TURNS) -> str:
        """Последние max_turns ходов для промпта (компактно, с усечением)."""
        tail = self.turns[-max_turns:]
        lines = []
        for t in tail:
            if t["role"] == "user":
                lines.append(f"Пользователь: {_clip(t['text'], 500)}")
            else:
                line = f"Ассистент: {_clip(t['text'], 400)}"
                if t.get("status") == "low_relevance":
                    line += " [контекст не найден, LLM не вызывалась]"
                elif t.get("sources"):
                    line += f" [источников: {len(t['sources'])}]"
                lines.append(line)
        return "\n".join(lines) if lines else "(диалог только начался)"

    def state_text(self) -> str:
        return self.state.text()

    # ------------------------------------------------------------ файлы
    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "created": self.created,
                "task_state": self.state.to_dict(), "turns": self.turns}

    @classmethod
    def from_dict(cls, d: dict) -> "ChatSession":
        s = cls(session_id=d["session_id"])
        s.created = d.get("created", s.created)
        s.state = TaskState.from_dict(d.get("task_state"))
        s.turns = list(d.get("turns") or [])
        return s

    def save(self, directory: str = CHATS_DIR) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.session_id}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        log.info("chat: сессия %s сохранена (%d ходов)", self.session_id,
                 len(self.turns))
        return path

    @classmethod
    def load(cls, path_or_id: str, directory: str = CHATS_DIR
             ) -> "ChatSession":
        path = path_or_id
        if not path.endswith(".json"):
            path = os.path.join(directory, path_or_id + ".json")
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
