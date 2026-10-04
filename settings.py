# -*- coding: utf-8 -*-
"""Персистентные настройки приложения (settings.json в корне проекта).

Сейчас хранится одна настройка:
  gate_sim — порог релевантности детерминированного gate (слой 1 режима
  «не знаю»): если топ-скор retrieval ниже порога, LLM не вызывается,
  ассистент отвечает «не знаю» и просит уточнить вопрос.

Значение по умолчанию 0.78 выбрано по калибровке на корпусе (см. отчёт,
раздел 10.2): нерелевантные вопросы дают top-score 0.776/0.777/0.831,
релевантные — в основном 0.85–0.90; 0.78 отсекает заведомо слабый контекст,
не трогая реальные вопросы.
"""
import json
import logging
import os

log = logging.getLogger("rag1c")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(ROOT, "settings.json")

DEFAULTS = {"gate_sim": 0.78}

# допустимые границы порога (косинусная близость e5-large на этом корпусе)
GATE_MIN, GATE_MAX = 0.50, 0.95


def load() -> dict:
    """Настройки из settings.json; при отсутствии/битом файле — DEFAULTS."""
    if not os.path.exists(SETTINGS_PATH):
        return dict(DEFAULTS)
    try:
        data = json.load(open(SETTINGS_PATH, encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("settings.json: ожидается JSON-объект")
        out = dict(DEFAULTS)
        if isinstance(data.get("gate_sim"), (int, float)):
            out["gate_sim"] = float(data["gate_sim"])
        return out
    except (OSError, ValueError) as e:
        log.warning("settings: не удалось прочитать %s: %s — используются "
                    "значения по умолчанию", SETTINGS_PATH, e)
        return dict(DEFAULTS)


def get_gate_sim() -> float:
    return load()["gate_sim"]


def set_gate_sim(value: float) -> float:
    """Валидация и сохранение порога. Возвращает сохранённое значение."""
    v = float(value)
    if not (GATE_MIN <= v <= GATE_MAX):
        raise ValueError(
            f"порог релевантности должен быть в диапазоне "
            f"{GATE_MIN:.2f}–{GATE_MAX:.2f}, получено {value!r}")
    data = load()
    data["gate_sim"] = round(v, 4)
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SETTINGS_PATH)  # атомарная замена, без полузаписанных файлов
    log.info("settings: gate_sim = %.4f сохранён в %s", data["gate_sim"],
             SETTINGS_PATH)
    return data["gate_sim"]
