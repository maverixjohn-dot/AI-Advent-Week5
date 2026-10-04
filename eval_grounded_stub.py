# -*- coding: utf-8 -*-
"""Прогон eval_grounded без доступа к LLM-API: ответы всех LLM-стадий
(реранк, генерация ответа) подставляются из файлов ../tests/grounded_llm/.

Назначение:
  - офлайн-регресс: зафиксированные ответы LLM -> детерминированная
    проверка всего остального конвейера (retrieval, gate, парсинг,
    верификация цитат, сборка источников, метрики);
  - прогон в среде без DEEPSEEK_API_KEY: LLM-стадии выполняются внешней
    моделью РОВНО по промптам проекта (те же system/user), ответы
    сохраняются в файлы и подставляются сюда.

Протокол файлов (../tests/grounded_llm/):
  {qid}.ranking.json — ответ LLM-реранкера: {"ranking": [N, ...]};
  {qid}.txt          — ответ генератора: JSON {"answer": ..., "citations": ...}
                       ровно в том виде, как вернула модель.

Запуск (из папки src):
    python3 eval_grounded_stub.py --pipeline rerank --gate-sim 0.78
"""
import json
import os
import sys

import grounded
import llm
import rerank

TESTS = os.path.join(os.path.dirname(__file__), "..", "tests")
GND_DIR = os.path.join(TESTS, "grounded_llm")


def _load(path: str) -> str:
    return open(path, encoding="utf-8").read()


def make_stub(questions: dict[str, str]):
    """Диспетчер llm.chat: по system-промпту и тексту вопроса -> файл."""
    calls = {"rerank": 0, "answer": 0}

    def stub(messages, **kwargs):
        sys_prompt = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        qid = next((i for i, q in questions.items() if q in user), None)
        if qid is None:
            raise RuntimeError("stub: вопрос не опознан в сообщении LLM")
        if sys_prompt == rerank._RERANK_SYS:
            calls["rerank"] += 1
            return {"text": _load(os.path.join(GND_DIR, f"{qid}.ranking.json")),
                    "model": "stub", "usage": {}, "latency_s": 0.0}
        if sys_prompt == grounded.SYSTEM_GROUNDED:
            calls["answer"] += 1
            return {"text": _load(os.path.join(GND_DIR, f"{qid}.txt")),
                    "model": "external-llm", "usage": {}, "latency_s": 0.0}
        if sys_prompt == rerank._REWRITE_SYS:
            # rewrite не используется в этом прогоне; честный fallback —
            # исходный вопрос (как при сбое API)
            return {"text": user, "model": "stub", "usage": {},
                    "latency_s": 0.0}
        raise RuntimeError(f"stub: неизвестная LLM-стадия: {sys_prompt[:60]}")

    return stub, calls


def main() -> None:
    import eval_grounded

    cfg, questions = eval_grounded.load_questions(
        os.path.join(TESTS, "questions.json"))
    qmap = {q["id"]: q["question"] for q in questions}
    qmap.update({q["id"]: q["question"] for q in eval_grounded.NEGATIVES})
    stub, calls = make_stub(qmap)
    llm.chat = stub  # все LLM-стадии конвейера -> файлы

    missing = []
    for qid in qmap:
        for suffix in (".txt",):
            if not os.path.exists(os.path.join(GND_DIR, qid + suffix)):
                missing.append(qid + suffix)
    if missing:
        print("внимание: отсутствуют файлы ответов:", ", ".join(missing),
              "\n(для вопросов, отсечённых gate, файлы не нужны)\n",
              flush=True)

    sys.argv = [sys.argv[0], *sys.argv[1:]]
    eval_grounded.main()
    print(f"\nLLM-вызовов через stub: реранк={calls['rerank']}, "
          f"ответы={calls['answer']}")


if __name__ == "__main__":
    main()
