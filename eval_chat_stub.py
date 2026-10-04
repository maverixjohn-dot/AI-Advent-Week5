# -*- coding: utf-8 -*-
"""Прогон eval_chat без доступа к LLM-API: ответы LLM-стадий (реранк,
генерация с памятью) подставляются из файлов ../tests/chat_llm/.

Протокол файлов (../tests/chat_llm/):
  {tid}.ranking.json — ответ LLM-реранкера: {"ranking": [N, ...]};
  {tid}.txt          — ответ генератора: JSON {"answer","citations","memory"}
                       ровно в том виде, как вернула внешняя модель.

Запуск (из папки src):
    python3 eval_chat_stub.py [--scenario smoke] [--gate-sim 0.78]
"""
import json
import os
import sys

import grounded
import llm
import rerank

TESTS = os.path.join(os.path.dirname(__file__), "..", "tests")
CHAT_LLM = os.path.join(TESTS, "chat_llm")


def _load(path: str) -> str:
    return open(path, encoding="utf-8").read()


def make_stub(questions: dict[str, str]):
    """Диспетчер llm.chat: по system-промпту и тексту вопроса -> файл."""
    calls = {"rerank": 0, "answer": 0}

    def stub(messages, **kwargs):
        sys_prompt = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        # Текущий вопрос — ПОСЛЕДНЕЕ вхождение в user-сообщение: в блоке
        # истории диалога присутствуют вопросы прошлых ходов, а актуальный
        # вопрос стоит в конце ("Вопрос: ..."). Поэтому не первое вхождение,
        # а максимальный rfind.
        found = [(user.rfind(q), i) for i, q in questions.items()
                 if q in user]
        tid = max(found)[1] if found else None
        if tid is None:
            raise RuntimeError("stub: вопрос не опознан в сообщении LLM")
        if sys_prompt == rerank._RERANK_SYS:
            calls["rerank"] += 1
            return {"text": _load(os.path.join(CHAT_LLM, f"{tid}.ranking.json")),
                    "model": "stub", "usage": {}, "latency_s": 0.0}
        if sys_prompt == grounded.SYSTEM_GROUNDED_MEM:
            calls["answer"] += 1
            return {"text": _load(os.path.join(CHAT_LLM, f"{tid}.txt")),
                    "model": "external-llm", "usage": {}, "latency_s": 0.0}
        raise RuntimeError(f"stub: неизвестная LLM-стадия: {sys_prompt[:60]}")

    return stub, calls


def main() -> None:
    import eval_chat

    scens = eval_chat.load_scenarios()
    qmap = {t["id"]: t["question"] for s in scens for t in s["turns"]}
    stub, calls = make_stub(qmap)
    llm.chat = stub  # все LLM-стадии конвейера -> файлы

    missing = [f"{tid}{suf}" for tid in qmap
               for suf in (".ranking.json", ".txt")
               if not os.path.exists(os.path.join(CHAT_LLM, tid + suf))]
    if missing:
        print("внимание: отсутствуют файлы:", ", ".join(missing),
              "\n(для ходов, отсечённых gate, файлы не нужны)\n", flush=True)

    eval_chat.main()
    print(f"\nLLM-вызовов через stub: реранк={calls['rerank']}, "
          f"ответы={calls['answer']}")


if __name__ == "__main__":
    main()
