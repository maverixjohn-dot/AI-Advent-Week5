# -*- coding: utf-8 -*-
"""Мини-чат с RAG + памятью (CLI).

Каждый вопрос: retrieval -> gate -> grounded-ответ с источниками и цитатами.
Память: история диалога + память задачи (цель, уточнения, ограничения,
термины) — подаётся в контекст каждого следующего ответа.

Команды:
  /new            новая сессия
  /state          показать память задачи
  /history        показать историю диалога
  /save           сохранить сессию в chats/<id>.json (автосохранение и так есть)
  /gate X.XX      порог релевантности на эту сессию (постоянно — через Админ в UI)
  /help           справка
  /exit           выход

Запуск (из папки src):
    python3 chat_cli.py [--session ID] [--pipeline rerank] [--gate 0.78]
"""
import argparse
import logging
import sys

import chat_memory
import rag_agent
import settings

log = logging.getLogger("rag1c")

_MARK = {"strict": "✓", "soft": "~", "fail": "✗ НЕ ПОДТВЕРЖДЕНА"}


def print_answer(r: dict) -> None:
    g = r["gate"]
    print(f"\n[статус: {r['status']} | gate: top {g['top_score']:.2f} / "
          f"порог {g['gate_sim']:.2f}"
          + ("" if r.get("model") else " | LLM не вызывалась") + "]")
    print(r["answer"])
    if r["sources"]:
        print("\nИсточники:")
        for s in r["sources"]:
            print(f'  • {s["source"]} — {s["section"] or "(без раздела)"} '
                  f'[{s["chunk_id"]}]')
    if r["quotes"]:
        print("\nЦитаты:")
        for q in r["quotes"]:
            print(f'  {_MARK[q["verified"]]} [{q["chunk_id"]}] '
                  f'«{q["quote"]}»')
    print()


def print_state(sess: chat_memory.ChatSession) -> None:
    print("\n--- Память задачи ---")
    print(sess.state_text())
    print("---------------------\n")


def print_history(sess: chat_memory.ChatSession) -> None:
    print("\n--- История ---")
    for t in sess.turns:
        who = "Вы" if t["role"] == "user" else "Ассистент"
        print(f"{who} [{t['ts'][11:]}]: {t['text'][:200]}")
        if t["role"] == "assistant" and t.get("sources"):
            print(f"   источники: "
                  + "; ".join(f'{s["source"]} [{s["chunk_id"]}]'
                              for s in t["sources"]))
    print("---------------\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="продолжить сессию из chats/<id>.json")
    ap.add_argument("--pipeline", default="rerank",
                    choices=["baseline", "filter", "rerank", "rewrite", "full"])
    ap.add_argument("--gate", type=float, default=None,
                    help="порог gate (по умолчанию — из settings.json, "
                         "задаётся на странице Админ)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    sess = (chat_memory.ChatSession.load(args.session)
            if args.session else chat_memory.ChatSession())
    gate = args.gate if args.gate is not None else settings.get_gate_sim()
    print(f"RAG-чат 1С | сессия {sess.session_id} | pipeline={args.pipeline} "
          f"| gate={gate:.2f} | /help — команды")
    if sess.turns:
        print(f"продолжение диалога: {len(sess.turns)} ходов в истории")

    while True:
        try:
            q = input("Вы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.startswith("/"):
            cmd, _, arg = q.partition(" ")
            if cmd == "/exit":
                break
            if cmd == "/new":
                sess = chat_memory.ChatSession()
                print(f"новая сессия {sess.session_id}")
            elif cmd == "/state":
                print_state(sess)
            elif cmd == "/history":
                print_history(sess)
            elif cmd == "/save":
                print("сохранено:", sess.save())
            elif cmd == "/gate":
                try:
                    v = float(arg)
                    if not (settings.GATE_MIN <= v <= settings.GATE_MAX):
                        raise ValueError
                    gate = v
                    print(f"порог gate на сессию: {gate:.2f} "
                          f"(постоянно — через страницу Админ в UI)")
                except ValueError:
                    print(f"нужно число {settings.GATE_MIN:.2f}–"
                          f"{settings.GATE_MAX:.2f}, например /gate 0.80")
            elif cmd == "/help":
                print(__doc__)
            else:
                print("неизвестная команда, /help — справка")
            continue

        sess.add_user(q)
        try:
            r = rag_agent.ask_grounded(
                q, k=args.k, pipeline=args.pipeline, gate_sim=gate,
                history=sess.history_text(), task_state=sess.state_text(),
                memory=True)
        except Exception as ex:
            log.exception("chat: ошибка ответа")
            print(f"Ошибка: {ex}")
            continue
        sess.add_assistant(r)
        changed = sess.update_memory(r)
        print_answer(r)
        if changed:
            print("[память задачи обновлена: "
                  + ", ".join(changed) + "]")
        sess.save()  # автосохранение после каждого хода
    sess.save()
    print("сессия сохранена:", sess.session_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
