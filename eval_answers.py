# -*- coding: utf-8 -*-
"""Прогон контрольных вопросов: режимы plain vs rag + авто-проверка фактов.

Запуск (из папки src):
    python3 eval_answers.py                 # все вопросы, конфиг из questions.json
    python3 eval_answers.py --only q1,q3    # выборочно

Требует: DEEPSEEK_API_KEY в окружении и построенный индекс ../index.

Выход (в ../tests):
    results_YYYYmmdd_HHMMSS.json — полные ответы, usage, детали проверки;
    results_YYYYmmdd_HHMMSS.md   — сводная таблица для отчёта.

Авто-проверка — необъективная нижняя оценка: regex-алиасы ловят факт только
в ожидаемой формулировке. Финальный вердикт — за ручным разбором таблицы.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys

import rag_agent

TESTS = os.path.join(os.path.dirname(__file__), "..", "tests")


def load_questions(path: str) -> tuple[dict, list[dict]]:
    data = json.load(open(path, encoding="utf-8"))
    return data.get("config", {}), data["questions"]


def fact_hit(fact: dict, answer: str) -> bool:
    hits = [bool(re.search(a, answer, re.I)) for a in fact["aliases"]]
    return all(hits) if fact.get("match") == "all" else any(hits)


def run_question(q: dict, cfg: dict) -> dict:
    plain = rag_agent.ask(q["question"], mode="plain")
    rag = rag_agent.ask(q["question"], mode="rag",
                        strategy=cfg.get("strategy", "structural"),
                        backend=cfg.get("backend", "faiss"),
                        k=cfg.get("k", 5))
    checks = [{"fact": f["fact"],
               "plain": fact_hit(f, plain["answer"]),
               "rag": fact_hit(f, rag["answer"])} for f in q["expect"]]
    got_sources = sorted({c["source"] for c in rag["chunks"]})
    return {
        "id": q["id"], "question": q["question"],
        "expected_sources": q["sources"], "retrieved_sources": got_sources,
        "sources_ok": set(q["sources"]) <= set(got_sources),
        "checks": checks,
        "plain_hits": sum(c["plain"] for c in checks),
        "rag_hits": sum(c["rag"] for c in checks),
        "n_facts": len(checks),
        "plain": {k: plain[k] for k in ("answer", "usage", "latency_s")},
        "rag": {k: rag[k] for k in ("answer", "usage", "latency_s")},
    }


def verdict(r: dict) -> str:
    p, g, n = r["plain_hits"], r["rag_hits"], r["n_facts"]
    if g > p:
        return f"RAG лучше ({g}/{n} vs {p}/{n})"
    if g == p:
        return f"одинаково ({g}/{n})"
    return f"plain лучше ({p}/{n} vs {g}/{n})"


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| # | Вопрос | Факты без RAG | Факты с RAG | Источники | Вердикт |",
        "|---|--------|---------------|-------------|-----------|---------|",
    ]
    for r in rows:
        src = "✓" if r["sources_ok"] else "✗ " + ", ".join(r["retrieved_sources"])
        lines.append(
            f'| {r["id"]} | {r["question"][:60]}… '
            f'| {r["plain_hits"]}/{r["n_facts"]} | {r["rag_hits"]}/{r["n_facts"]} '
            f'| {src} | {verdict(r)} |')
    n = sum(r["n_facts"] for r in rows)
    p = sum(r["plain_hits"] for r in rows)
    g = sum(r["rag_hits"] for r in rows)
    src_ok = sum(r["sources_ok"] for r in rows)
    lines += ["",
              f"**Итого фактов:** без RAG {p}/{n} ({p / n:.0%}), "
              f"с RAG {g}/{n} ({g / n:.0%}). "
              f"Ожидаемые источники найдены: {src_ok}/{len(rows)}."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=os.path.join(TESTS, "questions.json"))
    ap.add_argument("--only", default="", help="id через запятую, напр. q1,q3")
    args = ap.parse_args()

    cfg, questions = load_questions(args.questions)
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    print(f"вопросов: {len(questions)} | конфиг: {cfg}", flush=True)

    rows = []
    for i, q in enumerate(questions, 1):
        print(f'[{i}/{len(questions)}] {q["id"]}: {q["question"][:70]}…', flush=True)
        try:
            rows.append(run_question(q, cfg))
        except Exception as e:
            print(f"  ОШИБКА: {e}", flush=True)
            rows.append({"id": q["id"], "question": q["question"], "error": str(e),
                         "expected_sources": q["sources"], "retrieved_sources": [],
                         "sources_ok": False, "checks": [], "plain_hits": 0,
                         "rag_hits": 0, "n_facts": len(q["expect"]),
                         "plain": {}, "rag": {}})

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    js = os.path.join(TESTS, f"results_{stamp}.json")
    md = os.path.join(TESTS, f"results_{stamp}.md")
    json.dump({"config": cfg, "results": rows},
              open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(md, "w", encoding="utf-8").write(to_markdown(rows))
    print("\n" + to_markdown(rows))
    print(f"\nзаписано: {js}\nзаписано: {md}")


if __name__ == "__main__":
    sys.exit(main())
