# -*- coding: utf-8 -*-
"""Сравнение конвейеров RAG: plain и RAG в режимах
baseline / filter / rerank / rewrite / full.

Два уровня метрик:
  retrieval — ранг needle-чанка в выдаче конвейера (детерминированно);
  ответы    — доля найденных ключевых фактов (regex-алиасы; нижняя оценка,
              финальный вердикт — ручной разбор таблицы).

Запуск (из папки src):
    python3 eval_answers.py                              # все 5 конвейеров
    python3 eval_answers.py --pipelines baseline,full    # подмножество
    python3 eval_answers.py --only q1,q3 --pipelines baseline,rerank
    python3 eval_answers.py --min-sim 0.78 --k-pre 30    # переопределения

Требует: DEEPSEEK_API_KEY в окружении и построенный индекс ../index.

Выход (в ../tests):
    results_pipes_YYYYmmdd_HHMMSS.json — полные ответы, usage, статистика;
    results_pipes_YYYYmmdd_HHMMSS.md   — таблицы для отчёта.
"""
import argparse
import datetime as dt
import json
import os
import re

import rag_agent
from rerank import PIPELINES

TESTS = os.path.join(os.path.dirname(__file__), "..", "tests")


def load_questions(path: str) -> tuple[dict, list[dict]]:
    data = json.load(open(path, encoding="utf-8"))
    return data.get("config", {}), data["questions"]


def fact_hit(fact: dict, answer: str) -> bool:
    hits = [bool(re.search(a, answer, re.I)) for a in fact["aliases"]]
    return all(hits) if fact.get("match") == "all" else any(hits)


def needle_rank(chunks: list[dict], source: str, needle: str) -> int | None:
    """Ранг первого релевантного чанка (source + дословная подстрока)."""
    for i, c in enumerate(chunks, 1):
        if c["source"] == source and needle in c["text"]:
            return i
    return None


def run_question(q: dict, cfg: dict, pipelines: list[str],
                 k_pre: int, min_sim: float, rerank_engine: str) -> dict:
    plain = rag_agent.ask(q["question"], mode="plain")
    row = {"id": q["id"], "question": q["question"],
           "expected_sources": q["sources"], "n_facts": len(q["expect"]),
           "plain": {"answer": plain["answer"], "usage": plain["usage"],
                     "latency_s": plain["latency_s"],
                     "fact_hits": sum(fact_hit(f, plain["answer"])
                                      for f in q["expect"])},
           "pipelines": {}}
    for p in pipelines:
        print(f"    конвейер {p}…", flush=True)
        try:
            rag = rag_agent.ask(q["question"], mode="rag",
                                strategy=cfg.get("strategy", "structural"),
                                backend=cfg.get("backend", "faiss"),
                                k=cfg.get("k", 5), pipeline=p,
                                k_pre=k_pre, min_sim=min_sim,
                                rerank_engine=rerank_engine)
            got = sorted({c["source"] for c in rag["chunks"]})
            row["pipelines"][p] = {
                "answer": rag["answer"], "usage": rag["usage"],
                "latency_s": rag["latency_s"],
                "fact_hits": sum(fact_hit(f, rag["answer"])
                                 for f in q["expect"]),
                "query_used": rag["query_used"], "stats": rag["stats"],
                "retrieved_sources": got,
                "sources_ok": set(q["sources"]) <= set(got),
                "needle_rank": needle_rank(rag["chunks"], q["sources"][0],
                                           q["needle"]),
            }
        except Exception as e:
            print(f"    ОШИБКА: {e}", flush=True)
            row["pipelines"][p] = {"error": str(e), "fact_hits": 0,
                                   "sources_ok": False, "needle_rank": None,
                                   "retrieved_sources": []}
    return row


def to_markdown(rows: list[dict], pipelines: list[str]) -> str:
    n_facts = sum(r["n_facts"] for r in rows)
    # --- таблица 1: качество ответов (fact-hits)
    t1 = ["| # | Вопрос | Без RAG | "
          + " | ".join(pipelines) + " |",
          "|---|--------|---------|" + "---------|" * len(pipelines)]
    for r in rows:
        cells = [f'{r["pipelines"][p]["fact_hits"]}/{r["n_facts"]}'
                 for p in pipelines]
        t1.append(f'| {r["id"]} | {r["question"][:55]}… '
                  f'| {r["plain"]["fact_hits"]}/{r["n_facts"]} | '
                  + " | ".join(cells) + " |")
    # --- таблица 2: retrieval (ранг needle-чанка)
    t2 = ["| # | " + " | ".join(pipelines) + " |",
          "|---|" + "------|" * len(pipelines)]
    for r in rows:
        cells = []
        for p in pipelines:
            rk = r["pipelines"][p]["needle_rank"]
            cells.append(str(rk) if rk else "—")
        t2.append(f'| {r["id"]} | ' + " | ".join(cells) + " |")
    # --- сводки
    def hit_rate(key):
        return sum(r[key]["fact_hits"] for r in rows) / n_facts
    lines = ["### Качество ответов (доля найденных фактов)", "", *t1, "",
             f'**Итого:** без RAG {hit_rate("plain"):.0%}; '
             + "; ".join(f'{p} {sum(r["pipelines"][p]["fact_hits"] for r in rows) / n_facts:.0%}'
                         for p in pipelines) + ".",
             "",
             "### Retrieval: ранг релевантного чанка (needle)",
             "", *t2, ""]
    for p in pipelines:
        ranks = [r["pipelines"][p]["needle_rank"] for r in rows]
        found = [x for x in ranks if x]
        recall = len(found) / len(rows)
        mrr = sum(1.0 / x for x in found) / len(rows)
        src = sum(r["pipelines"][p]["sources_ok"] for r in rows)
        lines.append(f"- {p}: Recall {recall:.1f}, MRR {mrr:.2f}, "
                     f"источники {src}/{len(rows)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=os.path.join(TESTS, "questions.json"))
    ap.add_argument("--only", default="", help="id через запятую, напр. q1,q3")
    ap.add_argument("--pipelines", default=",".join(PIPELINES),
                    help="конвейеры через запятую, напр. baseline,rerank")
    ap.add_argument("--k-pre", type=int, default=0, help="кандидатов до фильтра")
    ap.add_argument("--min-sim", type=float, default=0.0,
                    help="порог косинусного сходства")
    ap.add_argument("--reranker", default="llm", choices=["llm", "ce"],
                    help="движок реранкинга: LLM DeepSeek или cross-encoder")
    args = ap.parse_args()

    import rerank
    k_pre = args.k_pre or rerank.K_PRE
    min_sim = args.min_sim or rerank.MIN_SIM
    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    bad = set(pipelines) - set(PIPELINES)
    if bad:
        raise SystemExit(f"неизвестные конвейеры: {bad}; есть: {PIPELINES}")

    cfg, questions = load_questions(args.questions)
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    print(f"вопросов: {len(questions)} | конвейеры: {pipelines} | "
          f"k_pre={k_pre} min_sim={min_sim}", flush=True)

    rows = []
    for i, q in enumerate(questions, 1):
        print(f'[{i}/{len(questions)}] {q["id"]}: {q["question"][:70]}…',
              flush=True)
        rows.append(run_question(q, cfg, pipelines, k_pre, min_sim,
                                 args.reranker))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    js = os.path.join(TESTS, f"results_pipes_{stamp}.json")
    md = os.path.join(TESTS, f"results_pipes_{stamp}.md")
    json.dump({"config": {**cfg, "k_pre": k_pre, "min_sim": min_sim,
                          "pipelines": pipelines, "reranker": args.reranker},
               "results": rows},
              open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(md, "w", encoding="utf-8").write(to_markdown(rows, pipelines))
    print("\n" + to_markdown(rows, pipelines))
    print(f"\nзаписано: {js}\nзаписано: {md}")


if __name__ == "__main__":
    main()
