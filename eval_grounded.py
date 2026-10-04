# -*- coding: utf-8 -*-
"""Проверка строгого режима RAG (grounded): обязательные источники и цитаты
в каждом ответе + режим «не знаю» при слабом контексте.

Матрица:
  10 вопросов из tests/questions.json — ожидается status="ok", непустые
  sources и quotes, все цитаты verified=strict, факты ответа подкреплены
  цитатами;
  3 негативных контроля (вне корпуса) — ожидается status="low_relevance"
  и ОТСУТСТВИЕ вызова LLM (gate срабатывает до генерации).

Автопроверки на вопрос:
  has_sources     — список источников непуст;
  has_quotes      — есть хотя бы одна цитата;
  quotes_strict   — все цитаты дословно подтверждены чанками;
  facts_answer    — ключевые факты (regex-алиасы) найдены в ответе;
  facts_backed    — те же факты найдены в объединении ВЕРИФИЦИРОВАННЫХ
                    цитат (прокси «смысл ответа совпадает с цитатами»;
                    финальный вердикт — ручной разбор);
  sources_ok      — ожидаемые источники ⊆ процитированных.

Запуск (из папки src):
    python3 eval_grounded.py                          # DeepSeek из окружения
    python3 eval_grounded.py --pipeline baseline
    python3 eval_grounded.py --answers-dir ../tests/grounded_llm
        # ответы LLM подставляются из файлов {qid}.txt (прогон без API:
        # генерация выполнена внешней LLM ровно по тому же промпту)

Выход (в ../tests):
    results_grounded_YYYYmmdd_HHMMSS.json — полные ответы, цитаты, проверки;
    results_grounded_YYYYmmdd_HHMMSS.md   — таблица для отчёта.
"""
import argparse
import datetime as dt
import json
import os
import re

import rag_agent

TESTS = os.path.join(os.path.dirname(__file__), "..", "tests")

# Вопросы гарантированно вне корпуса: gate обязан их отсечь без вызова LLM.
NEGATIVES = [
    {"id": "neg1", "question": "Какой сегодня курс евро к казахстанскому тенге "
                               "и прогноз на неделю?"},
    {"id": "neg2", "question": "Как приготовить пасту карбонара по классическому "
                               "римскому рецепту?"},
    {"id": "neg3", "question": "Как настроить BGP-сессию между двумя "
                               "маршрутизаторами Cisco?"},
]


def load_questions(path: str) -> tuple[dict, list[dict]]:
    data = json.load(open(path, encoding="utf-8"))
    return data.get("config", {}), data["questions"]


def fact_hit(fact: dict, text: str) -> bool:
    hits = [bool(re.search(a, text, re.I)) for a in fact["aliases"]]
    return all(hits) if fact.get("match") == "all" else any(hits)


def make_injector(answers_dir: str, qid: str, counter: dict):
    """llm_fn, подставляющий заранее сгенерированный ответ из {qid}.txt."""
    path = os.path.join(answers_dir, f"{qid}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"нет файла ответа {path}")
    text = open(path, encoding="utf-8").read()

    def _fn(messages, **kwargs):
        counter["calls"] += 1
        return {"text": text, "model": "external-llm (тот же промпт)",
                "usage": {"prompt": 0, "completion": 0, "total": 0},
                "latency_s": 0.0}
    return _fn


def run_one(q: dict, cfg: dict, pipeline: str, gate_sim: float,
            answers_dir: str | None) -> dict:
    counter = {"calls": 0}
    llm_fn = (make_injector(answers_dir, q["id"], counter)
              if answers_dir else None)
    res = rag_agent.ask_grounded(
        q["question"], strategy=cfg.get("strategy", "structural"),
        backend=cfg.get("backend", "faiss"), k=cfg.get("k", 5),
        pipeline=pipeline, gate_sim=gate_sim, llm_fn=llm_fn)
    if not answers_dir:
        counter["calls"] = 1 if res["status"] not in ("low_relevance",) else 0

    verified_quotes = [x for x in res["quotes"] if x["verified"] == "strict"]
    quotes_text = "\n".join(x["quote"] for x in res["quotes"]
                            if x["verified"] != "fail")
    expect = q.get("expect", [])
    fa = sum(fact_hit(f, res["answer"]) for f in expect)
    fb = sum(fact_hit(f, quotes_text) for f in expect)
    cited = {s["source"] for s in res["sources"]}
    checks = {
        "has_sources": len(res["sources"]) > 0,
        "has_quotes": len(res["quotes"]) > 0,
        "quotes_strict": (len(res["quotes"]) > 0
                          and len(verified_quotes) == len(res["quotes"])),
        "n_quotes": len(res["quotes"]),
        "n_quotes_fail": sum(1 for x in res["quotes"]
                             if x["verified"] == "fail"),
        "facts_answer": f"{fa}/{len(expect)}" if expect else "—",
        "facts_backed": f"{fb}/{len(expect)}" if expect else "—",
        "facts_all_backed": bool(expect) and fa == fb == len(expect),
        "sources_ok": (set(q.get("sources", [])) <= cited
                       if q.get("sources") else None),
        "llm_calls": counter["calls"],
    }
    return {"id": q["id"], "question": q["question"],
            "expected_sources": q.get("sources", []),
            "status": res["status"], "gate": res["gate"],
            "answer": res["answer"], "sources": res["sources"],
            "quotes": res["quotes"], "checks": checks,
            "query_used": res["query_used"], "stats": res["stats"]}


def to_markdown(rows: list[dict], neg_rows: list[dict]) -> str:
    t = ["| # | status | gate top | источники | цитаты | строгие | "
         "факты в ответе | факты в цитатах | источники ок |",
         "|---|--------|----------|-----------|--------|---------|"
         "----------------|-----------------|--------------|"]
    for r in rows:
        c = r["checks"]
        t.append(
            f'| {r["id"]} | {r["status"]} | {r["gate"]["top_score"]:.2f} '
            f'| {"да" if c["has_sources"] else "НЕТ"} '
            f'| {c["n_quotes"]} '
            f'| {"да" if c["quotes_strict"] else "НЕТ"} '
            f'| {c["facts_answer"]} | {c["facts_backed"]} '
            f'| {"да" if c["sources_ok"] else "НЕТ"} |')
    lines = ["### Grounded-режим: обязательные источники и цитаты (10 вопросов)",
             "", *t, ""]
    ok = sum(r["status"] == "ok" for r in rows)
    src = sum(r["checks"]["has_sources"] for r in rows)
    quo = sum(r["checks"]["quotes_strict"] for r in rows)
    back = sum(r["checks"]["facts_all_backed"] for r in rows)
    lines.append(f"**Итого:** status ok {ok}/{len(rows)}; источники есть "
                 f"{src}/{len(rows)}; все цитаты дословные {quo}/{len(rows)}; "
                 f"факты полностью подкреплены цитатами {back}/{len(rows)}.")
    lines += ["", "### Негативный контроль (вопросы вне корпуса): "
              "режим «не знаю»", "",
              "| # | status | gate top | LLM вызван | вердикт |",
              "|---|--------|----------|------------|---------|"]
    for r in neg_rows:
        is_refusal = r["status"] in ("low_relevance", "model_refusal")
        good = is_refusal and not r["checks"]["has_sources"]
        lines.append(f'| {r["id"]} | {r["status"]} | '
                     f'{r["gate"]["top_score"]:.2f} '
                     f'| {r["checks"]["llm_calls"]} '
                     f'| {"ок" if good else "СБОЙ"} |')
    refused = sum(r["status"] in ("low_relevance", "model_refusal")
                  for r in neg_rows)
    lines += ["", f"**Итого:** корректных отказов «не знаю» "
                  f"{refused}/{len(neg_rows)} (low_relevance — отсечено gate "
                  "без вызова LLM; model_refusal — отказ на уровне модели)."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=os.path.join(TESTS, "questions.json"))
    ap.add_argument("--only", default="", help="id через запятую, напр. q1,q3")
    ap.add_argument("--pipeline", default="baseline",
                    help="конвейер retrieval (baseline/filter/rerank/rewrite/full)")
    ap.add_argument("--gate-sim", type=float, default=0.0,
                    help="порог gate (0 — берётся min_sim конвейера)")
    ap.add_argument("--no-negatives", action="store_true",
                    help="пропустить негативный контроль")
    ap.add_argument("--answers-dir", default="",
                    help="папка с готовыми ответами LLM {qid}.txt (без API)")
    args = ap.parse_args()

    import rerank
    gate_sim = args.gate_sim or rerank.MIN_SIM
    cfg, questions = load_questions(args.questions)
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    answers_dir = args.answers_dir or None
    print(f"вопросов: {len(questions)} | pipeline={args.pipeline} | "
          f"gate_sim={gate_sim} | answers_dir={answers_dir}", flush=True)

    rows = []
    for i, q in enumerate(questions, 1):
        print(f'[{i}/{len(questions)}] {q["id"]}: {q["question"][:70]}…',
              flush=True)
        rows.append(run_one(q, cfg, args.pipeline, gate_sim, answers_dir))
        r = rows[-1]
        print(f'    -> {r["status"]}, источников {len(r["sources"])}, '
              f'цитат {r["checks"]["n_quotes"]} '
              f'(fail {r["checks"]["n_quotes_fail"]})', flush=True)

    neg_rows = []
    if not args.no_negatives:
        for q in NEGATIVES:
            print(f'[neg] {q["id"]}: {q["question"][:70]}…', flush=True)
            neg_rows.append(run_one(q, cfg, args.pipeline, gate_sim,
                                    answers_dir))
            print(f'    -> {neg_rows[-1]["status"]}, LLM вызовов: '
                  f'{neg_rows[-1]["checks"]["llm_calls"]}', flush=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    js = os.path.join(TESTS, f"results_grounded_{stamp}.json")
    md = os.path.join(TESTS, f"results_grounded_{stamp}.md")
    json.dump({"config": {**cfg, "pipeline": args.pipeline,
                          "gate_sim": gate_sim, "answers_dir": answers_dir},
               "results": rows, "negatives": neg_rows},
              open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    report = to_markdown(rows, neg_rows)
    open(md, "w", encoding="utf-8").write(report)
    print("\n" + report)
    print(f"\nзаписано: {js}\nзаписано: {md}")


if __name__ == "__main__":
    main()
