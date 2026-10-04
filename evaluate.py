# -*- coding: utf-8 -*-
"""
Сравнение стратегий chunking.

1. Retrieval: 10 вопросов с ground truth (source + needle).
   Чанк релевантен, если chunk.source == expected и needle входит в текст чанка.
   Метрики: Recall@5 (доля вопросов с >= 1 релевантным чанком в top-5),
            MRR@5, mean rank первого релевантного.
2. Геометрия: число чанков, mean/min/max/дисперсия размера.
3. Целостность границ: доля чанков, начинающихся/кончающихся
   посередине предложения (эвристика по знакам препинания).
"""
import json
import os
import sqlite3

import numpy as np

from search import search

TESTS = os.path.join(os.path.dirname(__file__), "..", "tests", "questions.json")
BASE = os.path.join(os.path.dirname(__file__), "..", "index")


def load_questions() -> list[dict]:
    """Поддержка схемы v2 (dict с config/questions) и v1 (плоский список)."""
    data = json.load(open(TESTS, encoding="utf-8"))
    if isinstance(data, dict):
        return [{"question": q["question"], "source": q["sources"][0],
                 "needle": q["needle"]} for q in data["questions"]]
    return data


def retrieval_metrics(ks=(5,), k_max=5):
    questions = load_questions()
    report = {}
    for strat in ("fixed", "structural"):
        hits, rr, first_ranks = 0, 0.0, []
        per_q = []
        for q in questions:
            res = search(q["question"], k=k_max, strategy=strat)
            rel = [i + 1 for i, r in enumerate(res)
                   if r["source"] == q["source"] and q["needle"] in r["text"]]
            rank = rel[0] if rel else None
            if rank and rank <= ks[0]:
                hits += 1
            if rank:
                first_ranks.append(rank)
                rr += 1.0 / rank
            per_q.append({"q": q["question"][:45], "rank": rank})
        report[strat] = {
            "recall@5": hits / len(questions),
            "mrr@5": rr / len(questions),
            "mean_rank": float(np.mean(first_ranks)) if first_ranks else None,
            "found": len(first_ranks),
            "per_question": per_q,
        }
    return report


def geometry():
    out = {}
    db = sqlite3.connect(os.path.join(BASE, "meta.db"))
    for strat in ("fixed", "structural"):
        sizes = [r[0] for r in db.execute(
            f"SELECT n_tokens FROM chunks_{strat}").fetchall()]
        out[strat] = {
            "chunks": len(sizes),
            "mean": float(np.mean(sizes)),
            "std": float(np.std(sizes)),
            "min": min(sizes),
            "max": max(sizes),
        }
    db.close()
    return out


def boundary_integrity(sample=120):
    """Доля чанков с обрывом предложения на границах."""
    rng = np.random.default_rng(42)
    out = {}
    db = sqlite3.connect(os.path.join(BASE, "meta.db"))
    for strat in ("fixed", "structural"):
        rows = db.execute(f"SELECT text FROM chunks_{strat}").fetchall()
        idx = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
        broken_start = broken_end = 0
        for i in idx:
            t = rows[int(i)][0].strip()
            head_ok = t[0].isupper() or t[0] in "#«\"0123456789-|`&/" or t[:2].isupper()
            tail_ok = t[-1] in ".!?:;…»\"`)|" or t.endswith("```") or \
                t.rstrip().endswith(("КонецПроцедуры", "КонецФункции",
                                     "КонецЕсли;", "КонецЦикла;"))
            broken_start += not head_ok
            broken_end += not tail_ok
        n = len(idx)
        out[strat] = {
            "sample": n,
            "broken_start_pct": round(100 * broken_start / n, 1),
            "broken_end_pct": round(100 * broken_end / n, 1),
        }
    db.close()
    return out


if __name__ == "__main__":
    print(json.dumps({"retrieval": retrieval_metrics(),
                      "geometry": geometry(),
                      "boundaries": boundary_integrity()},
                     ensure_ascii=False, indent=2))
