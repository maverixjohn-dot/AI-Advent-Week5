# -*- coding: utf-8 -*-
"""Сбор кандидатов конвейера rerank для внешнего ранжирования (без LLM-API).

Для каждого вопроса tests/questions.json + негативных контролей
eval_grounded.NEGATIVES выполняет retrieval k_pre=20 -> порог min_sim
(ровно как run_pipeline(mode="rerank")) и сохраняет кандидатов в
../tests/harvest/{qid}.json (метаданные + полные тексты, в том порядке,
в каком их увидел бы LLM-реранкер) и читаемый {qid}.md.

Запуск (из папки src):  python3 harvest_candidates.py [--k-pre 20] [--min-sim 0.75]
"""
import argparse
import json
import os

import eval_grounded
import rerank
from search import search

TESTS = os.path.join(os.path.dirname(__file__), "..", "tests")
OUT = os.path.join(TESTS, "harvest")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-pre", type=int, default=rerank.K_PRE)
    ap.add_argument("--min-sim", type=float, default=rerank.MIN_SIM)
    ap.add_argument("--strategy", default="structural")
    ap.add_argument("--backend", default="json")
    args = ap.parse_args()

    _, questions = eval_grounded.load_questions(
        os.path.join(TESTS, "questions.json"))
    questions = questions + eval_grounded.NEGATIVES
    os.makedirs(OUT, exist_ok=True)

    for q in questions:
        cand = search(q["question"], k=args.k_pre, strategy=args.strategy,
                      backend=args.backend)
        cand = rerank.apply_threshold(cand, args.min_sim)
        payload = {"id": q["id"], "question": q["question"],
                   "k_pre": args.k_pre, "min_sim": args.min_sim,
                   "candidates": cand}
        json.dump(payload, open(os.path.join(OUT, f'{q["id"]}.json'), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=1)
        lines = [f'# {q["id"]}: {q["question"]}', ""]
        for i, c in enumerate(cand, 1):
            lines.append(f'## [{i}] score={c["score"]:.3f} | {c["source"]} | '
                         f'{c["section"] or "без раздела"} | {c["chunk_id"]}')
            lines.append("")
            lines.append(c["text"])
            lines.append("")
        open(os.path.join(OUT, f'{q["id"]}.md'), "w",
             encoding="utf-8").write("\n".join(lines))
        print(f'{q["id"]:>5}: кандидатов после порога {len(cand)}, '
              f'top={cand[0]["score"]:.3f}' if cand else f'{q["id"]}: пусто',
              flush=True)


if __name__ == "__main__":
    main()
