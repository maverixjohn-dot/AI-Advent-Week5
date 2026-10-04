# -*- coding: utf-8 -*-
"""Сбор кандидатов retrieval для ходов чат-сценариев (без LLM-API).

Для каждого хода tests/chat_scenarios/*.json выполняет поиск k_pre=20 ->
порог min_sim (ровно как run_pipeline до стадии реранка) и сохраняет
кандидатов в ../tests/chat_harvest/{tid}.json (+ читаемый .md).

Запуск (из папки src):  python3 harvest_chat.py [--k-pre 20] [--min-sim 0.75]
"""
import argparse
import glob
import json
import os

import rerank
from search import search

TESTS = os.path.join(os.path.dirname(__file__), "..", "tests")
SCEN = os.path.join(TESTS, "chat_scenarios")
OUT = os.path.join(TESTS, "chat_harvest")


def load_scenarios() -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(SCEN, "*.json"))):
        out.append(json.load(open(path, encoding="utf-8")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-pre", type=int, default=rerank.K_PRE)
    ap.add_argument("--min-sim", type=float, default=rerank.MIN_SIM)
    ap.add_argument("--strategy", default="structural")
    ap.add_argument("--backend", default="json")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    for scen in load_scenarios():
        print(f"== сценарий {scen['id']}: {scen['title']}")
        for turn in scen["turns"]:
            tid, q = turn["id"], turn["question"]
            cand = search(q, k=args.k_pre, strategy=args.strategy,
                          backend=args.backend)
            cand = rerank.apply_threshold(cand, args.min_sim)
            payload = {"id": tid, "question": q, "k_pre": args.k_pre,
                       "min_sim": args.min_sim, "candidates": cand}
            json.dump(payload, open(os.path.join(OUT, f"{tid}.json"), "w",
                                    encoding="utf-8"), ensure_ascii=False,
                      indent=1)
            lines = [f"# {tid}: {q}", ""]
            for i, c in enumerate(cand, 1):
                lines.append(f'## [{i}] score={c["score"]:.3f} | {c["source"]} '
                             f'| {c["section"] or "без раздела"} | '
                             f'{c["chunk_id"]}')
                lines += ["", c["text"], ""]
            open(os.path.join(OUT, f"{tid}.md"), "w",
                 encoding="utf-8").write("\n".join(lines))
            top = f'top={cand[0]["score"]:.3f}' if cand else "ПУСТО"
            print(f"  {tid}: кандидатов {len(cand):>2}, {top}", flush=True)


if __name__ == "__main__":
    main()
