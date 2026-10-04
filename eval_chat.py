# -*- coding: utf-8 -*-
"""Проверка мини-чата на длинных сценариях (10–15 сообщений).

Для каждого сценария tests/chat_scenarios/*.json прогоняет диалог через
ChatSession + ask_grounded(memory=True) и проверяет:

  per-turn (содержательные ходы, т.е. не из expect.gated_turns):
    status == ok, есть источники, есть цитаты, все цитаты strict;
  gated-ходы:
    status == low_relevance, источников/цитат нет, LLM не вызывалась;
  память задачи:
    - цель появляется не позже 2-го хода и далее НИКОГДА не пустая;
    - цель не меняется на gated-ходе (LLM в нём не участвует);
    - финальная цель соответствует expect.goal_regex;
    - финальные constraints покрывают expect.constraints_regex;
    - финальные terms покрывают expect.terms_regex;
  история: 2 записи на ход (user+assistant), источники сохранены в ходах.

Результат: tests/results_chat_<stamp>.json/.md + сводка в stdout.

Запуск (из папки src):
    python3 eval_chat.py [--scenario smoke] [--gate-sim 0.78] [--pipeline rerank]
Без DEEPSEEK_API_KEY: python3 eval_chat_stub.py ...
"""
import argparse
import glob
import json
import logging
import os
import re
import time

import chat_memory
import rag_agent
import settings

log = logging.getLogger("rag1c")

TESTS = os.path.join(os.path.dirname(__file__), "..", "tests")
SCEN_DIR = os.path.join(TESTS, "chat_scenarios")
CHAT_LLM = os.path.join(TESTS, "chat_llm")

GATE_SIM = 0.78
PIPELINE = "rerank"


def load_scenarios(only: str | None = None) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(SCEN_DIR, "*.json"))):
        s = json.load(open(path, encoding="utf-8"))
        if only is None or s["id"] == only:
            out.append(s)
    return out


def check_turn(turn_res: dict, gated: bool) -> dict:
    """Проверки одного хода. Для оффтопик-ходов (expect.gated_turns)
    корректен любой из двух слоёв отказа: low_relevance (отсек gate,
    LLM не вызывалась) или model_refusal (gate пропустил, отказала модель);
    в обоих случаях источников и цитат быть не должно."""
    r = turn_res
    ok_quotes = [q for q in r["quotes"] if q["verified"] == "strict"]
    if gated:
        layer = {"low_relevance": "gate", "model_refusal": "model"}.get(
            r["status"], "-")
        return {"refusal_ok": r["status"] in ("low_relevance",
                                              "model_refusal"),
                "no_sources": not r["sources"], "no_quotes": not r["quotes"],
                "layer": layer}
    return {"status_ok": r["status"] == "ok",
            "has_sources": bool(r["sources"]),
            "has_quotes": bool(r["quotes"]),
            "quotes_strict": bool(r["quotes"]) and
            len(ok_quotes) == len(r["quotes"])}


def run_scenario(scen: dict, gate_sim: float, pipeline: str,
                 chats_dir: str) -> dict:
    sess = chat_memory.ChatSession(session_id=f"eval_{scen['id']}")
    gated_ids = set(scen["expect"]["gated_turns"])
    turns, goal_history = [], []
    for turn in scen["turns"]:
        tid, q = turn["id"], turn["question"]
        goal_before = sess.state.goal
        sess.add_user(q)
        r = rag_agent.ask_grounded(
            q, pipeline=pipeline, gate_sim=gate_sim,
            history=sess.history_text(), task_state=sess.state_text(),
            memory=True)
        sess.add_assistant(r)
        sess.update_memory(r)
        checks = check_turn(r, tid in gated_ids)
        turns.append({"id": tid, "question": q, "status": r["status"],
                      "top_score": r["gate"]["top_score"],
                      "sources": len(r["sources"]), "quotes": len(r["quotes"]),
                      "checks": checks,
                      "memory_changed": bool(r.get("memory")),
                      "goal_after": sess.state.goal})
        goal_history.append({"turn": tid, "goal": sess.state.goal,
                             "goal_kept_on_gate": (tid not in gated_ids) or
                             sess.state.goal == goal_before})
        log.info("eval-chat %s/%s: status=%s sources=%d quotes=%d",
                 scen["id"], tid, r["status"], len(r["sources"]),
                 len(r["quotes"]))

    st = sess.state.to_dict()
    exp = scen["expect"]
    goal_set_by = next((g["turn"] for g in goal_history if g["goal"]), None)
    goals_after_set = [g["goal"] for g in goal_history
                       if goal_set_by and g["turn"] >= goal_set_by]
    memory_checks = {
        "goal_set_by_turn_2": goal_set_by in (scen["turns"][0]["id"],
                                              scen["turns"][1]["id"]),
        "goal_never_empty_after_set": all(goals_after_set),
        "goal_kept_on_gated_turns": all(g["goal_kept_on_gate"]
                                        for g in goal_history),
        "final_goal_matches": bool(re.search(exp["goal_regex"], st["goal"],
                                             re.I)),
        "constraints_cover": {rx: any(re.search(rx, c, re.I)
                                      for c in st["constraints"])
                              for rx in exp["constraints_regex"]},
        "terms_cover": {rx: any(re.search(rx, t, re.I) for t in st["terms"])
                        for rx in exp["terms_regex"]},
    }
    memory_checks["constraints_all"] = all(
        memory_checks["constraints_cover"].values())
    memory_checks["terms_all"] = all(memory_checks["terms_cover"].values())

    n_subst = sum(1 for t in turns if t["id"] not in gated_ids)
    n_gated = len(gated_ids)
    summary = {
        "turns_total": len(turns), "substantive": n_subst, "gated": n_gated,
        "ok_with_sources": sum(1 for t in turns
                               if t["checks"].get("has_sources")),
        "ok_with_quotes": sum(1 for t in turns
                              if t["checks"].get("has_quotes")),
        "all_quotes_strict": sum(1 for t in turns
                                 if t["checks"].get("quotes_strict")),
        "gated_correct": sum(1 for t in turns
                             if t["id"] in gated_ids and
                             t["checks"].get("refusal_ok") and
                             t["checks"].get("no_sources") and
                             t["checks"].get("no_quotes")),
        "history_records": len(sess.turns),
        "sources_in_history": sum(1 for t in sess.turns
                                  if t["role"] == "assistant"
                                  and t.get("sources")),
    }
    passed = (
        summary["ok_with_sources"] == n_subst
        and summary["ok_with_quotes"] == n_subst
        and summary["all_quotes_strict"] == n_subst
        and summary["gated_correct"] == n_gated
        and all(v for k, v in memory_checks.items()
                if k not in ("constraints_cover", "terms_cover")))
    save_path = sess.save(directory=chats_dir)
    return {"scenario": scen["id"], "title": scen["title"],
            "session_file": save_path, "turns": turns,
            "final_state": st, "memory_checks": memory_checks,
            "summary": summary, "passed": passed}


def render_md(results: list[dict], gate_sim: float, pipeline: str) -> str:
    lines = [f"# Проверка мини-чата (RAG + память) — "
             f"{time.strftime('%Y-%m-%d %H:%M')}",
             f"",
             f"pipeline={pipeline}, gate={gate_sim:.2f}, "
             f"LLM-см. примечание в отчёте (раздел 11.5)", ""]
    for res in results:
        s = res["summary"]
        lines.append(f"## Сценарий «{res['scenario']}»: {res['title']}")
        lines.append("")
        lines.append(f"**Вердикт: {'PASS' if res['passed'] else 'FAIL'}** | "
                     f"ходов {s['turns_total']} (содержательных "
                     f"{s['substantive']}, gated {s['gated']})")
        lines.append("")
        lines.append("| Ход | Статус | top | Источн. | Цитат | Проверки |")
        lines.append("|-----|--------|-----|---------|-------|----------|")
        for t in res["turns"]:
            ck = " ".join(k for k, v in t["checks"].items() if v)
            lines.append(f"| {t['id']} | {t['status']} | {t['top_score']:.2f} "
                         f"| {t['sources']} | {t['quotes']} | {ck} |")
        lines.append("")
        lines.append("Проверки памяти задачи:")
        lines.append("")
        for k, v in res["memory_checks"].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    lines.append(f"- {k} / `{k2}`: {'✓' if v2 else '✗'}")
            else:
                lines.append(f"- {k}: {'✓' if v else '✗'}")
        lines.append("")
        st = res["final_state"]
        lines.append(f"Финальная цель: {st['goal']}")
        lines.append(f"Ограничения: {'; '.join(st['constraints']) or '—'}")
        lines.append(f"Термины: {'; '.join(st['terms']) or '—'}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", help="id сценария (по умолчанию — все)")
    ap.add_argument("--gate-sim", type=float, default=None)
    ap.add_argument("--pipeline", default=PIPELINE)
    ap.add_argument("--chats-dir",
                    default=os.path.join(TESTS, "chat_sessions"))
    args = ap.parse_args()
    gate = args.gate_sim if args.gate_sim is not None else \
        settings.get_gate_sim()

    results = [run_scenario(s, gate, args.pipeline, args.chats_dir)
               for s in load_scenarios(args.scenario)]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json.dump({"gate_sim": gate, "pipeline": args.pipeline,
               "results": results},
              open(os.path.join(TESTS, f"results_chat_{stamp}.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(TESTS, f"results_chat_{stamp}.md"), "w",
         encoding="utf-8").write(render_md(results, gate, args.pipeline))

    print(f"\n{'сценарий':<10} {'ходов':>5} {'ok+источн.':>10} "
          f"{'ok+цитаты':>9} {'strict':>6} {'gated':>5} вердикт")
    for res in results:
        s = res["summary"]
        print(f"{res['scenario']:<10} {s['turns_total']:>5} "
              f"{s['ok_with_sources']:>10} {s['ok_with_quotes']:>9} "
              f"{s['all_quotes_strict']:>6} {s['gated_correct']:>5} "
              f"{'PASS' if res['passed'] else 'FAIL'}")
    print(f"\nрезультаты: tests/results_chat_{stamp}.md")


if __name__ == "__main__":
    main()
