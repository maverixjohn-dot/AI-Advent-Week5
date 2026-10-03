# -*- coding: utf-8 -*-
"""
Веб-UI для локального индекса документов (Flet, web-режим).

Страницы: Поиск / Администрирование / Логи.
Запуск:  python3 ui_app.py [--host 127.0.0.1] [--port 8550]
"""
import argparse
import asyncio
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time

import flet as ft

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
LOG_DIR = os.path.join(ROOT, "logs")
APP_LOG = os.path.join(LOG_DIR, "app.log")
BUILD_LOG = os.path.join(LOG_DIR, "build.log")
INDEX_DIR = os.path.join(ROOT, "index")
META_DB = os.path.join(INDEX_DIR, "meta.db")

os.makedirs(LOG_DIR, exist_ok=True)

log = logging.getLogger("rag1c")
log.setLevel(logging.INFO)
log.propagate = False
_fh = logging.FileHandler(APP_LOG, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_fh)

sys.path.insert(0, SRC)
import ingest  # noqa: E402
import search as rag_search  # noqa: E402

STRATEGIES = ("structural", "fixed")
K_OPTIONS = (1, 3, 5, 10)

# ------------------------------------------------------------------ helpers
def _file_mb(name: str) -> float:
    p = os.path.join(INDEX_DIR, name)
    return os.path.getsize(p) / 1e6 if os.path.exists(p) else 0.0

def index_stats() -> dict:
    """Статистика индексов из meta.db; модель при этом не загружается."""
    stats = {}
    if not os.path.exists(META_DB):
        return stats
    db = sqlite3.connect(META_DB)
    for s in STRATEGIES:
        try:
            rows = db.execute(
                f"SELECT COUNT(*), AVG(n_tokens), MIN(n_tokens), MAX(n_tokens) "
                f"FROM chunks_{s}").fetchone()
            if rows[0]:
                stats[s] = {"chunks": rows[0], "mean": round(rows[1], 1),
                            "min": rows[2], "max": rows[3],
                            "faiss_mb": round(_file_mb(f"index_{s}.faiss"), 2),
                            "json_mb": round(_file_mb(f"index_{s}.json"), 2)}
        except sqlite3.OperationalError:
            pass
    db.close()
    return stats

def corpus_rows() -> list:
    docs = ingest.load_corpus()
    return docs, sum(len(d["text"]) for d in docs)

def do_search(query: str, strategy: str, k: int, backend: str = "faiss"):
    t0 = time.time()
    res = rag_search.search(query, k=k, strategy=strategy, backend=backend)
    dt = time.time() - t0
    top = res[0]["score"] if res else 0.0
    log.info('поиск: "%s" | %s | %s | k=%d | hits=%d | top=%.3f | %.2f c',
             query, strategy, backend, k, len(res), top, dt)
    return res, dt

# ------------------------------------------------------------------ pages
class App:
    def __init__(self, page: ft.Page):
        self.page = page
        self.rebuild_proc = None
        self.logs_task = None
        self.auto_refresh = False
        self.content = ft.Column(expand=True, scroll=ft.ScrollMode.AUTO,
                                 spacing=16)
        self.rail = ft.NavigationRail(
            selected_index=0, label_type=ft.NavigationRailLabelType.ALL,
            destinations=[
                ft.NavigationRailDestination(icon=ft.Icons.SEARCH, label="Поиск"),
                ft.NavigationRailDestination(icon=ft.Icons.SETTINGS_APPLICATIONS,
                                             label="Админ"),
                ft.NavigationRailDestination(icon=ft.Icons.LIST_ALT, label="Логи"),
            ],
            on_change=self.on_nav)
        page.add(ft.Row([self.rail, ft.VerticalDivider(width=1),
                         self.content], expand=True))

    # ------------------------------------------------------- навигация
    async def on_nav(self, e):
        idx = int(e.control.selected_index)
        await self.show(idx)

    async def show(self, idx: int):
        if self.logs_task:
            self.logs_task.cancel()
            self.logs_task = None
        self.content.controls.clear()
        if idx == 0:
            self.build_search()
            self.page.route = "/search"
        elif idx == 1:
            self.build_admin()
            self.page.route = "/admin"
            log.info("открыта страница администрирования")
        else:
            self.build_logs()
            self.page.route = "/logs"
            log.info("открыта страница логов")
        self.page.update()

    # ------------------------------------------------------- поиск
    def build_search(self):
        self.q = ft.TextField(label="Запрос", hint_text="например: что такое OneScript?",
                              expand=True, on_submit=self.on_search)
        self.strategy = ft.Dropdown(
            label="Стратегия", value="structural", width=170,
            options=[ft.dropdown.Option(s) for s in STRATEGIES])
        self.backend = ft.Dropdown(
            label="Бэкенд", value="faiss", width=120,
            options=[ft.dropdown.Option("faiss"), ft.dropdown.Option("json")])
        self.k = ft.Dropdown(
            label="k", value="5", width=90,
            options=[ft.dropdown.Option(str(x)) for x in K_OPTIONS])
        self.status = ft.Text("Модель загружается в фоне…", italic=True,
                              color=ft.Colors.GREY)
        self.results = ft.Column(spacing=10)
        btn = ft.FilledButton("Найти", icon=ft.Icons.SEARCH, on_click=self.on_search)
        self.content.controls += [
            ft.Text("Поиск по индексу", size=22, weight=ft.FontWeight.BOLD),
            ft.Row([self.q, self.strategy, self.backend, self.k, btn],
                   alignment=ft.MainAxisAlignment.START),
            self.status, self.results]

    async def on_search(self, e):
        query = (self.q.value or "").strip()
        if not query:
            return
        self.status.value = "Ищу…"
        self.status.color = ft.Colors.GREY
        self.results.controls.clear()
        self.page.update()
        try:
            res, dt = await asyncio.to_thread(
                do_search, query, self.strategy.value, int(self.k.value),
                self.backend.value)
        except Exception as ex:
            log.exception("ошибка поиска")
            self.status.value = f"Ошибка: {ex}"
            self.status.color = ft.Colors.RED
            self.page.update()
            return
        self.status.value = f"Найдено: {len(res)} за {dt:.2f} c"
        self.status.color = ft.Colors.GREEN
        for r in res:
            score = r["score"]
            color = (ft.Colors.GREEN if score >= 0.85 else
                     ft.Colors.ORANGE if score >= 0.78 else ft.Colors.RED)
            self.results.controls.append(ft.Card(content=ft.Container(
                padding=12, content=ft.Column([
                    ft.Row([
                        ft.Chip(ft.Text(f"{score:.3f}"), bgcolor=color),
                        ft.Text(r["source"], weight=ft.FontWeight.BOLD),
                        ft.Text(r["type"], color=ft.Colors.GREY),
                    ]),
                    ft.Text(r["section"] or "(без раздела)", italic=True,
                            color=ft.Colors.GREY, size=12),
                    ft.ExpansionTile(
                        title=ft.Text(r["text"][:160].replace("\n", " ") + "…",
                                      size=13),
                        controls=[ft.Container(
                            padding=ft.Padding(16, 0, 16, 12),
                            content=ft.Text(r["text"], selectable=True,
                                            size=12))],
                        expanded=False),
                ], spacing=6))))
        self.page.update()

    # ------------------------------------------------------- админ
    def build_admin(self):
        docs, total_chars = corpus_rows()
        istats = index_stats()

        docs_table = ft.DataTable(
            columns=[ft.DataColumn(ft.Text("Файл")),
                     ft.DataColumn(ft.Text("Тип")),
                     ft.DataColumn(ft.Text("Знаков"), numeric=True)],
            rows=[ft.DataRow(cells=[
                ft.DataCell(ft.Text(d["source"])),
                ft.DataCell(ft.Text(d["type"])),
                ft.DataCell(ft.Text(f'{len(d["text"]):,}'.replace(",", " "))),
            ]) for d in docs],
            heading_row_height=36, data_row_min_height=32,
            column_spacing=24)

        idx_rows = []
        for s, st in istats.items():
            idx_rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(s)),
                ft.DataCell(ft.Text(str(st["chunks"]))),
                ft.DataCell(ft.Text(f'{st["mean"]:.0f}')),
                ft.DataCell(ft.Text(f'{st["min"]}–{st["max"]}')),
                ft.DataCell(ft.Text(f'{st["faiss_mb"]:.1f} МБ')),
                ft.DataCell(ft.Text(f'{st["json_mb"]:.1f} МБ'
                                    if st["json_mb"] else "—")),
            ]))
        idx_table = ft.DataTable(
            columns=[ft.DataColumn(ft.Text("Стратегия")),
                     ft.DataColumn(ft.Text("Чанков"), numeric=True),
                     ft.DataColumn(ft.Text("Ср. токенов"), numeric=True),
                     ft.DataColumn(ft.Text("min–max")),
                     ft.DataColumn(ft.Text("FAISS")),
                     ft.DataColumn(ft.Text("JSON"))],
            rows=idx_rows, heading_row_height=36, column_spacing=24)

        self.admin_status = ft.Text("", color=ft.Colors.GREY, italic=True)
        busy = self.rebuild_proc is not None and self.rebuild_proc.poll() is None
        self.admin_status.value = ("Идёт пересборка индекса…" if busy else "")

        def mk_btn(label, strategy):
            return ft.OutlinedButton(
                label, disabled=busy,
                on_click=lambda e, s=strategy: self.page.run_task(
                    self.on_rebuild, e, s))

        self.content.controls += [
            ft.Text("Администрирование", size=22, weight=ft.FontWeight.BOLD),
            ft.Card(content=ft.Container(padding=16, content=ft.Column([
                ft.Text(f"Корпус: {len(docs)} документов, "
                        f"{total_chars:,} знаков".replace(",", " "),
                        weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Column([docs_table], scroll=ft.ScrollMode.AUTO),
                    height=360)]))),
            ft.Card(content=ft.Container(padding=16, content=ft.Column([
                ft.Text("Индексы (модель: intfloat/multilingual-e5-large, "
                        "dim=1024)", weight=ft.FontWeight.BOLD),
                idx_table,
                ft.Row([mk_btn("Пересобрать fixed", "fixed"),
                        mk_btn("Пересобрать structural", "structural"),
                        mk_btn("Пересобрать оба", "all")]),
                self.admin_status]))),
        ]

    async def on_rebuild(self, e, strategy: str):
        if self.rebuild_proc and self.rebuild_proc.poll() is None:
            return
        log.info("пересборка индекса запущена: %s", strategy)
        bf = open(BUILD_LOG, "a", encoding="utf-8")
        bf.write(f"\n===== пересборка {strategy} {time.strftime('%F %T')} =====\n")
        bf.flush()
        self.rebuild_proc = subprocess.Popen(
            [sys.executable, "build_index.py", strategy],
            cwd=SRC, stdout=bf, stderr=subprocess.STDOUT)
        self.admin_status.value = (f"Идёт пересборка «{strategy}»… "
                                   f"(CPU: ~20 мин на стратегию)")
        self.admin_status.color = ft.Colors.ORANGE
        self.page.update()
        while self.rebuild_proc.poll() is None:
            await asyncio.sleep(5)
        rc = self.rebuild_proc.returncode
        msg = (f"пересборка «{strategy}» завершена, rc={rc}")
        log.info(msg)
        self.admin_status.value = msg + (" — ошибка, см. build.log" if rc else "")
        self.admin_status.color = ft.Colors.RED if rc else ft.Colors.GREEN
        await self.show(1)

    # ------------------------------------------------------- логи
    def build_logs(self):
        self.log_view = ft.ListView(expand=True, spacing=2, auto_scroll=False)
        self.level = ft.Dropdown(
            label="Уровень", value="Все", width=140,
            options=[ft.dropdown.Option(x) for x in
                     ("Все", "INFO", "WARNING", "ERROR")])
        self.level.on_change = lambda e: self.refresh_logs()
        sw = ft.Switch(label="автообновление", value=False,
                       on_change=self.on_toggle_auto)
        self.btn_clear = ft.OutlinedButton(
            "Очистить", icon=ft.Icons.DELETE_OUTLINE, on_click=self.on_clear_logs)
        btns = ft.Row([
            ft.FilledTonalButton("Обновить", icon=ft.Icons.REFRESH,
                                 on_click=lambda e: self.refresh_logs()),
            self.btn_clear,
            sw, self.level])
        src = ft.Dropdown(
            label="Источник", value="app.log", width=200,
            options=[ft.dropdown.Option("app.log"),
                     ft.dropdown.Option("build.log"),
                     ft.dropdown.Option("индекс: fixed"),
                     ft.dropdown.Option("индекс: structural")])
        src.on_change = self.on_source_change
        btns.controls.append(src)
        self.log_source = src
        self.content.controls += [
            ft.Text("Логи и просмотр индексов", size=22, weight=ft.FontWeight.BOLD),
            btns,
            ft.Container(self.log_view, expand=True, border=ft.Border.all(
                1, ft.Colors.GREY_300), padding=8)]
        self.refresh_logs()

    def on_source_change(self, e):
        is_index = self.log_source.value.startswith("индекс:")
        self.level.disabled = is_index      # фильтр уровней к чанкам неприменим
        self.btn_clear.disabled = is_index  # очистка индекса из логов — нет
        self.refresh_logs()

    def _chunk_lines(self, strategy: str) -> list:
        if not os.path.exists(META_DB):
            return ["индекс не найден: meta.db отсутствует"]
        db = sqlite3.connect(META_DB)
        rows = db.execute(
            f"SELECT chunk_id, type, source, section, n_tokens, text "
            f"FROM chunks_{strategy} ORDER BY id").fetchall()
        db.close()
        lines = []
        for cid, typ, source, section, ntok, text in rows:
            preview = re.sub(r"\s+", " ", text)[:150]
            sec = section if section else "(без раздела)"
            lines.append(f"{cid} | {typ} | {source} | {sec} | "
                         f"{ntok} ток. | {preview}")
        log.info("просмотр индекса «%s»: %d чанков", strategy, len(lines))
        return lines

    def _log_lines(self):
        src = self.log_source.value
        if src.startswith("индекс:"):
            return self._chunk_lines(src.split(": ")[1])
        path = os.path.join(LOG_DIR, src)
        if not os.path.exists(path):
            return []
        lines = open(path, encoding="utf-8", errors="replace").readlines()
        lines = [ln.rstrip("\n") for ln in lines][-400:]
        lvl = self.level.value
        if lvl != "Все":
            lines = [ln for ln in lines if f" {lvl} " in ln]
        return list(reversed(lines))

    def refresh_logs(self):
        color_map = {"ERROR": ft.Colors.RED, "WARNING": ft.Colors.ORANGE}
        self.log_view.controls = [
            ft.Text(ln, size=12, font_family="monospace", selectable=True,
                    color=next((c for k, c in color_map.items()
                                if f" {k} " in ln), None))
            for ln in self._log_lines()]
        self.page.update()

    async def on_toggle_auto(self, e):
        self.auto_refresh = e.control.value
        log.info("логи: автообновление %s",
                 "включено" if self.auto_refresh else "выключено")
        if self.auto_refresh and self.logs_task is None:
            self.logs_task = asyncio.create_task(self._auto_loop())

    async def _auto_loop(self):
        try:
            while self.auto_refresh:
                await asyncio.sleep(3)
                self.refresh_logs()
        except asyncio.CancelledError:
            pass

    def on_clear_logs(self, e):
        if self.log_source.value.startswith("индекс:"):
            return  # индексы из страницы логов не очищаются
        open(os.path.join(LOG_DIR, self.log_source.value), "w").close()
        log.info("лог %s очищен", self.log_source.value)
        self.refresh_logs()

# ------------------------------------------------------------------ entry
async def main(page: ft.Page):
    page.title = "RAG-индекс 1С"
    page.theme_mode = ft.ThemeMode.LIGHT
    app = App(page)
    initial = {"/admin": 1, "/logs": 2}.get(page.route, 0)
    await app.show(initial)
    log.info("UI запущен (route=%s)", page.route)

    # предзагрузка модели в фоне, чтобы первый поиск не ждал ~минуту
    def preload():
        rag_search._get_model()
        log.info("модель эмбеддингов загружена")
        try:
            app.status.value = "Модель загружена — можно искать"
            app.status.color = ft.Colors.GREEN
            page.update()
        except Exception:
            pass
    asyncio.create_task(asyncio.to_thread(preload))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8550)
    args = parser.parse_args()
    log.info("запуск UI: host=%s port=%d", args.host, args.port)
    ft.run(main, view=ft.AppView.WEB_BROWSER,
           host=args.host, port=args.port, assets_dir=None)
