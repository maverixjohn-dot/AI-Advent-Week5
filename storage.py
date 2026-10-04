# -*- coding: utf-8 -*-
"""
JSON-хранилище индекса: полный переносимый дамп чанков + эмбеддингов.

Роль в архитектуре: FAISS — быстрый поиск, SQLite — структурированные запросы
к метаданным, JSON — переносимое и человекочитаемое представление всего индекса
(метаданные + текст + векторы). Из JSON индекс восстанавливается без модели
эмбеддингов и без FAISS.

Формат index_{strategy}.json:
{
  "strategy": ..., "model": ..., "dim": 1024, "built_at": ...,
  "chunks": [{"id", "chunk_id", "doc_id", "source", "title", "section",
              "type", "char_start", "char_end", "n_tokens", "text",
              "embedding": [float, ...] (round 6)}]
}
"""
import json
import os
import sqlite3
import time

BASE = os.path.join(os.path.dirname(__file__), "..", "index")
META_DB = os.path.join(BASE, "meta.db")

META_FIELDS = ("id", "chunk_id", "doc_id", "source", "title", "section",
               "type", "char_start", "char_end", "n_tokens", "text")

def json_path(strategy: str) -> str:
    return os.path.join(BASE, f"index_{strategy}.json")

def write_json(strategy: str, chunks: list[dict], emb, model: str) -> str:
    """Пишет JSON из данных в памяти (вызывается из build_index.build)."""
    payload = {
        "strategy": strategy,
        "model": model,
        "dim": int(emb.shape[1]),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": [
            {**{k: c[k] for k in
                ("chunk_id", "doc_id", "source", "title", "section", "type",
                 "char_start", "char_end", "n_tokens", "text")},
             "id": i,
             "embedding": [round(float(x), 6) for x in emb[i]]}
            for i, c in enumerate(chunks)
        ],
    }
    path = json_path(strategy)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return path

def export_existing(strategy: str, model: str = "intfloat/multilingual-e5-large") -> str:
    """Экспорт JSON из уже построенных FAISS + SQLite (без перевычисления)."""
    import faiss
    import numpy as np

    index = faiss.read_index(os.path.join(BASE, f"index_{strategy}.faiss"))
    emb = np.empty((index.ntotal, index.d), dtype="float32")
    index.reconstruct_n(0, index.ntotal, emb)

    db = sqlite3.connect(META_DB)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        f"SELECT * FROM chunks_{strategy} ORDER BY id").fetchall()
    db.close()
    chunks = [dict(r) for r in rows]
    assert len(chunks) == index.ntotal, "рассинхрон FAISS и SQLite"
    return write_json(strategy, chunks, emb, model)

_cache: dict[str, tuple] = {}

def load_json(strategy: str):
    """Загружает JSON-индекс: (матрица векторов float32 [n, dim], метаданные)."""
    if strategy in _cache:
        return _cache[strategy]
    import numpy as np

    with open(json_path(strategy), encoding="utf-8") as f:
        payload = json.load(f)
    emb = np.array([c["embedding"] for c in payload["chunks"]], dtype="float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    meta = [{k: c[k] for k in META_FIELDS} for c in payload["chunks"]]
    _cache[strategy] = (emb, meta)
    return _cache[strategy]

if __name__ == "__main__":
    import sys
    for s in (sys.argv[1:] or ["fixed", "structural"]):
        p = export_existing(s)
        print(f"{s}: {p} ({os.path.getsize(p) / 1e6:.1f} МБ)")
