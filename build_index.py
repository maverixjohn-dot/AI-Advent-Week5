import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # HF недоступен напрямую
# -*- coding: utf-8 -*-
"""
Построение индекса: эмбеддинги (multilingual-e5-large) + FAISS + SQLite.

FAISS хранит только векторы; метаданные и текст — в SQLite,
связь по порядковому номеру (rowid = позиция в FAISS).
"""
import os
import sqlite3
import sys
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import build_all

BASE = os.path.join(os.path.dirname(__file__), "..", "index")
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH = 16


def build(strategy: str, model=None) -> dict:
    t0 = time.time()
    chunks = build_all(strategy)
    print(f"[{strategy}] чанков: {len(chunks)}")

    if model is None:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
    texts = ["passage: " + c["text"] for c in chunks]  # префикс обязателен для e5
    emb = model.encode(texts, batch_size=BATCH, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    emb = np.ascontiguousarray(emb.astype("float32"))

    os.makedirs(BASE, exist_ok=True)
    index = faiss.IndexFlatIP(emb.shape[1])   # нормированные векторы -> cosine
    index.add(emb)
    faiss.write_index(index, os.path.join(BASE, f"index_{strategy}.faiss"))

    db = sqlite3.connect(os.path.join(BASE, "meta.db"))
    db.execute(f"DROP TABLE IF EXISTS chunks_{strategy}")
    db.execute(f"""CREATE TABLE chunks_{strategy}(
        id INTEGER PRIMARY KEY, chunk_id TEXT, doc_id TEXT, source TEXT,
        title TEXT, section TEXT, type TEXT,
        char_start INT, char_end INT, n_tokens INT, text TEXT)""")
    db.executemany(
        f"INSERT INTO chunks_{strategy} VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        [(i, c["chunk_id"], c["doc_id"], c["source"], c["title"], c["section"],
          c["type"], c["char_start"], c["char_end"], c["n_tokens"], c["text"])
         for i, c in enumerate(chunks)])
    db.commit()
    db.close()

    import storage
    jp = storage.write_json(strategy, chunks, emb, MODEL_NAME)
    print(f"[{strategy}] JSON: {jp} ({os.path.getsize(jp) / 1e6:.1f} МБ)")

    dt = time.time() - t0
    print(f"[{strategy}] готово за {dt:.0f} c, dim={emb.shape[1]}")
    return {"chunks": len(chunks), "seconds": dt}


if __name__ == "__main__":
    # использование: python3 build_index.py [fixed|structural|all]
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = ("fixed", "structural") if which == "all" else (which,)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    stats = {s: build(s, model) for s in targets}
    print(stats)
