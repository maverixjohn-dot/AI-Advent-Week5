import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # HF недоступен напрямую
# -*- coding: utf-8 -*-
"""
Поиск по индексу: search(query, k, strategy) -> top-k чанков с метаданными.
"""
import os
import sqlite3

import faiss
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "index")
MODEL_NAME = "intfloat/multilingual-e5-large"

_model = None
_indexes = {}

def _load(strategy: str):
    if strategy not in _indexes:
        _indexes[strategy] = faiss.read_index(
            os.path.join(BASE, f"index_{strategy}.faiss"))
    return _indexes[strategy]

def _get_model():
    # ленивый импорт: страницы администрирования/логов работают без torch
    from sentence_transformers import SentenceTransformer
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model

def search(query: str, k: int = 5, strategy: str = "structural",
           backend: str = "faiss") -> list[dict]:
    """backend: "faiss" (по умолчанию) или "json" (поиск по JSON-дампу,
    numpy cosine — демонстрация, что JSON-хранилище самодостаточно)."""
    model = _get_model()
    q = model.encode(["query: " + query], normalize_embeddings=True,
                     convert_to_numpy=True).astype("float32")
    if backend == "json":
        import storage
        emb, meta = storage.load_json(strategy)
        import numpy as np
        scores = emb @ q[0]
        top = np.argsort(-scores)[:k]
        return [{**meta[int(i)], "score": float(scores[int(i)])} for i in top]
    index = _load(strategy)
    scores, ids = index.search(q, k)
    db = sqlite3.connect(os.path.join(BASE, "meta.db"))
    db.row_factory = sqlite3.Row
    out = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        row = db.execute(
            f"SELECT * FROM chunks_{strategy} WHERE id=?", (int(i),)).fetchone()
        out.append({**dict(row), "score": float(score)})
    db.close()
    return out

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "как запустить тестирование конфигурации"
    strat = sys.argv[2] if len(sys.argv) > 2 else "structural"
    for r in search(q, 5, strat):
        print(f'{r["score"]:.3f} | {r["source"]} | {r["section"][:40]} | '
              f'{r["text"][:90].replace(chr(10), " ")}...')
