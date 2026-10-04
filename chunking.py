import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # HF недоступен напрямую
# -*- coding: utf-8 -*-
"""
Две стратегии chunking.

1. fixed      — окна по 512 токенов, перекрытие 15% (77 токенов).
                Конец окна притягивается к ближайшему пробелу (<= 20 токенов),
                чтобы не резать слова; предложения и абзацы режется — это
                осознанный, документируемый дефект стратегии.
2. structural — по структуре документа:
                md  -> по заголовкам (#/##/###), путь раздела сохраняется;
                code -> по процедурам/функциям 1С (Процедура/Функция ...
                        КонецПроцедуры/КонецФункции);
                pdf -> эвристика заголовков (нумерованные/короткие строки,
                       Введение/Заключение/...), fallback — как fixed.
                Куски > 512 токенов дорежутся вторично окнами с overlap,
                куски < 80 токенов склеиваются с соседом.

Чанк: dict(chunk_id, doc_id, source, title, section, type,
           char_start, char_end, n_tokens, text)
"""
import re
from typing import Callable

from transformers import AutoTokenizer

from ingest import load_corpus

MAX_TOKENS = 512
OVERLAP = 77          # 15%
MIN_TOKENS = 80
SNAP_WINDOW = 20      # допуск притяжки конца окна к пробелу

_tokenizer = None
def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    return _tokenizer

def n_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])

def token_spans(text: str):
    """Возвращает (offsets, total_tokens): offsets[i] = char-позиция i-го токена."""
    enc = tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["offset_mapping"], len(enc["input_ids"])

# ============================================================ стратегия 1
def fixed_chunks(doc: dict) -> list[dict]:
    text = doc["text"]
    offsets, total = token_spans(text)
    chunks, seq, start_tok = [], 0, 0
    while start_tok < total:
        end_tok = min(start_tok + MAX_TOKENS, total)
        if end_tok < total:  # притяжка конца окна к пробелу
            snapped = end_tok
            for t in range(end_tok, max(end_tok - SNAP_WINDOW, start_tok), -1):
                c0 = offsets[t - 1][1]
                if c0 < len(text) and text[c0:c0 + 1].isspace():
                    snapped = t
                    break
            end_tok = snapped
        c_start = offsets[start_tok][0]
        c_end = offsets[end_tok - 1][1] if end_tok > start_tok else len(text)
        chunk_text = text[c_start:c_end].strip()
        if chunk_text:
            chunks.append(_mk(doc, seq, "", c_start, c_end,
                              end_tok - start_tok, chunk_text))
            seq += 1
        if end_tok >= total:
            break
        start_tok = end_tok - OVERLAP
    return chunks

# ============================================================ общие куски
def _mk(doc, seq, section, c0, c1, ntok, text):
    return {
        "chunk_id": f'{doc["doc_id"]}:{seq}',
        "doc_id": doc["doc_id"],
        "source": doc["source"],
        "title": doc["title"],
        "section": section,
        "type": doc["type"],
        "char_start": c0,
        "char_end": c1,
        "n_tokens": ntok,
        "text": text,
    }

def _secondary_split(doc, base_seq, section, text, doc_char0, pieces):
    """Дорезка куска > MAX_TOKENS окнами с overlap (как fixed, но локально)."""
    sub_doc = dict(doc, text=text)
    subs = fixed_chunks(sub_doc)
    for j, s in enumerate(subs):
        pieces.append(_mk(doc, base_seq + j, section,
                          doc_char0 + s["char_start"], doc_char0 + s["char_end"],
                          s["n_tokens"], s["text"]))
    return base_seq + len(subs)

def _emit_sections(doc, sections, min_merge=True):
    """sections: list[(section_name, char0, char1)] -> chunks"""
    if min_merge:  # склейка слишком мелких разделов с следующим
        merged, buf = [], None
        for name, c0, c1 in sections:
            if buf is not None:
                c0, name = buf[0], (buf[2] + " / " + name if buf[2] else name)
                buf = None
            if n_tokens(doc["text"][c0:c1]) < MIN_TOKENS:
                buf = (c0, c1, name)
                continue
            merged.append((name, c0, c1))
        if buf is not None:
            if merged:
                n0, s0, e0 = merged[-1]
                merged[-1] = (n0 + " / " + buf[2], s0, buf[1])
            else:
                merged.append((buf[2], buf[0], buf[1]))
    pieces, seq = [], 0
    for name, c0, c1 in merged:
        seg = doc["text"][c0:c1].strip()
        if not seg:
            continue
        nt = n_tokens(seg)
        if nt <= MAX_TOKENS:
            pieces.append(_mk(doc, seq, name, c0, c1, nt, seg))
            seq += 1
        else:
            seq = _secondary_split(doc, seq, name, doc["text"][c0:c1], c0, pieces)
    return pieces

# ------------------------------------------------- md: заголовки
_HDR = re.compile(r"^(#{1,6})\s+(.*)$", re.M)

def _md_sections(text):
    marks = [(m.start(), len(m.group(1)),
              m.group(2).strip().rstrip("#").strip())
             for m in _HDR.finditer(text)]
    if not marks:
        return [("", 0, len(text))]
    sections = []
    if marks[0][0] > 0:
        sections.append(("(вводная часть)", 0, marks[0][0]))
    stack = []  # [(level, title)]
    for i, (pos, lvl, title) in enumerate(marks):
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        stack.append((lvl, title))
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        sections.append((" / ".join(t for _, t in stack), pos, end))
    return sections

# ------------------------------------------------- code: процедуры 1С
_PROC = re.compile(
    r"^\s*(?:&\w+\s*\n\s*)?(?:Асинх\s+)?(Процедура|Функция|Procedure|Function)"
    r"\s+([\wА-Яа-я]+)\s*\(", re.M | re.I)
_ENDP = re.compile(r"^\s*(КонецПроцедуры|КонецФункции|EndProcedure|EndFunction)",
                   re.M | re.I)

def _code_sections(text):
    starts = [(m.start(), f'{m.group(1)} {m.group(2)}') for m in _PROC.finditer(text)]
    if not starts:
        return [("", 0, len(text))]
    ends = [m.end() for m in _ENDP.finditer(text)]
    sections = []
    if starts[0][0] > 0:
        sections.append(("(заголовок модуля)", 0, starts[0][0]))
    for i, (pos, name) in enumerate(starts):
        end = next((e for e in ends if e > pos),
                   starts[i + 1][0] if i + 1 < len(starts) else len(text))
        sections.append((name, pos, end))
    return sections

# ------------------------------------------------- pdf: эвристика заголовков
_PDF_HDR = re.compile(
    r"^(?:\d+\.?\d*\.?\s+)?(Введение|Заключение|Выводы?|Литература"
    r"|Список литературы|Методика[^\n]{0,60}|Результаты[^\n]{0,60}"
    r"|[А-ЯЁ][А-ЯЁ\s\-]{8,80})$", re.M)
_PDF_NUMHDR = re.compile(r"^(\d+\.?\s+[А-ЯЁ][^,\n]{4,58}[^.,\n])$", re.M)

def _pdf_sections(text):
    marks = [(m.start(), m.group(0).strip()) for m in _PDF_HDR.finditer(text)]
    marks = [m for m in marks if len(m[1]) < 90]
    if len(marks) < 2:
        return [("", 0, len(text))]
    sections = []
    if marks[0][0] > 0:
        sections.append(("(начало документа)", 0, marks[0][0]))
    for i, (pos, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        sections.append((title, pos, end))
    return sections

# ============================================================ стратегия 2
def structural_chunks(doc: dict) -> list[dict]:
    if doc["type"] == "code":
        sections = _code_sections(doc["text"])
    elif doc["type"] == "pdf":
        sections = _pdf_sections(doc["text"])
    else:
        sections = _md_sections(doc["text"])
    return _emit_sections(doc, sections)

# ============================================================ прогон
STRATEGIES: dict[str, Callable] = {
    "fixed": fixed_chunks,
    "structural": structural_chunks,
}

def build_all(strategy: str) -> list[dict]:
    docs = load_corpus()
    fn = STRATEGIES[strategy]
    out = []
    for d in docs:
        out.extend(fn(d))
    return out

if __name__ == "__main__":
    import sys
    for strat in ("fixed", "structural"):
        chunks = build_all(strat)
        sizes = [c["n_tokens"] for c in chunks]
        print(f"{strat}: {len(chunks)} чанков | "
              f"mean={sum(sizes)/len(sizes):.0f} "
              f"min={min(sizes)} max={max(sizes)}")
