# -*- coding: utf-8 -*-
"""
Ingestion-слой: адаптеры форматов -> единый список документов.

Документ: dict(doc_id, source, title, type, url, text)
  type: readme | article | doc | code | pdf
"""
import os
import re

import pymupdf as fitz  # PyMuPDF

CORPUS = os.path.join(os.path.dirname(__file__), "..", "corpus")

# ---------------------------------------------------------------- md / README
def _clean_md(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # картинки
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)      # ссылки -> текст
    text = re.sub(r"<[^>]+>", " ", text)                      # остатки HTML
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _md_title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else fallback

# ---------------------------------------------------------------- PDF
_JUNK_PDF = re.compile(
    r"^(©|\d+\s*$|SCIENTIFIC|Международный научно|electronic scientific)", re.I)

def _clean_pdf_page(text: str) -> str:
    lines = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln or _JUNK_PDF.match(ln):
            continue
        lines.append(ln)
    return "\n".join(lines)

def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    parts = [_clean_pdf_page(p.get_text()) for p in doc]
    text = "\n".join(p for p in parts if p)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# ---------------------------------------------------------------- реестр
URLS = {
    "habr_01.md": "https://habr.com/ru/articles/1088750/",
    "habr_02.md": "https://habr.com/ru/articles/1088070/",
    "habr_03.md": "https://habr.com/ru/companies/postgrespro/articles/1086788/",
    "habr_04.md": "https://habr.com/ru/companies/sportmaster_lab/articles/1089212/",
    "habr_05.md": "https://habr.com/ru/articles/1088652/",
    "habr_06.md": "https://habr.com/ru/articles/1086200/",
    "readme_onescript.md": "https://github.com/EvilBeaver/OneScript",
    "readme_vanessa_automation.md": "https://github.com/Pr-Mex/vanessa-automation",
    "pdf_cyberleninka_1c_yazyk.pdf": "https://cyberleninka.ru/article/n/osobennosti-izucheniya-studentami-vstroennogo-yazyka-programmirovaniya-1s-kak-vtorogo-i-posleduyuschego",
    "pdf_cyberleninka_diplomnoe.pdf": "https://cyberleninka.ru/article/n/ispolzovanie-vstroennogo-yazyka-programmirovaniya-1s-predpriyatie-8-3-v-diplomnom-proektirovanii",
    "pdf_cyberleninka_formalizatsiya.pdf": "https://cyberleninka.ru/article/n/formalizatsiya-dannyh-v-yazyke-programmirovaniya-1s",
    "pdf_cyberleninka_testirovanie_1cerp.pdf": "https://cyberleninka.ru/article/n/avtomatizatsiya-protsessa-testirovaniya-konfiguratsiy-1s-erp-upravlenie-predpriyatiem-v-kompanii-integratore",
}

PDF_TITLES = {
    "pdf_cyberleninka_1c_yazyk.pdf":
        "Особенности изучения студентами встроенного языка программирования 1С",
    "pdf_cyberleninka_diplomnoe.pdf":
        "Использование встроенного языка программирования 1С:Предприятие 8.3 в дипломном проектировании",
    "pdf_cyberleninka_formalizatsiya.pdf":
        "Формализация данных в языке программирования 1С",
    "pdf_cyberleninka_testirovanie_1cerp.pdf":
        "Автоматизация процесса тестирования конфигураций 1С:ERP в компании-интеграторе",
}

def doc_type(fname: str) -> str:
    if fname.endswith(".os"):
        return "code"
    if fname.endswith(".pdf"):
        return "pdf"
    # .md: префиксы сохранены для старых файлов, новые .md -> doc
    if fname.startswith("readme_"):
        return "readme"
    if fname.startswith("habr_"):
        return "article"
    return "doc"

def load_corpus(corpus_dir: str = CORPUS) -> list[dict]:
    docs = []
    for fname in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, fname)
        raw = open(path, encoding="utf-8", errors="replace").read() \
            if not fname.endswith(".pdf") else None
        if fname.endswith(".md"):
            text = _clean_md(raw)
            title = _md_title(raw, fname)
        elif fname.endswith(".os"):
            text = raw.strip()
            title = fname
        elif fname.endswith(".pdf"):
            text = load_pdf(path)
            title = PDF_TITLES.get(fname) or (text.split("\n")[0][:120] if text else fname)
        else:
            continue
        docs.append({
            "doc_id": os.path.splitext(fname)[0],
            "source": fname,
            "title": title,
            "type": doc_type(fname),
            "url": URLS.get(fname, ""),
            "text": text,
        })
    return docs

if __name__ == "__main__":
    for d in load_corpus():
        print(f'{d["type"]:8} {d["source"]:45} {len(d["text"]):7} зн. | {d["title"][:50]}')
