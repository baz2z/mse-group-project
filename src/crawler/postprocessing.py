from typing import NamedTuple

import pandas as pd
from bs4 import BeautifulSoup
from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm

from crawler.config import BASE_DIR, HTML_DIR, TXT_DIR

EXCLUDE_TAGS = {
    "script",
    "style",
    "head",
    "header",
    "meta",
    "link",
    "noscript",
    "iframe",
    "nav",
    "footer",
    "aside",
    "a",
    "img",
}

detector = LanguageDetectorBuilder.from_all_languages().build()


class ProcessedDoc(NamedTuple):
    doc_id: str
    text: str
    lang: Language or None

    @property
    def code(self):
        if self.lang is None:
            return "UNK"

        return self.lang.iso_code_639_3.name

    @property
    def fn(self) -> str:
        return f"{self.doc_id}_{self.code}.txt"


def process_doc(content: bytes, doc_id: str):
    html_content = content.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup.find_all(EXCLUDE_TAGS):
        tag.decompose()

    text_content = soup.get_text(strip=True, separator=" ")
    text_content = " ".join(text_content.split())
    if len(text_content) < 128:
        return ProcessedDoc(doc_id, text_content, None)

    lang = detector.detect_language_of(text_content)
    return ProcessedDoc(doc_id, text_content, lang)


def txt_exists(doc_id: str) -> bool:
    return bool(list(TXT_DIR.glob(f"{doc_id}_*.txt")))


def process_all():
    mappings = pd.read_csv(BASE_DIR / "20240713030800.csv").query(
        "status == 'completed'"
    )
    for doc_id in tqdm(mappings.doc_id.unique()):
        fn = HTML_DIR / f"{doc_id}.html"
        if not fn.exists():
            print(f"File {fn} does not exist")
            continue

        if txt_exists(doc_id):
            continue

        try:
            res = process_doc(fn.read_bytes(), doc_id)
        except Exception as e:
            print(f"Error processing {doc_id}: {e}")
            res = ProcessedDoc(doc_id, "", None)

        (TXT_DIR / res.fn).write_text(res.text, encoding="utf-8")


if __name__ == "__main__":
    process_all()
