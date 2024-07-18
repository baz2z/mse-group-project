import shutil
from typing import NamedTuple

import pandas as pd
from bs4 import BeautifulSoup
from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm

from crawler.config import HARD_DRIVE

BASE_DIR = HARD_DRIVE / "mse_latest"

detector = LanguageDetectorBuilder.from_all_languages().build()

exclude_tags = {
    "script",
    "style",
    "header",
    "footer",
    "aside",
    "nav",
    "form",
    "iframe",
    "noscript",
    "link",
    "meta",
    "input",
    "button",
    "select",
    "textarea",
    "option",
    "datalist",
    "output",
    "canvas",
    "svg",
    "object",
    "embed",
    "audio",
    "video",
    "source",
    "track",
    "map",
    "area",
    "applet",
    "param",
    "col",
    "colgroup",
    "thead",
    "tfoot",
    "tbody",
    "th",
    "tr",
    "td",
    "caption",
    "figure",
    "figcaption",
    "a",
    "img",
}
include_tags = {"title", "h1", "h2", "h3", "h4", "h5", "h6", "p"}

HTML_DIR = HARD_DRIVE / "mse_latest" / "html"
TXT_DIR = HARD_DRIVE / "mse_latest" / "txt"
TXT_EN_DIR = HARD_DRIVE / "mse_latest" / "txt_en"

TXT_DIR.mkdir(parents=True, exist_ok=True)
TXT_EN_DIR.mkdir(parents=True, exist_ok=True)


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

    for tag in soup.find_all(exclude_tags):
        tag.decompose()

    text_content = "\n\n".join(
        " ".join(p.get_text().split())
        for p in soup.find_all(include_tags)
        if len(p.get_text().split()) > 3
    )
    if len(text_content.split()) < 32:
        return ProcessedDoc(doc_id, text_content, None)

    lang = detector.detect_language_of(text_content)
    return ProcessedDoc(doc_id, text_content, lang)


def process_all():
    mappings = (
        pd.read_csv(BASE_DIR / "latest.csv")
        .query("status == 'completed' and features_tubingen")
        .sort_values("created", ascending=True)
        .drop_duplicates("url", keep="first")
    )
    txt_exist = set(fn.stem.split("_")[0] for fn in TXT_DIR.glob("*.txt"))

    for doc_id in tqdm(mappings.doc_id.unique()):
        if doc_id in txt_exist:
            continue

        html_file = HTML_DIR / f"{doc_id}.html"
        if not html_file.exists():
            print(f"File {html_file} does not exist.")
            continue

        try:
            res = process_doc(html_file.read_bytes(), doc_id)
        except Exception as e:
            print(f"Error processing {doc_id}: {e}")
            res = ProcessedDoc(doc_id, "", None)

        f_out = TXT_DIR / res.fn
        f_out.write_text(res.text, encoding="utf-8")


def clean_dir():
    mappings = (
        pd.read_csv(BASE_DIR / "latest.csv")
        .query("status == 'completed' and features_tubingen")
        .sort_values("created", ascending=True)
        .drop_duplicates("url", keep="first")
    )
    for file in TXT_DIR.glob("*.txt"):
        doc_id = file.stem.split("_")[0]
        if doc_id not in mappings.doc_id.values:
            file.unlink()


def copy_en():
    for file in TXT_DIR.glob("*_ENG.txt"):
        f_out = TXT_EN_DIR / file.name

        if f_out.exists():
            continue

        shutil.copy(file, f_out)


if __name__ == "__main__":
    copy_en()
