import shutil
from typing import NamedTuple

import pandas as pd
from bs4 import BeautifulSoup
from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm

from crawler.config import HARD_DRIVE

BASE_DIR = HARD_DRIVE / "mse_latest"

# Language detector
detector = LanguageDetectorBuilder.from_all_languages().build()

# Tags to exclude from the HTML content when extracting text
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

# Tags to include from the HTML content when extracting text
include_tags = {"title", "h1", "h2", "h3", "h4", "h5", "h6", "p"}

# Directories
HTML_DIR = HARD_DRIVE / "mse_latest" / "html"
TXT_DIR = HARD_DRIVE / "mse_latest" / "txt"
TXT_EN_DIR = HARD_DRIVE / "mse_latest" / "txt_en"

TXT_DIR.mkdir(parents=True, exist_ok=True)
TXT_EN_DIR.mkdir(parents=True, exist_ok=True)


class ProcessedDoc(NamedTuple):
    """
    Processed document object.
    """

    doc_id: str
    text: str
    lang: Language or None

    @property
    def code(self):
        """
        Get the ISO 639-3 language code of the document.

        Returns:
            str: ISO 639-3 language code or "UNK" if the language is unknown.
        """
        if self.lang is None:
            return "UNK"

        return self.lang.iso_code_639_3.name

    @property
    def fn(self) -> str:
        """
        Get the filename of the document.

        Returns:
            str: Filename of the document in the format "{doc_id}_{code}.txt".
        """
        return f"{self.doc_id}_{self.code}.txt"


def process_doc(content: bytes, doc_id: str) -> ProcessedDoc:
    """
    Process the HTML content of a document. Extract the text and detect the language.

    Args:
        content: HTML content of the document.
        doc_id: Document ID.

    Returns:
        ProcessedDoc: Processed document object.
    """
    # Parse the HTML content
    html_content = content.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html_content, "html.parser")

    # Drop the excluded tags
    for tag in soup.find_all(exclude_tags):
        tag.decompose()

    # Extract the text content, keep only paragraphs with more than 3 words
    text_content = "\n\n".join(
        " ".join(p.get_text().split())
        for p in soup.find_all(include_tags)
        if len(p.get_text().split()) > 3
    )

    # Skip documents with less than 32 words
    if len(text_content.split()) < 32:
        return ProcessedDoc(doc_id, text_content, None)

    lang = detector.detect_language_of(text_content)
    return ProcessedDoc(doc_id, text_content, lang)


def process_all():
    """
    Process all HTML documents and save the text content to TXT files.

    Returns:
        None
    """
    # Load the latest mappings and filter the completed documents with tuebingen features
    mappings = (
        pd.read_csv(BASE_DIR / "latest.csv")
        .query("status == 'completed' and features_tubingen")
        .sort_values("created", ascending=True)
        .drop_duplicates("url", keep="first")
    )

    # Get the existing TXT files
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
    """
    Clean the TXT directory by removing TXT files that are not in the latest mappings.

    Returns:
        None
    """
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
    """
    Copy the English TXT files to the TXT_EN directory.

    Returns:
        None
    """
    for file in TXT_DIR.glob("*_ENG.txt"):
        f_out = TXT_EN_DIR / file.name

        if f_out.exists():
            continue

        shutil.copy(file, f_out)


if __name__ == "__main__":
    process_all()
    clean_dir()
    copy_en()
