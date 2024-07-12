import re
from typing import NamedTuple

from lingua import Language, LanguageDetectorBuilder
from markdownify import markdownify as md
from tqdm import tqdm

from crawler.config import HTML_DIR, MD_DIR

NEWLINE_PATTERN = re.compile(r"\n{3,}")

detector = LanguageDetectorBuilder.from_all_languages().build()


class ProcessedDoc(NamedTuple):
    doc_id: str
    md_text: str
    lang: Language | None

    @property
    def code(self) -> str:
        if self.lang is None:
            return "und"

        return self.lang.iso_code_639_3.name

    @property
    def fn(self) -> str:
        return f"{self.doc_id}_{self.code}.md"


def process_doc(content: bytes, doc_id: str):
    html_str = content.decode("utf-8", errors="ignore")
    md_text = md(html_str, strip=["a", "img"])
    md_text = NEWLINE_PATTERN.sub("\n\n", md_text).strip()

    if len(md_text) < 256:
        return ProcessedDoc(doc_id, md_text, None)

    lang = detector.detect_language_of(md_text)
    return ProcessedDoc(doc_id, md_text, lang)


def md_exists(doc_id: str) -> bool:
    return bool(list(MD_DIR.glob(f"{doc_id}_*.md")))


def process_all():
    pbar = tqdm(total=len(list(HTML_DIR.glob("*.html"))))
    for file in HTML_DIR.glob("*.html"):
        doc_id = file.stem
        if md_exists(doc_id):
            pbar.update(1)
            continue

        pbar.set_description(f"Processing: {doc_id}")
        try:
            content = file.read_bytes()
            res = process_doc(content, doc_id)
            (MD_DIR / res.fn).write_text(res.md_text, encoding="utf-8")
        except Exception as e:
            print(f"Error processing {doc_id}: {e}")

        pbar.update(1)


if __name__ == "__main__":
    process_all()
