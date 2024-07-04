from typing import NamedTuple, TypedDict

from httpx import URL


class ScrapingRequest(NamedTuple):
    doc_id: str
    url: URL
    depth: int


class Index(TypedDict):
    url: str
    text: str
