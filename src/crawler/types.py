from enum import auto, Enum, StrEnum
from typing import NamedTuple, TypedDict

from httpx import URL


class ScrapingRequest(NamedTuple):
    doc_id: str
    url: URL
    root: str
    depth: int


class Index(TypedDict):
    url: str
    text: str


class Priority(Enum):
    high = 1
    low = 0


class Status(StrEnum):
    pending = auto()
    completed = auto()
    failed = auto()
