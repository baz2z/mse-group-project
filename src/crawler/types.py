from enum import auto, Enum, StrEnum
from typing import NamedTuple

from httpx import URL


class ScrapingRequest(NamedTuple):
    """
    NamedTuple for the scraping request.
    """

    doc_id: str
    url: URL
    root: str
    depth: int


class Priority(Enum):
    """
    Enum for the priority of the scraping request.
    """

    high = 1
    low = 0


class Status(StrEnum):
    """
    Enum for the status of the scraping request.
    """

    pending = auto()
    completed = auto()
    failed = auto()
