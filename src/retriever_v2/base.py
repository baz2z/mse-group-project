from abc import ABC, abstractmethod
from typing import Literal, NamedTuple


class Document(NamedTuple):
    doc_id: str
    text: str


class RetrievalScore(NamedTuple):
    ranker: Literal["bm25", "sim", "nli", "ensemble"]
    doc_id: str
    score: float


class BaseRetriever(ABC):
    @abstractmethod
    def score(self, query: str) -> list[RetrievalScore]:
        pass
