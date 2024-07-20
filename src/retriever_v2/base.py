from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Literal, NamedTuple


class Document(NamedTuple):
    doc_id: str
    text: str


class RetrievalScore(NamedTuple):
    ranker: Literal["bm25", "sim", "nli"]
    doc_id: str
    score: float


class BaseRetriever(ABC):
    @abstractmethod
    def score(self, query: str) -> list[RetrievalScore]:
        pass

    @staticmethod
    def integrity_hash(documents: list[Document]) -> str:
        return sha256(
            "".join(
                doc.doc_id + doc.text
                for doc in sorted(documents, key=lambda x: x.doc_id)
            ).encode(encoding="utf-8")
        ).hexdigest()
