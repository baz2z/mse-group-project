from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Literal, NamedTuple

import numpy as np


class Document(NamedTuple):
    doc_id: str
    text: str


class RetrievalScore(NamedTuple):
    ranker: Literal["bm25", "sim", "nli"]
    doc_id: str
    score: float
    dist: np.ndarray = np.zeros(8)


class BaseRetriever(ABC):
    """
    Base class for retrievers.
    """

    # Hash of the main text corpus. Used to verify that the correct data is being used.
    CORPUS_HASH: str = (
        "09eee5888c6a2009773e3e8607a24f328494b28fcee7984b30356871d284fdd2"
    )

    @abstractmethod
    def score(self, query: str) -> list[RetrievalScore]:
        pass

    def check_corpus_hash(self, documents: list[Document]) -> bool:
        """
        Check if the documents match the main corpus hash.

        Args:
            documents: List of Document objects

        Returns:
            True if the documents match the main corpus hash, False otherwise.
        """
        # concatenate all document IDs and texts and hash them
        return (
            sha256(
                "".join(
                    doc.doc_id + doc.text
                    for doc in sorted(documents, key=lambda x: x.doc_id)
                ).encode(encoding="utf-8")
            ).hexdigest()
            == self.CORPUS_HASH
        )
