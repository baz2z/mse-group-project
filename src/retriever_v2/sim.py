from pathlib import Path

import numpy as np

from retriever_v2.base import BaseRetriever, RetrievalScore


class SIMRetriever(BaseRetriever):
    def __init__(self, embedding_dir: Path):
        self.embeddings = np.load(embedding_dir / "embeddings.npy")
        self.ids = np.load(embedding_dir / "ids.npy")

    def score(self, query: str) -> list[RetrievalScore]:
        return []
