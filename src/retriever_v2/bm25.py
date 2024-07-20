import logging
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
from nltk import SnowballStemmer
from tqdm import tqdm

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import PICKLES_DIR, tokenize

STEMMER = SnowballStemmer("english")


class BM25(NamedTuple):
    idf: dict[str, float]
    doc_len: list[int]
    doc_freq: list[dict[str, int]]


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever that uses the BM25 algorithm.
    """

    def __init__(
        self,
        documents: list[Document],
        pickles_dir: Path = PICKLES_DIR,
        k1: float = 2,
        b: float = 1,
    ):
        """
        Initialize the BM25 retriever.

        Args:
            documents: List of Document objects
            pickles_dir: Directory to save/load BM25 config
            k1: BM25 parameter, controls term frequency scaling
            b: BM25 parameter, controls document length scaling
        """
        self.k1 = k1
        self.b = b
        self.ids: list[str] = [doc.doc_id for doc in documents]
        self.bm25: BM25 = self.load_bm25(documents=documents, pickles_dir=pickles_dir)

    @staticmethod
    def tokenize_documents(documents: list[Document]) -> list[list[str]]:
        """
        Tokenize documents.

        Args:
            documents: List of Document objects

        Returns:
            List of tokenized documents
        """
        return [
            [STEMMER.stem(word) for word in tokenize(doc.text, remove_tubingen=False)]
            for doc in tqdm(documents, desc="Tokenizing documents")
        ]

    def load_bm25(self, documents: list[Document], pickles_dir: Path) -> BM25:
        """
        Load BM25 config.

        Args:
            documents: List of Document objects
            pickles_dir: Directory to save/load BM25 config

        Returns:
            BM25 config
        """
        if not self.check_corpus_hash(documents):
            logging.warning("Corpus hash mismatch. Recomputing BM25.")
            return self.prep_bm25(documents)

        if (fp := pickles_dir / f"bm25.pkl").exists():
            return pickle.loads(fp.read_bytes())

        logging.warning("BM25 pickle not found. Recomputing BM25.")
        bm25 = self.prep_bm25(documents)
        fp.write_bytes(pickle.dumps(bm25))
        return bm25

    def prep_bm25(self, documents: list[Document]) -> BM25:
        """
        Preprocess documents to compute BM25 config.

        Args:
            documents: List of Document objects

        Returns:
            BM25 config
        """
        if not documents:
            raise ValueError("No documents provided")

        doc_len: list[int] = []
        doc_freq: list[dict[str, int]] = []

        nd: dict[str, int] = defaultdict(int)
        for doc in self.tokenize_documents(documents):
            frequencies: dict[str, int] = Counter(doc)
            doc_len.append(len(doc))
            doc_freq.append(frequencies)

            for word, freq in frequencies.items():
                nd[word] += 1

        idf: dict[str, float] = {}
        total_docs = len(documents)
        for word, freq in nd.items():
            idf[word] = math.log(total_docs - freq + 0.5) - math.log(freq + 0.5)

        return BM25(doc_freq=doc_freq, idf=idf, doc_len=doc_len)

    def score(self, query: str) -> list[RetrievalScore]:
        """
        Compute BM25 scores for each document given a query.

        Args:
            query: Query string, e.g., "tübingen football club"

        Returns:
            List of RetrievalScore objects
        """
        query_terms = tokenize(query, remove_tubingen=True) + ["tübingen"]
        query_terms_stemmed = [STEMMER.stem(word) for word in query_terms]

        score = np.zeros(len(self.ids))
        doc_len = np.array(self.bm25.doc_len)
        avg_doc_len = np.mean(doc_len)

        for q in query_terms_stemmed:
            q_freq = np.array([doc.get(q, 0) for doc in self.bm25.doc_freq])
            score += self.bm25.idf.get(q, 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len))
            )

        return [
            RetrievalScore(doc_id=doc_id, score=score, ranker="bm25")
            for doc_id, score in zip(self.ids, score)
        ]
