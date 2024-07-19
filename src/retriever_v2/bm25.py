from pathlib import Path
from pickle import dump, load

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import PICKLES_DIR, STOPWORDS, TOKEN_PATTERN

STEMMER = SnowballStemmer("english")


def stem_tokenize(text: str):
    return [
        STEMMER.stem(word.lower())
        for word in word_tokenize(text)
        if word.lower() not in STOPWORDS and TOKEN_PATTERN.match(word)
    ]


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        pickles_dir: Path = PICKLES_DIR,
        documents: list[Document] = None,
        k1: float = 2,
        b: float = 1,
    ):
        self.ids = [doc.doc_id for doc in documents]

        try:
            with open(pickles_dir / "bm25.pkl", "rb") as f:
                self.bm25 = load(f)
        except FileNotFoundError:
            if not documents:
                raise ValueError("No documents provided and no index found")

            self.bm25 = self.precompute(documents, pickles_dir, k1, b)

    def score(self, query: str) -> list[RetrievalScore]:
        if not (query_tokenized := stem_tokenize(query)):
            return []

        scores = self.bm25.get_scores(query_tokenized)
        return [
            RetrievalScore(doc_id=doc_id, score=score, ranker="bm25")
            for doc_id, score in zip(self.ids, scores)
        ]

    @staticmethod
    def precompute(
        documents: list[Document],
        pickles_dir: Path = PICKLES_DIR,
        k1: float = 2,
        b: float = 1,
    ):
        bm25 = BM25Okapi(
            [stem_tokenize(doc.text) for doc in tqdm(documents)],
            k1=k1,
            b=b,
        )
        with open(pickles_dir / "bm25.pkl", "wb") as f:
            dump(bm25, f)

        return bm25
