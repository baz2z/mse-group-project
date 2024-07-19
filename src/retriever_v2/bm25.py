from pathlib import Path
from pickle import dump, load

from nltk import SnowballStemmer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import PICKLES_DIR, tokenize

STEMMER = SnowballStemmer("english")


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
        query_terms = tokenize(query, remove_tubingen=True) + ["tübingen"]
        query_terms_stemmed = [STEMMER.stem(word) for word in query_terms]
        scores = self.bm25.get_scores(query_terms_stemmed)
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
            [
                [
                    STEMMER.stem(word)
                    for word in tokenize(doc.text, remove_tubingen=False)
                ]
                for doc in tqdm(documents, desc="Precomputing BM25")
            ],
            k1=k1,
            b=b,
        )
        with open(pickles_dir / "bm25.pkl", "wb") as f:
            dump(bm25, f)

        return bm25
