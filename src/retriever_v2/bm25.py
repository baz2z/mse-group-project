from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import STOPWORDS, TOKEN_PATTERN

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
        documents: list[Document],
        k1: float = 2,
        b: float = 1,
    ):
        self.ids = [doc.doc_id for doc in documents]
        self.bm25 = BM25Okapi(
            [stem_tokenize(doc.text) for doc in tqdm(documents)],
            k1=k1,
            b=b,
        )

    def score(self, query: str) -> list[RetrievalScore]:
        if not (query_tokenized := stem_tokenize(query)):
            return []

        scores = self.bm25.get_scores(query_tokenized)
        return [
            RetrievalScore(doc_id=doc_id, score=score, ranker="bm25")
            for doc_id, score in zip(self.ids, scores)
        ]
