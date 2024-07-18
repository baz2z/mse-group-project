from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

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
    def __init__(self, documents: list[Document]):
        self.ids = [doc.doc_id for doc in documents]
        self.bm25 = BM25Okapi([stem_tokenize(doc.text) for doc in documents])

    def score(self, query: str) -> list[RetrievalScore]:
        scores = self.bm25.get_scores(stem_tokenize(query))
        return [
            RetrievalScore(doc_id=doc_id, score=score, ranker="bm25")
            for doc_id, score in zip(self.ids, scores)
        ]


if __name__ == '__main__':
    docs = [
        Document("doc1", "This is a test document."),
        Document("doc2", "This document is another test."),
    ]
    retriever = BM25Retriever(docs)
    test_scores = retriever.score("test document")
    print(test_scores)
