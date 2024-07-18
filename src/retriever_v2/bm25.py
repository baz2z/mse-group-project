from retriever_v2.base import BaseRetriever, RetrievalScore


class BM25Retriever(BaseRetriever):
    def score(self, query: str) -> list[RetrievalScore]:
        return []
