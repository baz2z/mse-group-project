import pickle
from pathlib import Path

import pandas as pd

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.bm25 import BM25Retriever
from retriever_v2.nli import NLIRetriever
from retriever_v2.sim import SimRetriever
from retriever_v2.utils import INDEX_DIR


class EnsembleRetriever(BaseRetriever):
    @classmethod
    def create(cls, index_path: Path = INDEX_DIR):
        index = pd.read_csv(index_path / "index.csv")
        documents = [
            Document(file.stem["_"][0], file.read_text(encoding="utf-8"))
            for file in (INDEX_DIR / "docs").glob("*.txt")
        ]
        sim_retriever = SimRetriever()
        bm25_retriever = BM25Retriever(documents=documents)
        nli_retriever = NLIRetriever(documents=documents)
        return cls(index, bm25_retriever, sim_retriever, nli_retriever)

    @classmethod
    def from_saved(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self):
        with open("ensemble_retriever.pkl", "wb") as f:
            pickle.dump(self, f)

    def __init__(
            self,
            index: pd.DataFrame,
            bm25_retriever: BM25Retriever,
            sim_retriever: SimRetriever,
            nli_retriever: NLIRetriever,
            pre_k: int = 1_000,
            max_res_per_domain: int = 10,
    ):
        self.index = index
        self.bm25_retriever = bm25_retriever
        self.sim_retriever = sim_retriever
        self.nli_retriever = nli_retriever
        self.pre_k = pre_k
        self.max_res_per_domain = max_res_per_domain

    def score(self, query: str) -> list[RetrievalScore]:
        bm25_results = self.bm25_retriever.score(query=query)
        sim_results = self.sim_retriever.score(query=query)
        pre_results = pd.DataFrame(bm25_results + sim_results)
        pre_results["rank"] = pre_results.groupby("ranker")["score"].rank(
            method="first", ascending=False
        )
        pre_results = (
            pre_results.sort_values("rank", ascending=True)
            .drop_duplicates(subset="doc_id")
            .head(self.pre_k)
        )
        return list(
            self.nli_retriever.score(
                query=query,
                filter_ids=set(pre_results.doc_id),
            )
        )

    def query(self, query: str, *, k: int = 100) -> pd.DataFrame:
        scores = pd.DataFrame(self.score(query))
        df = pd.merge(self.index, scores, on="doc_id", how="inner")
        return (
            df.sort_values("score", ascending=False)
            .loc[df.groupby("domain").cumcount() < self.max_res_per_domain]
            .head(k)
        )

    def query_batch(
            self, queries: list[str], *, k: int = 100
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError
