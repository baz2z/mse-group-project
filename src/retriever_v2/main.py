from pathlib import Path

import pandas as pd

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.bm25 import BM25Retriever
from retriever_v2.nli import NLIRetriever
from retriever_v2.sim import SimRetriever
from retriever_v2.utils import INDEX_DIR


class EnsembleRetriever(BaseRetriever):
    @classmethod
    def create(cls, index_path: Path = INDEX_DIR, **kwargs):
        index = pd.read_csv(index_path / "index.csv")
        documents = [
            Document(file.stem.split("_")[0], file.read_text(encoding="utf-8"))
            for file in (INDEX_DIR / "docs").glob("*.txt")
        ]
        sim_retriever = SimRetriever(documents=documents)
        bm25_retriever = BM25Retriever(documents=documents)
        nli_retriever = NLIRetriever(documents=documents)
        return cls(index, bm25_retriever, sim_retriever, nli_retriever, **kwargs)

    def __init__(
        self,
        index: pd.DataFrame,
        bm25_retriever: BM25Retriever,
        sim_retriever: SimRetriever,
        nli_retriever: NLIRetriever,
        pre_k: int = 200,
        max_res_per_domain: int = 20,
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
        return self.nli_retriever.score(
            query=query,
            filter_ids=set(pre_results.doc_id),
        )

    def query(self, query: str, *, k: int = 100) -> pd.DataFrame:
        scores = pd.DataFrame(self.score(query))
        df = pd.merge(self.index, scores, on="doc_id", how="inner").sort_values(
            "score", ascending=False
        )
        domain_count = df.groupby("domain").cumcount()
        df = pd.concat(
            [
                df.loc[domain_count < self.max_res_per_domain],
                df.loc[domain_count >= self.max_res_per_domain],
            ]
        )
        df["rank"] = list(range(1, len(df) + 1))
        return df.head(k)

    def query_batch(
        self, queries: list[str], *, k: int = 100
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError


if __name__ == "__main__":
    ensemble_retriever = EnsembleRetriever.create()
    r = ensemble_retriever.query("Carsten Eickhoff Tübingen")
    r.to_csv("example_food_and_drinks.csv", index=False)
