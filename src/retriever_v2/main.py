from pathlib import Path

import pandas as pd

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.bm25 import BM25Retriever
from retriever_v2.nli import NLIRetriever
from retriever_v2.sim import SimRetriever
from retriever_v2.utils import INDEX_DIR


class EnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever that combines BM25, similarity, and NLI retrievers.
    """
    def __init__(
        self,
        index_path: Path = INDEX_DIR,
        pre_k: int = 128,
        max_res_per_domain: int = 20,
    ):
        """
        Initialize the ensemble retriever.

        Args:
            index_path: Path to the index directory
            pre_k: Number of documents to retrieve in pre-ranking
            max_res_per_domain: Maximum number of results per domain

        Returns:
            None
        """
        documents = [
            Document(file.stem.split("_")[0], file.read_text(encoding="utf-8"))
            for file in (index_path / "docs").glob("*.txt")
        ]
        if not self.check_corpus_hash(documents):
            raise ValueError("Corpus hash mismatch. Please re-index.")

        self.index = pd.read_csv(index_path / f"index.csv")
        self.bm25_retriever = BM25Retriever(documents=documents)
        self.sim_retriever = SimRetriever(documents=documents)
        self.nli_retriever = NLIRetriever(documents=documents)
        self.pre_k = pre_k
        self.max_res_per_domain = max_res_per_domain

    def score(self, query: str) -> list[RetrievalScore]:
        """
        Score documents based on the ensemble retriever.

        Args:
            query: Query string, e.g., "tübingen football club"

        Returns:
            List of RetrievalScore objects
        """
        # Pre-rank with BM25 and similarity retrievers
        bm25_results = self.bm25_retriever.score(query=query)
        sim_results = self.sim_retriever.score(query=query)
        pre_results = pd.DataFrame(bm25_results + sim_results)

        # Rank the pre-results, keep the lower rank for each doc_id, and filter the top pre_k
        pre_results["rank"] = pre_results.groupby("ranker")["score"].rank(
            method="first", ascending=False
        )
        pre_results = (
            pre_results.sort_values("rank", ascending=True)
            .drop_duplicates(subset="doc_id")
            .head(self.pre_k)
        )

        # Score the pre-results with the NLI retriever
        return self.nli_retriever.score(
            query=query,
            filter_ids=set(pre_results.doc_id),
        )

    def query(self, query: str, *, k: int = 100) -> pd.DataFrame:
        """
        Query the ensemble retriever.

        Args:
            query: Query string, e.g., "tübingen football club"
            k: Number of results to return

        Returns:
            DataFrame with the top k results
        """
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
        """
        Query the ensemble retriever in batch.

        Args:
            queries: List of query strings
            k: Number of results to return

        Returns:
            Dictionary of DataFrames with the top k results for each query
        """
        raise NotImplementedError


if __name__ == "__main__":
    ensemble_retriever = EnsembleRetriever()
    r = ensemble_retriever.query("attractions")
    r.to_csv("example_food_and_drinks.csv", index=False)
