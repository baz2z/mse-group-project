import sys
import pandas as pd
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.bm25 import BM25Retriever
from retriever_v2.nli import NLIRetriever
from retriever_v2.sim import SimRetriever
from retriever_v2.utils import INDEX_DIR, QUERIES_FILE


class EnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever that combines BM25, similarity, and NLI retrievers.
    """

    def __init__(
        self,
        index_path: Path = INDEX_DIR,
        pre_k: int = 512,
        max_res_per_domain: int = 20,
        use_fast: bool = False,
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

        self.use_fast = use_fast
        self.pre_k = pre_k
        self.max_res_per_domain = max_res_per_domain

        self.index = pd.read_csv(index_path / f"index.csv")
        self.bm25_retriever = BM25Retriever(documents=documents)
        self.sim_retriever = SimRetriever(documents=documents)
        self.nli_retriever = NLIRetriever(documents=documents, use_fast=use_fast)

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
        if self.use_fast:
            self.pre_k = k

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
        self, queries_file: Path, *, k: int = 100, save: bool = True
    ) -> pd.DataFrame:
        """
        Query the ensemble retriever in batch.

        Args:
            query_file (Path): Path to the query batch file (.txt)
            k (int): Number of results to return
            save (bool): Whether to save the results to a file

        Returns:
            DataFrame with the top k results for each query. 
            Columns: ['Query Number', 'Rank Number', 'URL', 'Score']
        """
        queries = [line for line in queries_file.read_text(encoding="utf-8").splitlines()]
        all_results = []

        for query_number, query in enumerate(queries, start=1):
            df = self.query(query, k=k)
            for rank, row in enumerate(df.itertuples(index=False), start=1):
                all_results.append([query_number, rank, row.url, row.score])
                if rank == k:
                    break

        results_df = pd.DataFrame(all_results, columns=['Query Number', 'Rank Number', 'URL', 'Score'])

        if save:
            results_df.to_csv(INDEX_DIR / "results.csv", index=False)
            results_df.to_csv(INDEX_DIR / "results.txt", sep="\t", index=False, header=False)

        return results_df


if __name__ == "__main__":
    
    ensemble_retriever = EnsembleRetriever(use_fast=False, pre_k=512)
    rs = ensemble_retriever.query_batch(QUERIES_FILE)
    
    # query = "attractions"
    # r = ensemble_retriever.query(query)
    # r.to_csv(f"example_{query}.csv", index=False)
    
    
    
    
    
    
