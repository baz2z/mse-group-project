import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import DEVICE, EMBEDDINGS_DIR, tokenize

MODEL_NAME = "all-MiniLM-L12-v2"


class SimRetriever(BaseRetriever):
    def __init__(
        self,
        documents: list[Document],
        model_name: str = MODEL_NAME,
        embedding_dir: Path = EMBEDDINGS_DIR,
        device: torch.device = DEVICE,
    ):
        self.model = SentenceTransformer(model_name).to(device)
        self.ids, self.embeddings = self._load_embeddings(
            documents=documents,
            model_name=model_name,
            embeddings_dir=embedding_dir,
            device=device,
        )

    @staticmethod
    def _compute_embeddings(
        documents: list[Document],
        split_str: str = "\n\n",
        max_chunks: int = 64,
        model_name: str = MODEL_NAME,
        device: torch.device = DEVICE,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_ids, all_chunks = [], []
        for doc in documents:
            if len(chunks := doc.text.split(split_str)) > max_chunks:
                chunks = random.sample(chunks, max_chunks)
            all_ids.extend([doc.doc_id] * len(chunks))
            all_chunks.extend(chunks)

        model = SentenceTransformer(model_name).to(device)
        embeddings = model.encode(all_chunks)

        return np.array(all_ids), np.array(embeddings)

    def _load_embeddings(
        self,
        documents: list[Document],
        model_name: str = MODEL_NAME,
        embeddings_dir: Path = EMBEDDINGS_DIR,
        device: torch.device = DEVICE,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.check_corpus_hash(documents):
            logging.warning("Corpus hash mismatch. Recomputing embeddings.")
            return self._compute_embeddings(
                documents=documents,
                model_name=model_name,
                device=device,
            )

        ids_file = embeddings_dir / "ids.npy"
        embeddings_file = embeddings_dir / f"{model_name}.npy"

        if ids_file.exists() and embeddings_file.exists():
            return np.load(ids_file), np.load(embeddings_file)

        logging.warning("Embeddings not found. Recomputing.")
        ids, embeddings = self._compute_embeddings(
            documents=documents,
            model_name=model_name,
            device=device,
        )

        np.save(ids_file, ids)
        np.save(embeddings_file, embeddings)

        return ids, embeddings

    def score(self, query: str) -> list[RetrievalScore]:
        query_terms = tokenize(query, remove_tubingen=True) + ["tübingen"]
        query_terms = list(set(query_terms + [" ".join(query_terms)]))

        query_embeddings = self.model.encode(query_terms)
        sim_df = pd.DataFrame(
            cosine_similarity(self.embeddings, query_embeddings),
            index=self.ids,
            columns=query_terms,
        )
        sim_df_agg = sim_df.groupby(sim_df.index).apply(lambda x: x.max(axis=0).mean())
        return [
            RetrievalScore(doc_id=str(doc_id), score=float(score), ranker="sim")
            for doc_id, score in sim_df_agg.items()
        ]
