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

        try:
            self.embeddings = np.load(embedding_dir / f"{model_name}.npy")
            self.ids = np.load(embedding_dir / "ids.npy")
        except FileNotFoundError:
            self.embeddings, self.ids = self.pre_compute_embeddings(
                documents=documents,
                model_name=model_name,
                embeddings_dir=embedding_dir,
                device=device,
            )

    def score(self, query: str) -> list[RetrievalScore]:
        query_terms = tokenize(query)
        query_terms = list(set(query_terms + " ".join(query_terms)))

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

    @staticmethod
    def pre_compute_embeddings(
        documents: list[Document],
        split_str: str = "\n\n",
        max_chunks: int = 64,
        model_name: str = MODEL_NAME,
        embeddings_dir: Path = EMBEDDINGS_DIR,
        device: torch.device = DEVICE,
    ):
        all_ids, all_chunks = [], []
        for doc in documents:
            if len(chunks := doc.text.split(split_str)) > max_chunks:
                chunks = random.sample(chunks, max_chunks)
            all_ids.extend([doc.doc_id] * len(chunks))
            all_chunks.extend(chunks)

        model = SentenceTransformer(model_name).to(device)
        embeddings = model.encode(all_chunks)

        all_ids = np.array(all_ids)
        np.save(embeddings_dir / "ids.npy", all_ids)
        np.save(embeddings_dir / model_name, embeddings)

        return embeddings, all_ids
