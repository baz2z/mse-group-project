import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import DEVICE, EMBEDDINGS_DIR, STOPWORDS, TOKEN_PATTERN

MODEL_NAME = "all-MiniLM-L12-v2"


def tokenize(text: str):
    return [
        word.lower()
        for word in word_tokenize(text)
        if word.lower() not in STOPWORDS and TOKEN_PATTERN.match(word)
    ]


class SimRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        embedding_dir: Path = EMBEDDINGS_DIR,
        device: torch.device = DEVICE,
    ):
        self.model = SentenceTransformer(model_name).to(device)
        self.embeddings = np.load(embedding_dir / f"{model_name}.npy")
        self.ids = np.load(embedding_dir / "ids.npy")

    def score(self, query: str) -> list[RetrievalScore]:
        query_terms = list({" ".join(query.split()), *tokenize(query)})
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

    @classmethod
    def pre_compute_embeddings(
        cls,
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
        np.save(embeddings_dir / "ids.npy", np.array(all_ids))
        np.save(embeddings_dir / model_name, embeddings)
