import logging
from pathlib import Path
from random import sample, seed

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import DEVICE, EMBEDDINGS_DIR, tokenize

MODEL_NAME = "all-MiniLM-L12-v2"
MAX_CHUNKS = 64


class SimRetriever(BaseRetriever):
    """
    Similarity retriever that uses a pretrained sentence transformer model.
    """

    def __init__(
        self,
        documents: list[Document],
        model_name: str = MODEL_NAME,
        max_chunks: int = MAX_CHUNKS,
        embedding_dir: Path = EMBEDDINGS_DIR,
        device: torch.device = DEVICE,
    ):
        """
        Initialize the similarity retriever.

        Args:
            documents: List of Document objects
            model_name: Pretrained model name
            max_chunks: Maximum number of chunks to consider
            embedding_dir: Directory to save/load embeddings
            device: Device to run the model on

        Returns:
            None
        """
        # load pretrained sentence transformer model
        self.model = SentenceTransformer(model_name).to(device)

        # load or compute embeddings for each document paragraph
        self.ids, self.embeddings = self._load_embeddings(
            documents=documents,
            model_name=model_name,
            max_chunks=max_chunks,
            embeddings_dir=embedding_dir,
        )

    def _load_embeddings(
        self,
        documents: list[Document],
        model_name: str,
        max_chunks: int,
        embeddings_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load embeddings from disk or compute them.

        Args:
            documents: List of Document objects
            model_name: Pretrained model name
            max_chunks: Maximum number of chunks to consider
            embeddings_dir: Directory to save/load embeddings

        Returns:
            Tuple of document IDs and embeddings.
        """
        # check if the corpus is the main corpus
        if not self.check_corpus_hash(documents):
            logging.warning("Corpus hash mismatch. Recomputing embeddings.")
            return self._compute_embeddings(documents=documents, max_chunks=max_chunks)

        # check if embeddings are already computed
        ids_file = embeddings_dir / "ids.npy"
        embeddings_file = embeddings_dir / f"{model_name}.npy"

        if ids_file.exists() and embeddings_file.exists():
            return np.load(ids_file), np.load(embeddings_file)

        # if embeddings are not found, recompute them
        logging.warning("Embeddings not found. Recomputing.")
        ids, embeddings = self._compute_embeddings(
            documents=documents, max_chunks=max_chunks
        )

        # save the embeddings to disk
        np.save(ids_file, ids)
        np.save(embeddings_file, embeddings)

        return ids, embeddings

    def _compute_embeddings(
        self,
        documents: list[Document],
        max_chunks: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute embeddings for each document paragraph.

        Args:
            documents: List of Document objects.
            max_chunks: Maximum number of chunks to consider.

        Returns:
            Tuple of document IDs and embeddings.
        """
        all_ids, all_chunks = [], []
        for doc in documents:
            # split the documents into paragraphs
            if len(chunks := doc.text.split("\n\n")) > max_chunks:
                # if we have more than max_chunks, sample max_chunks
                seed(0)
                sample_chunks = sample(range(len(chunks)), max_chunks)
                chunks = [chunks[i] for i in sorted(sample_chunks)]

            all_ids.extend([doc.doc_id] * len(chunks))
            all_chunks.extend(chunks)

        # encode the chunks using the pretrained model
        embeddings = self.model.encode(all_chunks)

        return np.array(all_ids), np.array(embeddings)

    def score(self, query: str) -> list[RetrievalScore]:
        """
        Score documents based on similarity for a given query.

        Args:
            query: Query string, e.g., "tübingen football club"

        Returns:
            List of RetrievalScore objects.
        """
        # tokenizing the query and adding "tübingen" to it
        query_terms = tokenize(query, remove_tubingen=True) + ["tübingen"]
        query_terms = list(set(query_terms + [" ".join(query_terms)]))

        # encoding the query using the pretrained model
        query_embeddings = self.model.encode(query_terms)

        # computing cosine similarity between the query and each paragraph embedding
        sim_df = pd.DataFrame(
            cosine_similarity(self.embeddings, query_embeddings),
            index=self.ids,
            columns=query_terms,
        )

        # aggregating the similarity scores, taking the maximum score for each document
        # and then averaging the scores across all query terms
        sim_df_agg = sim_df.groupby(sim_df.index).apply(lambda x: x.max(axis=0).mean())
        return [
            RetrievalScore(doc_id=str(doc_id), score=float(score), ranker="sim")
            for doc_id, score in sim_df_agg.items()
        ]
