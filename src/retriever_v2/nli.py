import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification as Model,
    AutoTokenizer as Tokenizer,
)

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import batched, DEVICE, slice_string, tokenize

MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
MODEL_NAME_FAST = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"


class NLIRetriever(BaseRetriever):
    """
    NLI retriever that uses a pretrained NLI model.
    """

    def __init__(
        self,
        documents: list[Document],
        model_name: str = MODEL_NAME,
        device: torch.device = DEVICE,
        use_fast: bool = False,
    ):
        """
        Initialize the NLI retriever.

        Args:
            documents: List of Document objects.
            model_name: Pretrained model name
            device: Device to run the model on

        Returns:
            None
        """
        if use_fast:
            model_name = MODEL_NAME_FAST

        self.docs = documents

        # loading the pretrained model and tokenizer
        self.tokenizer = Tokenizer.from_pretrained(model_name)
        self.model = Model.from_pretrained(model_name).half().to(device)

    def score(self, query: str, filter_ids: set[str] = None) -> list[RetrievalScore]:
        """
        Score documents based on NLI for a given query.

        Args:
            query: Query string, e.g., "tübingen football club"
            filter_ids: Set of document IDs to filter, e.g., {"12345", "67890"}

        Returns:
            List of RetrievalScore objects.
        """
        # tokenizing the query and adding "tübingen" to it
        query = "This text is about {}.".format(
            " ".join(tokenize(query, remove_tubingen=True) + ["tübingen"])
        )
        # filtering documents based on the provided IDs
        documents = [
            doc for doc in self.docs if filter_ids is None or doc.doc_id in filter_ids
        ]

        # splitting documents into chunks of 2048 characters (~512 tokens)
        # cutting off the text at 16384 characters (95th percentile of the dataset)
        batches, ids = [], []
        for doc in documents:
            for chunk in slice_string(doc.text[:16_384], 2_048):
                if not chunk:
                    continue

                ids.append(doc.doc_id)
                batches.append(chunk)

        # computing NLI scores for each chunk in batches
        scores = []
        with torch.no_grad():
            for batch in tqdm(batched(batches, 8), desc="Computing NLI scores"):
                enc = self.tokenizer(
                    batch,
                    [query] * len(batch),
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                logits = self.model(**enc.to(DEVICE)).logits.cpu()
                scores.extend(logit[0].item() for logit in logits)

        # return the maximum score for each document
        scores = pd.Series(scores, index=ids)
        scores = scores.groupby(scores.index).apply(lambda x: (x.max(), x.values))
        return [
            RetrievalScore(
                doc_id=str(doc_id),
                score=float(score),
                dist=np.array(dist, dtype=np.float16),
                ranker="nli",
            )
            for doc_id, (score, dist) in scores.items()
        ]
