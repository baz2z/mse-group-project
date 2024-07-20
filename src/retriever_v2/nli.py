import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification as Model,
    AutoTokenizer as Tokenizer,
)

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import batched, DEVICE, slice_string, tokenize

MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


class NLIRetriever(BaseRetriever):
    def __init__(
        self,
        documents: list[Document],
        model_name: str = MODEL_NAME,
        device: torch.device = DEVICE,
    ):
        self.docs = documents
        self.tokenizer = Tokenizer.from_pretrained(model_name)
        self.model = Model.from_pretrained(model_name).half().to(device)

    def score(self, query: str, filter_ids: set[str] = None) -> list[RetrievalScore]:
        query = "This text is about {}.".format(
            " ".join(tokenize(query, remove_tubingen=True) + ["tübingen"])
        )
        documents = [
            doc for doc in self.docs if filter_ids is None or doc.doc_id in filter_ids
        ]

        batches, ids = [], []
        for doc in documents:
            for chunk in slice_string(doc.text[:16_384], 2_048):
                if not chunk:
                    continue

                ids.append(doc.doc_id)
                batches.append(chunk)

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

        scores = pd.Series(scores, index=ids)
        return [
            RetrievalScore(doc_id=str(doc_id), score=float(score), ranker="nli")
            for doc_id, score in scores.groupby(scores.index).max().items()
        ]
