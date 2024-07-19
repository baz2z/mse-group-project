import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification as Model,
    AutoTokenizer as Tokenizer,
)

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import batched, DEVICE, tokenize

MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"


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
        docs = sorted(
            (
                doc
                for doc in self.docs
                if filter_ids is None or doc.doc_id in filter_ids
            ),
            key=lambda doc: len(doc.text),
            reverse=True,
        )
        scores = []
        with torch.no_grad():
            for batch in tqdm(batched(docs, 8)):
                enc = self.tokenizer(
                    [doc.text for doc in batch],
                    [query] * len(batch),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                logits = self.model(**enc.to(DEVICE)).logits.cpu()
                scores.extend(
                    RetrievalScore(
                        doc_id=doc.doc_id, score=logit[0].item(), ranker="nli"
                    )
                    for doc, logit in zip(batch, logits)
                )

        return scores
