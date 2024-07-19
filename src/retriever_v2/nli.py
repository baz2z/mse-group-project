import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification as AutoModel, AutoTokenizer

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import batched, DEVICE

MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


class NLIRetriever(BaseRetriever):
    def __init__(
        self,
        documents: list[Document],
        model_name: str = MODEL_NAME,
        device: torch.device = DEVICE,
    ):
        self.docs = documents
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def score(self, query: str, filter_ids: set[str] = None) -> list[RetrievalScore]:
        query_str = f"This text is about {query.lower()}"
        docs = [
            doc for doc in self.docs if filter_ids is None or doc.doc_id in filter_ids
        ]

        scores = []
        with torch.no_grad():
            for batch in tqdm(batched(docs, 8)):
                enc = self.tokenizer(
                    [doc.text for doc in batch],
                    [query_str] * len(batch),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                logits = self.model(**enc.to(DEVICE)).logits.cpu()

                for doc, logit in zip(batch, logits):
                    scores.append(
                        RetrievalScore(
                            doc_id=doc.doc_id, score=logit[0].item(), ranker="nli"
                        )
                    )

        return scores
