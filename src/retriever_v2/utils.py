import re
from itertools import islice
from pathlib import Path

import torch
from nltk.corpus import stopwords

INDEX_DIR = Path(__file__).parent / "index"
EMBEDDINGS_DIR = INDEX_DIR / "embeddings"
PICKLES_DIR = INDEX_DIR / "pickles"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STOPWORDS = set(stopwords.words("english"))
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >= 1")

    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch
