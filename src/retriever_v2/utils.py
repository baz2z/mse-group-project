import re
from itertools import islice
from pathlib import Path

import torch
from nltk import word_tokenize
from nltk.corpus import stopwords

INDEX_DIR = Path(__file__).parent / "index"
EMBEDDINGS_DIR = INDEX_DIR / "embeddings"
PICKLES_DIR = INDEX_DIR / "pickles"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TUBINGEN_PATTERN = re.compile(r"\bt(ü|ue|u)binge([nr])\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")

STOPWORDS = set(stopwords.words("english"))


def tokenize(text: str, remove_tubingen: bool):
    if remove_tubingen:
        text = TUBINGEN_PATTERN.sub("", text)

    return [
        word.lower()
        for word in word_tokenize(text)
        if (word.lower() not in STOPWORDS and TOKEN_PATTERN.match(word))
    ]


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >= 1")

    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def slice_string(text, max_len):
    if max_len < 1:
        raise ValueError("n must be >= 1")

    for i in range(0, len(text), max_len):
        if i + max_len < len(text):
            yield text[i : i + max_len].strip()
        else:
            yield text[-max_len:].strip()
