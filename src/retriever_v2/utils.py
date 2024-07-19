import re
from itertools import islice
from pathlib import Path

import torch
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords

INDEX_DIR = Path(__file__).parent / "index"
EMBEDDINGS_DIR = INDEX_DIR / "embeddings"
PICKLES_DIR = INDEX_DIR / "pickles"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TUBINGEN_PATTERN = re.compile(r"t(ü|ue|u)binge([nr])", re.IGNORECASE)
STOPWORDS = set(stopwords.words("english"))
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")

STEMMER = SnowballStemmer("english")


def tokenize(text: str):
    return [
        word.lower()
        for word in word_tokenize(text)
        if (
            word.lower() not in STOPWORDS
            and TOKEN_PATTERN.match(word)
            and not TUBINGEN_PATTERN.search(word)
        )
    ]


def stem_tokenize(text: str):
    return [STEMMER.stem(word) for word in tokenize(text)]


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >= 1")

    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch
