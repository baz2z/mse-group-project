import re
from pathlib import Path

import torch
from nltk.corpus import stopwords

INDEX_DIR = Path(__file__).parent / "index"
EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STOPWORDS = set(stopwords.words("english"))
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
