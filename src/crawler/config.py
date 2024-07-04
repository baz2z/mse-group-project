import re
from dataclasses import dataclass
from pathlib import Path

from httpx import URL

BASE_DIR = Path(__file__).parent.parent.parent / "dat" / "crawler"
HTML_DIR = BASE_DIR / "html"
INDEX_DIR = BASE_DIR / "index"

HTML_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

SEED_URLS = [
    URL("https://www.tuebingen.de/"),
    URL("https://www.tuebingen.de/en/"),
    URL("https://www.tuebingen-info.de/"),
    URL("https://www.tuepedia.de/"),
    URL("https://en.wikipedia.org/wiki/T%C3%BCbingen"),
    URL("https://de.wikipedia.org/wiki/T%C3%BCbingen"),
    URL("https://uni-tuebingen.de/en/"),
]

ALLOWED_DOMAINS_REGEX = [
    re.compile(r"\S+\.(de|com|org|net)$"),
]

FORBIDDEN_DOMAINS_REGEX = [
    re.compile(r"^(?!de\.)(?!en\.)\S+\.wikipedia\.org$"),
    re.compile(r"^(?!de\.)(?!en\.)\S+\.wikipedia\.org$"),
    re.compile(r"^(?!de\.)(?!en\.)\S+\.wikipedia\.org$"),
]


@dataclass(frozen=True)
class CrawlerConfig:
    seed_urls = SEED_URLS
    html_dir = HTML_DIR
    ids_dir = BASE_DIR
    index_dir = INDEX_DIR
    max_depth = 5
    headers = {"User-Agent": USER_AGENT}
    max_docs = 100
    sleep_time = 2
    allowed_domains_pattern = ALLOWED_DOMAINS_REGEX
    forbidden_domains_pattern = FORBIDDEN_DOMAINS_REGEX
