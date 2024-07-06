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

SEED_URLS = {
    URL("https://www.tuebingen.de/"),
    URL("https://www.tuebingen.de/en/"),
    URL("https://www.tuebingen-info.de/"),
    URL("https://www.tuepedia.de/"),
    URL("https://www.swabianalb.info/cities/tuebingen"),
    URL("https://www.tuemarkt.de/"),
    URL("https://uni-tuebingen.de/en/"),
    URL("https://en.wikipedia.org/wiki/T%C3%BCbingen"),
    URL("https://de.wikipedia.org/wiki/T%C3%BCbingen"),
}

ALLOWED_DOMAINS_REGEX = [
    # the domain must start with www, de, or en and end with com, de, org, or net
    # examples of valid domains: www.example.com, de.example.de, en.example.org, example.net
    re.compile(
        r"^(www|de|en)?\.?([a-zA-Z0-9-]+)\.(com|de|org|net|info)$", re.IGNORECASE
    ),
]

FORBIDDEN_DOMAINS_REGEX = []


@dataclass(frozen=True)
class CrawlerConfig:
    seed_urls = SEED_URLS
    html_dir = HTML_DIR
    ids_dir = BASE_DIR
    index_dir = INDEX_DIR
    max_depth = 5
    headers = {"User-Agent": USER_AGENT}
    max_docs = 1e5
    sleep_time = 1
    timeout = 10
    allowed_domains_pattern = ALLOWED_DOMAINS_REGEX
    forbidden_domains_pattern = FORBIDDEN_DOMAINS_REGEX
