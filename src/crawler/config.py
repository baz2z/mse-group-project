from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from httpx import URL

HARD_DRIVE = Path("D://")
assert HARD_DRIVE.exists(), "Please connect the hard drive used for the project."

BASE_DIR = HARD_DRIVE / "mse"
HTML_DIR = BASE_DIR / "html"
HTML_DIR.mkdir(parents=True, exist_ok=True)

SEED_URLS = (
    pd.read_json(BASE_DIR / "queries.jsonl", lines=True)
    .links
    .explode()
    .apply(URL)
    .tolist()
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class CrawlerConfig:
    html_dir: Path = HTML_DIR
    ids_dir: Path = BASE_DIR
    max_docs: int = 100_000
    max_depth: int = 5
    sleep_time: float = 2
    timeout: float = 10

    seed_urls: list[URL] = field(default_factory=lambda: SEED_URLS)
    headers: dict[str, str] = field(default_factory=lambda: {"User-Agent": USER_AGENT})
