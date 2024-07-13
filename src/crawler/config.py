from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from httpx import URL

HARD_DRIVE = Path("D://")
assert HARD_DRIVE.exists(), "Please connect the hard drive used for the project."

BASE_DIR = HARD_DRIVE / "mse"
HTML_DIR = BASE_DIR / "html"
TXT_DIR = BASE_DIR / "txt"

HTML_DIR.mkdir(parents=True, exist_ok=True)
TXT_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def get_seed_urls() -> set[URL]:
    return set(
        URL(url)
        for url in pd.read_json(
            BASE_DIR / "query_results.jsonl", lines=True
        ).links.explode()
    )


def get_allowed_domains() -> set[str]:
    return set(url.host for url in get_seed_urls())


def get_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT}


@dataclass(frozen=True)
class CrawlerConfig:
    html_dir: Path = HTML_DIR
    ids_dir: Path = BASE_DIR
    timeout: float = 10
    max_depth: int = 10

    allowed_domains: set[str] = field(default_factory=get_allowed_domains)
    seed_urls: set[URL] = field(default_factory=get_seed_urls)
    headers: dict[str, str] = field(default_factory=get_headers)
