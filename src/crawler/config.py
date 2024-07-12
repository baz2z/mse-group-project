from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from httpx import URL

HARD_DRIVE = Path("D://")
assert HARD_DRIVE.exists(), "Please connect the hard drive used for the project."

BASE_DIR = HARD_DRIVE / "mse"
HTML_DIR = BASE_DIR / "html"
MD_DIR = BASE_DIR / "md"

HTML_DIR.mkdir(parents=True, exist_ok=True)
MD_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def get_seed_urls() -> list[URL]:
    return list(
        pd.read_json(BASE_DIR / "query_results.jsonl", lines=True)
        .links
        .explode()
        .apply(URL)
        .dropna()
        .unique()
    )


def get_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT}


@dataclass(frozen=True)
class CrawlerConfig:
    html_dir: Path = HTML_DIR
    ids_dir: Path = BASE_DIR
    sleep_time: float = 2
    timeout: float = 10

    seed_urls: list[URL] = field(default_factory=get_seed_urls)
    headers: dict[str, str] = field(default_factory=get_headers)
