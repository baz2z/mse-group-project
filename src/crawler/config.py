from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from httpx import URL

HARD_DRIVE = Path("D://")
assert HARD_DRIVE.exists(), "Please connect the hard drive used for the project."

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def get_seed_urls() -> set[URL]:
    return set(
        URL(url)
        for url in pd.read_json(
            HARD_DRIVE / "query_results_v2.jsonl", lines=True
        ).links.explode()
    )


@dataclass(frozen=True)
class CrawlerConfig:
    DIR: Path = HARD_DRIVE
    batch_size: int = 256
    max_depth: int = 10
    timeout: float = 10
    seed_urls: set[URL] = field(default_factory=get_seed_urls)
