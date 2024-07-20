from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from httpx import URL

HARD_DRIVE = Path("D://")
assert HARD_DRIVE.exists(), "Please connect the hard drive used for the project."


def get_seed_urls() -> set[URL]:
    """
    Get the seed URLs from the query results.

    Returns:
        Set of seed URLs.
    """
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
    max_filesize: int = 10_000_000
    seed_urls: set[URL] = field(default_factory=get_seed_urls)
