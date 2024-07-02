import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import NamedTuple, TypedDict
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Response
from tenacity import retry, RetryError, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

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


@dataclass(frozen=True)
class CrawlerConfig:
    seed_urls = [
        "https://www.tuebingen.de/",
        "https://www.tuebingen.de/en/",
        "https://www.tuebingen-info.de/",
        "https://www.tuepedia.de/"
    ]
    allowed_domains = [
        "www.tuebingen.de",
        "www.tuebingen-info.de",
        "www.tuepedia.de"
    ]
    html_dir = HTML_DIR
    ids_dir = BASE_DIR
    index_dir = INDEX_DIR
    max_depth = 5
    headers = {"User-Agent": USER_AGENT}
    max_docs = 1000
    sleep_time = 2


class ScrapingRequest(NamedTuple):
    doc_id: str
    url: str
    netloc: str
    depth: int


class Index(TypedDict):
    text: str


class Crawler:
    def __init__(self, run_id: str = "", config: CrawlerConfig = None):
        self.run_id = run_id or pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        self.config = config or CrawlerConfig()
        self.frontier = self.load_frontier()

    @staticmethod
    def is_url_valid(url: str) -> bool:
        parsed = urlparse(url)
        _, ext = os.path.splitext(parsed.path)
        return bool(parsed.scheme) and bool(parsed.netloc) and not ext

    @staticmethod
    def extract_text_from_soup(soup: BeautifulSoup) -> str:
        paragraphs = (p.get_text().strip() for p in soup.find_all("p"))
        return " \n ".join(p for p in paragraphs if p)

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10))
    def _get(self, url: str) -> Response:
        response = requests.get(url, headers=self.config.headers)
        response.raise_for_status()
        return response

    def load_frontier(self) -> pd.DataFrame:
        fp = self.config.ids_dir / f"{self.run_id}.csv"
        if fp.exists():
            return pd.read_csv(fp).set_index("doc_id")
        else:
            return pd.DataFrame(self.convert_seed_to_frontier()).set_index("doc_id")

    def convert_seed_to_frontier(self):
        for url in self.config.seed_urls:
            parse = urlparse(url)
            yield {
                "doc_id": hashlib.md5(url.encode()).hexdigest(),
                "url": url,
                "scheme": parse.scheme,
                "netloc": parse.netloc,
                "path": parse.path,
                "query": parse.query,
                "fragment": parse.fragment,
                "depth": 0,
                "status": "pending",
                "created": pd.Timestamp.now(),
            }

    def count_docs(self) -> int:
        return (self.frontier["status"] == "completed").sum()

    def add_to_frontier(self, url: str, depth: int):
        if (doc_id := hashlib.md5(url.encode()).hexdigest()) in self.frontier.index:
            return None

        parse = urlparse(url)
        self.frontier.loc[doc_id] = {
            "url": url,
            "scheme": parse.scheme,
            "netloc": parse.netloc,
            "path": parse.path,
            "query": parse.query,
            "fragment": parse.fragment,
            "depth": depth,
            "status": "pending",
            "created": pd.Timestamp.now(),
        }

    def fetch_next_doc(self) -> ScrapingRequest or None:
        res = self.frontier.query(
            "status == 'pending' "
            "and depth < @self.config.max_depth "
            "and netloc in @self.config.allowed_domains"
        )

        if res.empty:
            return None

        return ScrapingRequest(
            doc_id=res.index[0],
            url=res.iloc[0]["url"],
            netloc=res.iloc[0]["netloc"],
            depth=res.iloc[0]["depth"],
        )

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        for link_element in soup.find_all("a", href=True):
            url = link_element["href"]
            if url := self.extract_url(url, base_url):
                yield url

    def extract_url(self, url: str, base_url: str) -> str or None:
        if self.is_url_valid(url):
            return url

        url = urljoin(base_url, url)
        if self.is_url_valid(url):
            return url

        return None

    def mark_status(self, doc_id: str, status: str):
        self.frontier.loc[doc_id, "status"] = status

    def save_data(self, doc_id: str, content: str, index: dict[str, any]):
        with open(self.config.html_dir / f"{doc_id}.html", "w", encoding="utf-8") as f:
            f.write(content)

        with open(self.config.index_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(index, f)

    def save_frontier(self):
        self.frontier.to_csv(self.config.ids_dir / f"{self.run_id}.csv")

    def create_index(self, soup: BeautifulSoup) -> Index:
        text = self.extract_text_from_soup(soup)
        return {"text": text}

    def crawl(self, req: ScrapingRequest):
        try:
            response = self._get(req.url)
        except RetryError:
            self.mark_status(req.doc_id, "failed")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        index = self.create_index(soup)
        self.save_data(req.doc_id, response.text, index)
        self.mark_status(req.doc_id, "completed")

        for url in self.extract_links(soup, response.url):
            self.add_to_frontier(url, req.depth + 1)

    def run(self):
        pbar = tqdm(total=self.config.max_docs - self.count_docs())
        while (req := self.fetch_next_doc()) and self.count_docs() < self.config.max_docs:
            sleep(random.uniform(0, self.config.sleep_time))
            self.crawl(req)
            self.save_frontier()
            pbar.update(1)

        pbar.close()


if __name__ == "__main__":
    c = Crawler()
    c.run()
