import asyncio
import hashlib
import json
import os
from asyncio import sleep
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple, TypedDict

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from cachetools import TTLCache
from httpx import AsyncClient, Response, URL
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

SEED_URLS = [
    URL("https://www.tuebingen.de/"),
    URL("https://www.tuebingen.de/en/"),
    URL("https://www.tuebingen-info.de/"),
    URL("https://www.tuepedia.de/"),
]


@dataclass(frozen=True)
class CrawlerConfig:
    seed_urls = SEED_URLS
    allowed_domains = []
    html_dir = HTML_DIR
    ids_dir = BASE_DIR
    index_dir = INDEX_DIR
    max_depth = 5
    headers = {"User-Agent": USER_AGENT}
    max_docs = 1e5
    sleep_time = 2


class ScrapingRequest(NamedTuple):
    doc_id: str
    url: URL
    depth: int


class Index(TypedDict):
    url: str
    text: str


class Crawler:
    def __init__(self, run_id: str = "", config: CrawlerConfig = None):
        self.run_id: str = run_id or pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        self.config: CrawlerConfig = config or CrawlerConfig()
        self.frontier: pd.DataFrame = self.load_frontier()
        self.recently_visited: TTLCache = TTLCache(maxsize=100, ttl=self.config.sleep_time)
        self._client: AsyncClient or None = None

    @property
    def client(self) -> AsyncClient:
        """
        Returns the HTTP client with the headers set
        """
        if self._client is None:
            self._client = AsyncClient(headers=self.config.headers)
        return self._client

    @staticmethod
    def is_url_valid(url: URL) -> bool:
        """
        Checks if the URL is valid by checking if it has a scheme and netloc
        and if it does not have an extension (e.g., .jpg, .png, .pdf)
        """
        _, ext = os.path.splitext(url.path)
        return bool(url.scheme) and bool(url.netloc) and not ext

    @staticmethod
    def extract_text_from_soup(soup: BeautifulSoup) -> str:
        """
        Extracts the text from the soup object by joining all paragraphs
        """
        paragraphs = (p.get_text().strip() for p in soup.find_all("p"))
        return " \n ".join(p for p in paragraphs if p)

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential())
    async def _get(self, url: URL) -> Response:
        """
        Fetches the URL and raises an exception if the status code is not 2xx
        We use the tenacity library to retry the request 3 times with exponential backoff
        We use the TTL cache to avoid visiting the same domain too often
        """
        while self.recently_visited.get(url.netloc):
            await sleep(0.1)

        response = await self.client.get(url)
        response.raise_for_status()
        return response

    def load_frontier(self) -> pd.DataFrame:
        """
        Loads the frontier from the CSV file if it exists, otherwise
        it converts the seed URLs to the frontier format and returns it
        """
        fp = self.config.ids_dir / f"{self.run_id}.csv"
        if fp.exists():
            return pd.read_csv(fp).set_index("doc_id")
        else:
            return pd.DataFrame(self.convert_seed_to_frontier()).set_index("doc_id")

    def convert_seed_to_frontier(self) -> Iterator[dict[str, any]]:
        """
        Converts the seed URLs to the frontier format
        """
        for url in self.config.seed_urls:
            yield {
                "doc_id": hashlib.md5(str(url).encode()).hexdigest(),
                "url": url,
                "scheme": url.scheme,
                "netloc": url.netloc,
                "path": url.path,
                "query": url.query,
                "fragment": url.fragment,
                "depth": 0,
                "status": "pending",
                "created": pd.Timestamp.now(),
            }

    def count_docs(self) -> int:
        """
        Counts the number of documents that are completed
        """
        return (self.frontier["status"] == "completed").sum()

    def add_to_frontier(self, url: URL, depth: int) -> None:
        """
        Adds the URL to the frontier if it does not exist already
        """
        if (
                doc_id := hashlib.md5(str(url).encode()).hexdigest()
        ) in self.frontier.index:
            return None

        self.frontier.loc[doc_id] = {
            "url": str(url),
            "scheme": url.scheme,
            "netloc": url.netloc,
            "path": url.path,
            "query": url.query,
            "fragment": url.fragment,
            "depth": depth,
            "status": "pending",
            "created": pd.Timestamp.now(),
        }

    def fetch_next_doc_batch(self, limit: int = 10) -> list[ScrapingRequest]:
        """
        Fetches the next batch of documents to be scraped
        1. Query the frontier for pending documents that are not yet visited
        2. Filter the documents by conditions (e.g., depth < max_depth and allowed_domains)
        3. Sort the documents by depth and created date
        4. Group the documents by netloc
        5. Prioritize the documents that are recently visited
        6. Return a batch of documents to be scraped (up to the limit, all from different domains)
        """
        query = "status == 'pending' and depth < @self.config.max_depth"
        if self.config.allowed_domains:
            query += " and netloc in @self.config.allowed_domains"
        pending_docs = self.frontier.query(query)

        pending_docs = pending_docs.sort_values(
            ["depth", "created"], ascending=[True, True]
        )
        pending_docs = pending_docs.groupby("netloc").first()

        pending_docs["recently_visited"] = pending_docs["netloc"].map(
            lambda x: x in self.recently_visited.keys()
        )
        pending_docs = pending_docs.sort_values(["recently_visited"], ascending=[True])
        return [
            ScrapingRequest(
                doc_id=str(doc_id),
                url=URL(row["url"]),
                depth=row["depth"],
            )
            for doc_id, row in pending_docs.head(limit).iterrows()
        ]

    def extract_links(self, soup: BeautifulSoup, base_url: URL) -> Iterator[URL]:
        for link_element in soup.find_all("a", href=True):
            url = link_element["href"]
            if url := self.extract_url(url, base_url):
                yield url

    def extract_url(self, url: str, base_url: URL) -> URL or None:
        try:
            url = URL(url)
        except httpx.InvalidURL:
            return None

        if url.is_relative_url:
            url = base_url.join(url)

        return url if self.is_url_valid(url) else None

    def mark_status(self, doc_id: str, status: str) -> None:
        self.frontier.loc[doc_id, "status"] = status

    def save_data(self, doc_id: str, content: str, index: dict[str, any]) -> None:
        with open(self.config.html_dir / f"{doc_id}.html", "w", encoding="utf-8") as f:
            f.write(content)

        with open(self.config.index_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(index, f)

    def save_frontier(self) -> None:
        self.frontier.to_csv(self.config.ids_dir / f"{self.run_id}.csv")

    def create_index(self, response: Response, soup: BeautifulSoup) -> Index:
        text = self.extract_text_from_soup(soup)
        return {"url": str(response.url), "text": text}

    async def crawl(self, req: ScrapingRequest) -> None:
        """
        Crawls the URL and extracts the text and links
        Creates an index and saves the HTML and index files
        Marks the status as completed if successful, otherwise as failed
        Saves new URLs to the frontier
        """
        try:
            response = await self._get(req.url)
        except RetryError:
            self.mark_status(req.doc_id, "failed")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        index = self.create_index(response, soup)
        self.save_data(req.doc_id, response.text, index)
        self.mark_status(req.doc_id, "completed")

        for url in self.extract_links(soup, response.url):
            self.add_to_frontier(url, req.depth + 1)

    async def close(self) -> None:
        """
        Closes the HTTP client
        """
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _run(self) -> None:
        """
        Main function to run the crawler
        """
        pbar = tqdm(total=self.config.max_docs - self.count_docs())
        while (
                req := self.fetch_next_doc_batch()
        ) and self.count_docs() < self.config.max_docs:
            await asyncio.gather(*(self.crawl(r) for r in req))
            self.save_frontier()
            pbar.update(len(req))

        pbar.close()

    async def run(self) -> None:
        """
        Wrapper function to run the crawler
        """
        try:
            await self._run()
        finally:
            await self.close()


if __name__ == "__main__":
    c = Crawler()
    asyncio.run(c.run())
