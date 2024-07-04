import asyncio
import hashlib
import json
import os
from asyncio import sleep
from typing import Iterator, Literal

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from cachetools import TTLCache
from httpx import AsyncClient, Response, URL
from tenacity import RetryError
from tqdm import tqdm

from crawler.config import CrawlerConfig
from crawler.types import Index, ScrapingRequest
from get_logger import get_logger
from retry_wrapper import get_retry_wrapper

logger = get_logger("crawler")
retry = get_retry_wrapper(logger)


class Crawler:
    def __init__(self, run_id: str = "", config: CrawlerConfig = None):
        self.run_id: str = run_id or pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        self.config: CrawlerConfig = config or CrawlerConfig()
        self.frontier: pd.DataFrame = self.load_frontier()
        self.recently_visited: TTLCache = TTLCache(
            maxsize=100, ttl=self.config.sleep_time
        )
        self._client: AsyncClient or None = None

    @property
    def client(self) -> AsyncClient:
        """
        Returns the HTTP client with the headers set
        """
        if self._client is None:
            logger.info("Creating a new HTTP client")
            self._client = AsyncClient(headers=self.config.headers)
        return self._client

    def check_allowed_domains(self, domain: bytes | str) -> bool:
        if not self.config.allowed_domains_pattern:
            return True

        for pattern in self.config.allowed_domains_pattern:
            if pattern.match(domain):
                return True

        return False

    def check_denied_domains(self, domain: bytes | str) -> bool:
        for pattern in self.config.forbidden_domains_pattern:
            if pattern.match(domain):
                return False

        return True

    @staticmethod
    def check_url_has_extension(url: URL) -> bool:
        for part in (url.path, url.query, url.params, url.fragment):
            if os.path.splitext(part)[1]:
                return True
        return False

    def is_url_valid(self, url: URL) -> bool:
        """
        Checks if the URL is valid by checking if it has a scheme and netloc
        and if it does not have an extension (e.g., .jpg, .png, .pdf)
        """
        if not (url.scheme and url.netloc):
            logger.info(f"URL {url} is not valid")
            return False

        if self.check_url_has_extension(url):
            logger.info(f"URL {url} has an extension")
            return False

        if not self.check_denied_domains(url.netloc):
            logger.info(f"URL {url} is denied")
            return False

        if not self.check_allowed_domains(url.netloc):
            logger.info(f"URL {url} is not allowed")
            return False

        return True

    @staticmethod
    def extract_text_from_soup(soup: BeautifulSoup) -> str:
        """
        Extracts the text from the soup object by joining all paragraphs
        """
        paragraphs = (p.get_text().strip() for p in soup.find_all("p"))
        return " \n ".join(p for p in paragraphs if p)

    @retry
    async def _get(self, url: URL) -> Response:
        """
        Fetches the URL and raises an exception if the status code is not 2xx
        We use the tenacity library to retry the request 3 times with exponential backoff
        We use the TTL cache to avoid visiting the same domain too often
        """
        while self.recently_visited.get(url.netloc):
            await sleep(0.1)

        response = await self.client.get(url, follow_redirects=False, timeout=10)
        response.raise_for_status()
        return response

    def load_frontier(self) -> pd.DataFrame:
        """
        Loads the frontier from the CSV file if it exists, otherwise
        it converts the seed URLs to the frontier format and returns it
        """
        fp = self.config.ids_dir / f"{self.run_id}.csv"
        if fp.exists():
            logger.info("Loading the existing frontier")
            return pd.read_csv(fp).set_index("doc_id")
        else:
            logger.info("Creating a new frontier")
            return pd.DataFrame(self.convert_seed_to_frontier()).set_index("doc_id")

    def convert_seed_to_frontier(self) -> Iterator[dict[str, any]]:
        """
        Converts the seed URLs to the frontier format
        """
        for url in self.config.seed_urls:
            yield {
                "doc_id": hashlib.md5(str(url).encode()).hexdigest(),
                "url": url,
                "domain": url.netloc,
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
            logger.debug(f"URL {url} already exists in the frontier")
            return None

        self.frontier.loc[doc_id] = {
            "url": str(url),
            "domain": url.netloc,
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
        pending_docs = self.frontier.query(
            "status == 'pending' and depth < @self.config.max_depth"
        )
        pending_docs = (
            pending_docs.reset_index(drop=False)
            .sort_values(["depth", "created"], ascending=[True, True])
            .groupby("netloc")
            .first()
            .reset_index(drop=False)
            .set_index("doc_id")
        )
        recently_visited = self.recently_visited.keys()
        pending_docs["recently_visited"] = pending_docs["domain"].map(
            lambda x: x in recently_visited
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
            if url := self.extract_url(link_element["href"], base_url):
                yield url

    def extract_url(self, url: str, base_url: URL) -> URL or None:
        try:
            url = URL(url)
        except httpx.InvalidURL:
            logger.debug(f"Invalid URL: {url}")
            return None

        if url.is_relative_url:
            url = base_url.join(url)

        return url if self.is_url_valid(url) else None

    def mark_status(
            self, doc_id: str, status: Literal["pending", "completed", "failed"]
    ) -> None:
        self.frontier.loc[doc_id, "status"] = status

    def save_data(self, doc_id: str, content: str, index: dict[str, any]) -> None:
        with open(self.config.html_dir / f"{doc_id}.html", "w", encoding="utf-8") as f:
            f.write(content)

        with open(self.config.index_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(index, f)

    def save_frontier(self) -> None:
        self.frontier.to_csv(self.config.ids_dir / f"{self.run_id}.csv")

    def create_index(self, response: Response, soup: BeautifulSoup) -> Index:
        url = str(response.url)
        text = self.extract_text_from_soup(soup)
        return {"url": url, "text": text}

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
            await asyncio.gather(*[self.crawl(r) for r in req])
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
