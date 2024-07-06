import asyncio
import hashlib
import json
import os
from asyncio import sleep
from typing import Iterator, Literal

import httpx
import pandas as pd
import tenacity
from bs4 import BeautifulSoup
from cachetools import TTLCache
from httpx import AsyncClient, Response, URL
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from crawler.config import CrawlerConfig
from crawler.types import Index, ScrapingRequest
from get_logger import get_logger


class Crawler:
    """
    Web crawler class to scrape the web pages
    """

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
            logger.debug("Creating a new HTTP client")
            self._client = AsyncClient(headers=self.config.headers)
        return self._client

    @staticmethod
    def create_id_for_url(url: URL) -> str:
        """
        Creates a unique identifier for the URL
        """
        return hashlib.md5(str(url).encode()).hexdigest()

    def check_allowed_domains(self, domain: bytes) -> bool:
        """
        Checks if the domain is in the allowed domains list
        """
        if not self.config.allowed_domains_pattern:
            return True

        for pattern in self.config.allowed_domains_pattern:
            if pattern.match(domain.decode()):
                return True

        return False

    def check_denied_domains(self, domain: bytes) -> bool:
        """
        Checks if the domain is in the forbidden domains list
        """
        for pattern in self.config.forbidden_domains_pattern:
            if pattern.match(domain.decode()):
                return False

        return True

    @staticmethod
    def check_url_has_extension(url: URL) -> bool:
        """
        Checks if the URL has an extension (e.g., .jpg, .png, .pdf)
        """
        _, ext = os.path.splitext(url.path)
        if ext and ext != ".html":
            return True

        return False

    def is_url_valid(self, url: URL) -> bool:
        """
        Checks if the URL is valid by checking if it has a scheme and netloc
        and if it does not have an extension (e.g., .jpg, .png, .pdf)
        """
        if not (url.scheme and url.netloc):
            return False

        if self.check_url_has_extension(url):
            return False

        if not self.check_denied_domains(url.netloc):
            return False

        if not self.check_allowed_domains(url.netloc):
            return False

        return True

    @staticmethod
    def extract_text_from_soup(soup: BeautifulSoup) -> str:
        """
        Extracts the text from the soup object by joining all paragraphs
        """
        paragraphs = (p.get_text().strip() for p in soup.find_all("p"))
        return " \n ".join(p for p in paragraphs if p)

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(max=10))
    async def _get(self, url: URL) -> Response:
        """
        Fetches the URL and raises an exception if the status code is not 2xx
        We use the tenacity library to retry the request 3 times with exponential backoff
        We use the TTL cache to avoid visiting the same domain too often
        """
        while self.recently_visited.get(url.netloc):
            await sleep(0.2)

        async with asyncio.timeout(self.config.timeout):
            response = await self.client.get(url, follow_redirects=True)

        response.raise_for_status()

        self.recently_visited[url.netloc] = True
        return response

    def load_frontier(self) -> pd.DataFrame:
        """
        Loads the frontier from the CSV file if it exists, otherwise
        it converts the seed URLs to the frontier format and returns it
        """
        fp = self.config.ids_dir / f"{self.run_id}.csv"
        if fp.exists():
            logger.debug("Loading the existing frontier")
            return pd.read_csv(fp).set_index("doc_id")
        else:
            logger.debug("Creating a new frontier")
            return pd.DataFrame(self.convert_seed_to_frontier()).set_index("doc_id")

    def convert_seed_to_frontier(self) -> Iterator[dict[str, any]]:
        """
        Converts the seed URLs to the frontier format
        """
        for url in self.config.seed_urls:
            yield {
                "doc_id": self.create_id_for_url(url),
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
        doc_id = self.create_id_for_url(url)

        try:
            entry = self.frontier.loc[doc_id]
        except KeyError:
            self.frontier.loc[doc_id] = {
                "url": str(url),
                "domain": url.netloc,
                "depth": depth,
                "status": "pending",
                "created": pd.Timestamp.now(),
            }
            return None

        if entry["status"] == "pending" and entry["depth"] > depth:
            self.frontier.loc[doc_id, "depth"] = depth

        return None

    def fetch_next_doc_batch(self, limit: int = 32) -> list[ScrapingRequest]:
        """
        Fetches the next batch of documents to be scraped
        1. Query the frontier for pending documents that are not yet visited
        2. Filter the documents by conditions (e.g., depth < max_depth and allowed_domains)
        3. Sort the documents by depth and created date
        4. Group the documents by netloc
        5. Prioritize the documents that are recently visited
        6. Return a batch of documents to be scraped (up to the limit, all from different domains)
        """
        pending_docs = (
            self.frontier.query(
                "status == 'pending' and depth < @self.config.max_depth"
            )
            .sort_values(["depth", "created"], ascending=[True, True])
            .drop_duplicates(subset=["domain"], keep="first")
            .head(limit)
        )
        return [
            ScrapingRequest(
                doc_id=str(doc_id),
                url=URL(row["url"]),
                depth=row["depth"],
            )
            for doc_id, row in pending_docs.iterrows()
        ]

    def extract_links(self, soup: BeautifulSoup, base_url: URL) -> Iterator[URL]:
        for link_element in soup.find_all("a", href=True):
            if url := self.extract_url(link_element["href"], base_url):
                yield url

    def extract_url(self, url: str, base_url: URL) -> URL or None:
        """
        Extracts the URL from the href attribute of the anchor tag
        """
        try:
            url = URL(url)
        except httpx.InvalidURL:
            logger.debug(f"Invalid URL: {url}")
            return None

        if url.is_relative_url:
            url = base_url.join(url)

        url = url.copy_with(fragment=None)

        return url if self.is_url_valid(url) else None

    def mark_status(
        self, doc_id: str, status: Literal["pending", "completed", "failed"]
    ) -> None:
        """
        Marks the status of the document in the frontier
        """
        self.frontier.loc[doc_id, "status"] = status

    def save_data(self, doc_id: str, content: str, index: dict[str, any]) -> None:
        """
        Saves the HTML content and index to the respective directories
        """
        with open(self.config.html_dir / f"{doc_id}.html", "w", encoding="utf-8") as f:
            f.write(content)

        with open(self.config.index_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(index, f)

    def save_frontier(self) -> None:
        """
        Saves the frontier to the CSV file
        """
        self.frontier.to_csv(self.config.ids_dir / f"{self.run_id}.csv")

    def create_index(self, response: Response, soup: BeautifulSoup) -> Index:
        """
        Creates an index for the document
        """
        url = str(response.url)
        text = self.extract_text_from_soup(soup)
        return {"url": url, "text": text}

    @staticmethod
    def _get_soup(response: Response) -> BeautifulSoup:
        """
        Parses the HTML content using BeautifulSoup
        """
        if response.text.startswith("<?xml"):
            return BeautifulSoup("", "lxml")

        try:
            return BeautifulSoup(response.text, "lxml")
        except Exception as e:
            logger.error(f"Error parsing the HTML: {e}")
            return BeautifulSoup("", "lxml")

    async def crawl(self, req: ScrapingRequest) -> bool:
        """
        Crawls the URL and extracts the text and links
        Creates an index and saves the HTML and index files
        Marks the status as completed if successful, otherwise as failed
        Saves new URLs to the frontier
        """
        try:
            response = await self._get(req.url)
        except tenacity.RetryError:
            response = None

        if (
            response is None
            or response.headers.get("content-type", "").split(";")[0] != "text/html"
        ):
            self.mark_status(req.doc_id, "failed")
            return False

        soup = self._get_soup(response)
        index = self.create_index(response, soup)
        self.save_data(req.doc_id, response.text, index)
        self.mark_status(req.doc_id, "completed")

        for url in self.extract_links(soup, response.url):
            self.add_to_frontier(url, req.depth + 1)

        return True

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
            logger.debug(f"Fetching {len(req)} documents")
            results = await asyncio.gather(*[self.crawl(r) for r in req])
            self.save_frontier()
            pbar.update(sum(results))

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
    import logging

    logger = get_logger("crawler", logging.INFO)

    c = Crawler()
    asyncio.run(c.run())
