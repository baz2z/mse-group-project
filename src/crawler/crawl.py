import asyncio
import hashlib
import re
from asyncio import sleep
from typing import Iterator, Literal
from urllib.parse import unquote

import httpx
import numpy as np
import pandas as pd
import tenacity
from cachetools import TTLCache
from httpx import AsyncClient, Response, URL
from lxml import html
from lxml.etree import ParserError
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from crawler.config import CrawlerConfig
from crawler.get_logger import get_logger
from crawler.types import ScrapingRequest

TUBINGEN_PATTERN = re.compile(r"t(ü|ue|u)binge([nr])", re.IGNORECASE)


class Crawler:
    def __init__(self, run_id: str = "", config: CrawlerConfig = None):
        self.run_id: str = run_id or pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        self.config: CrawlerConfig = config or CrawlerConfig()
        self.frontier: pd.DataFrame = self.load_frontier()
        self.recently_visited: TTLCache = TTLCache(
            maxsize=512, ttl=self.config.sleep_time
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

    @staticmethod
    def is_html(response: Response) -> bool:
        return response.headers.get("content-type", "").startswith("text/html")

    @staticmethod
    def check_tubingen_in_url(url: URL) -> bool:
        # special case for tuepedia.de
        if url.netloc == b'www.tuepedia.de':
            return True

        # tubingen must be in the domain or the path
        url = url.copy_with(query=None, fragment=None)
        url_string = unquote(str(url))
        return bool(TUBINGEN_PATTERN.search(url_string))

    def is_url_valid(self, url: URL) -> bool:
        return (
                url.is_absolute_url
                and url.scheme in {"http", "https"}
                and url.netloc
                and self.check_tubingen_in_url(url)
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential())
    async def _get(self, url: URL) -> Response:
        while self.recently_visited.get(url.netloc):
            await sleep(0.2)

        async with asyncio.timeout(self.config.timeout):
            response = await self.client.get(url, follow_redirects=True)

        self.recently_visited[url.netloc] = True
        response.raise_for_status()

        return response

    def load_frontier(self) -> pd.DataFrame:
        fp = self.config.ids_dir / f"{self.run_id}.csv"
        if fp.exists():
            logger.debug("Loading the existing frontier")
            return pd.read_csv(fp).set_index("doc_id")
        else:
            logger.debug("Creating a new frontier")
            return (
                pd.DataFrame(self.convert_seed_to_frontier())
                .drop_duplicates(subset=["doc_id"])
                .set_index("doc_id")
            )

    def convert_seed_to_frontier(self) -> Iterator[dict[str, any]]:
        for url in self.config.seed_urls:
            doc_id = self.create_id_for_url(url)
            yield {
                "doc_id": doc_id,
                "url": str(url),
                "domain": url.netloc,
                "depth": 0,
                "status": "pending",
                "created": pd.Timestamp.now(),
                "root": str(url),
            }

    def count_docs(self) -> int:
        return (self.frontier["status"] == "completed").sum()

    def add_to_frontier(self, url: URL, depth: int, root: str) -> None:
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
                "root": root,
            }
            return None

        if entry["status"] == "pending" and entry["depth"] > depth:
            self.frontier.loc[doc_id, "root"] = root
            self.frontier.loc[doc_id, "depth"] = depth

        return None

    def fetch_next_doc_batch(self, limit: int = 256) -> list[ScrapingRequest]:
        pending_docs = self.frontier.query(
            "status == 'pending' and depth < @self.config.max_depth"
        )
        pending_docs["sort_hier"] = np.random.rand(len(pending_docs))
        pending_docs = (
            pending_docs
            .sort_values(["depth", "sort_hier"])
            .drop_duplicates(subset=["domain"], keep="first")
            .head(limit)
        )
        return [
            ScrapingRequest(
                doc_id=str(doc_id),
                url=URL(row["url"]),
                depth=row["depth"],
                root=row["root"],
            )
            for doc_id, row in pending_docs.iterrows()
        ]

    def extract_links(self, content: bytes, base_url: URL) -> Iterator[URL]:
        try:
            for href in html.fromstring(content).xpath("//a/@href"):
                if url := self.extract_url(href, base_url):
                    yield url
        except ParserError:
            logger.debug("ParserError occurred while extracting links")

    def extract_url(self, url: str, base_url: URL) -> URL or None:
        try:
            url = URL(url)
        except httpx.InvalidURL:
            logger.debug(f"Invalid URL: {url}")
            return None

        if url.is_relative_url and base_url:
            url = base_url.join(url)
        elif url.is_relative_url:
            return None

        url = url.copy_with(fragment=None)
        return url if self.is_url_valid(url) else None

    def mark_status(
            self, doc_id: str, status: Literal["pending", "completed", "failed"]
    ) -> None:
        self.frontier.loc[doc_id, "status"] = status

    def save_response(self, doc_id: str, content: bytes) -> None:
        with open(self.config.html_dir / f"{doc_id}.html", "wb") as f:
            f.write(content)

    def save_frontier(self) -> None:
        self.frontier.to_csv(self.config.ids_dir / f"{self.run_id}.csv")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_response(self, req: ScrapingRequest) -> Response or None:
        try:
            return await self._get(req.url)
        except tenacity.RetryError:
            return None

    async def _run(self) -> None:
        pbar = tqdm(total=None)
        while req_batch := self.fetch_next_doc_batch():
            if self.count_docs() >= self.config.max_docs:
                break

            responses = await asyncio.gather(
                *[self.fetch_response(r) for r in req_batch]
            )
            for req, response in zip(req_batch, responses):
                if response is None or not self.is_html(response):
                    self.mark_status(req.doc_id, "failed")
                    continue

                pbar.set_description(f"Processing: {req.doc_id}")
                self.save_response(req.doc_id, response.content)
                for link in self.extract_links(response.content, response.url):
                    self.add_to_frontier(link, req.depth + 1, req.root)

                self.mark_status(req.doc_id, "completed")
                pbar.update(1)

            self.save_frontier()

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
    logger = get_logger("crawler")
    c = Crawler()
    asyncio.run(c.run())
