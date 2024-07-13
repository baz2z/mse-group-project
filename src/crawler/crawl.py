import asyncio
import hashlib
import logging
import re
from typing import Iterator
from urllib.parse import unquote

import httpx
import pandas as pd
import tenacity
from httpx import AsyncClient, Response, URL
from lxml import html
from lxml.etree import ParserError
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from crawler.config import CrawlerConfig
from crawler.get_logger import get_logger
from crawler.types import Priority, ScrapingRequest, Status

pd.options.mode.chained_assignment = None

TUBINGEN_PATTERN = re.compile(r"t(ü|ue|u)binge([nr])", re.IGNORECASE)
ENGLISH_PATTERN = re.compile(r"^en([-_](us|gb|de))?$", re.IGNORECASE)
HOSTNAME_PATTERN = re.compile(r"\S+\.(de|com|org|net)$")


class Crawler:
    def __init__(self, run_id: str = "", config: CrawlerConfig = None):
        self.run_id: str = run_id or pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        self.config: CrawlerConfig = config or CrawlerConfig()
        self.frontier: pd.DataFrame = self.load_frontier()
        self._client: AsyncClient or None = None

    @property
    def client(self) -> AsyncClient:
        if self._client is None:
            logger.debug("Creating a new HTTP client")
            self._client = AsyncClient(headers=self.config.headers)
        return self._client

    @staticmethod
    def create_id_for_url(url: URL) -> str:
        return hashlib.md5(str(url).encode()).hexdigest()

    @staticmethod
    def is_html(response: Response) -> bool:
        return (
            response.headers.get("content-type", "")
            .lower()
            .strip()
            .startswith("text/html")
        )

    @staticmethod
    def is_english(tree: html.HtmlElement) -> bool:
        return any(
            bool(ENGLISH_PATTERN.search(lang.strip()))
            for lang in tree.xpath("//html/@lang")
        )

    @staticmethod
    def is_url_valid(url: URL) -> bool:
        return (
            url.is_absolute_url
            and url.scheme in {"http", "https"}
            and bool(HOSTNAME_PATTERN.search(url.host))
            and bool(TUBINGEN_PATTERN.search(unquote(str(url))))
        )

    @staticmethod
    def parse_html(content: bytes) -> html.HtmlElement:
        try:
            return html.fromstring(content)
        except ParserError:
            logger.debug("ParserError occurred while parsing HTML")
            return html.fromstring("<html></html>")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _get(self, url: URL) -> Response:
        async with asyncio.timeout(self.config.timeout):
            response = await self.client.get(url, follow_redirects=True)

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
                "domain": url.host,
                "depth": 0,
                "priority": Priority.high.value,
                "status": "pending",
                "created": pd.Timestamp.now(),
                "root": doc_id,
            }

    def add_to_frontier(
        self, url: URL, priority: Priority, depth: int, root: str
    ) -> None:
        doc_id = self.create_id_for_url(url)

        try:
            entry = self.frontier.loc[doc_id]
        except KeyError:
            self.frontier.loc[doc_id] = {
                "url": str(url),
                "domain": url.host,
                "depth": depth,
                "priority": priority.value,
                "status": Status.pending.value,
                "created": pd.Timestamp.now(),
                "root": root,
            }
            return None

        if entry["status"] == Status.pending:
            if entry["depth"] > depth:
                self.frontier.loc[doc_id, "depth"] = depth

            if entry["priority"] < priority.value:
                self.frontier.loc[doc_id, "priority"] = priority.value

        return None

    def fetch_next_doc_batch(self, limit: int = 256) -> list[ScrapingRequest]:
        pending_docs = (
            self.frontier.query(
                "status == 'pending' and depth < @self.config.max_depth"
            )
            .sort_values(
                ["priority", "depth", "created"], ascending=[False, True, True]
            )
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

    def add_new_links_to_frontier(
        self, request: ScrapingRequest, response: Response
    ) -> None:
        tree = self.parse_html(response.content)
        priority = Priority.high if self.is_english(tree) else Priority.low

        for href in tree.xpath("//a/@href"):
            if url := self.extract_url(href, request.url):
                self.add_to_frontier(url, priority, request.depth + 1, request.root)

    def extract_url(self, url: str, base_url: URL) -> URL or None:
        try:
            url = URL(url)
        except httpx.InvalidURL:
            logger.debug(f"Invalid URL: {url}")
            return None

        if url.is_relative_url:
            url = base_url.join(url)

        url = url.copy_with(query=None, fragment=None, params=None)
        return url if self.is_url_valid(url) else None

    def mark_status(self, doc_id: str, status: Status) -> None:
        self.frontier.loc[doc_id, "status"] = status.value

    def save_response(self, doc_id: str, content: bytes) -> None:
        (self.config.html_dir / f"{doc_id}.html").write_bytes(content)

    def save_frontier(self) -> None:
        self.frontier.to_csv(self.config.ids_dir / f"{self.run_id}.csv")

    async def close(self) -> None:
        if self._client:
            logger.debug("Closing the HTTP client")
            await self._client.aclose()
            self._client = None

    async def fetch_response(self, req: ScrapingRequest) -> Response or None:
        try:
            return await self._get(req.url)
        except tenacity.RetryError:
            logger.debug(f"Failed to fetch: {req.url}")
            return None

    async def _run(self) -> None:
        pbar = tqdm(total=None)
        while req_batch := self.fetch_next_doc_batch():
            responses = await asyncio.gather(
                *(self.fetch_response(r) for r in req_batch)
            )
            for req, response in zip(req_batch, responses):
                if response is None or not self.is_html(response):
                    self.mark_status(req.doc_id, Status.failed)
                    continue

                pbar.set_description(f"Processing: {req.doc_id}")
                self.save_response(req.doc_id, response.content)
                self.add_new_links_to_frontier(req, response)
                self.mark_status(req.doc_id, Status.completed)
                pbar.update(1)

            self.save_frontier()

        pbar.close()

    async def run(self) -> None:
        try:
            await self._run()
        finally:
            await self.close()


if __name__ == "__main__":
    logger = get_logger("crawler", logging.DEBUG)
    c = Crawler()
    asyncio.run(c.run())
