import asyncio
import hashlib
import logging
import random
import re
from typing import Iterator

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
DENIED_DOMAINS = {
    re.compile(r"(?!en|de)\.wiki\w*\.org"),
    re.compile(r"web.archive.org"),
    re.compile(r"facebook.com"),
    re.compile(r"twitter.com"),
    re.compile(r"youtube.com"),
    re.compile(r"instagram.com"),
    re.compile(r"linkedin.com"),
    re.compile(r"reddit.com"),
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/88.0.4324.150 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;"
    "q=0.9,image/avif,image/webp,image/apng,*/*;"
    "q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
}


class Crawler:
    def __init__(self, run_id: str = "", config: CrawlerConfig | None = None):
        self.run_id: str = run_id or pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        self.config: CrawlerConfig = config or CrawlerConfig()

        self.base_dir = self.config.DIR / f"mse_{self.run_id}"
        self.html_dir = self.base_dir / "html"
        self.html_dir.mkdir(parents=True, exist_ok=True)

        self._client: AsyncClient or None = None
        self.logger = get_logger("crawler", logging.INFO)

        self.frontier: pd.DataFrame = self.load_frontier()
        self.logger.info(f"Run ID: {self.run_id}, Frontier size: {len(self.frontier)}")

    @property
    def client(self) -> AsyncClient:
        if self._client is None:
            self.logger.info("Creating a new HTTP client")
            self._client = AsyncClient(headers=headers)
        return self._client

    @staticmethod
    def get_main_domain(url: URL) -> str:
        return (
            ".".join(url.host.split(".")[-2:]) if url.host.count(".") > 1 else url.host
        )

    @staticmethod
    def clean_url(url: URL) -> URL:
        return url.copy_with(query=None, fragment=None, params=None)

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
            and not any(d.search(url.host) for d in DENIED_DOMAINS)
        )

    def parse_html(self, content: bytes) -> html.HtmlElement:
        try:
            return html.fromstring(content)
        except ParserError as e:
            self.logger.debug(f"Failed to parse HTML: {e})")
            return html.fromstring("<html></html>")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _get(self, url: URL) -> Response:
        async with asyncio.timeout(self.config.timeout):
            response = await self.client.get(url, follow_redirects=True)

        response.raise_for_status()
        return response

    def load_frontier(self) -> pd.DataFrame:
        fp = self.base_dir / f"{self.run_id}.csv"
        if fp.exists():
            self.logger.info("Loading the existing frontier")
            return pd.read_csv(fp).set_index("doc_id")
        else:
            self.logger.info("Creating a new frontier")
            return (
                pd.DataFrame(self.convert_seed_to_frontier())
                .drop_duplicates(subset=["doc_id"], keep="first")
                .set_index("doc_id")
            )

    def convert_seed_to_frontier(self) -> Iterator[dict[str, any]]:
        for _url in self.config.seed_urls:
            url = self.clean_url(_url)
            if not self.is_url_valid(url):
                continue

            doc_id = self.create_id_for_url(url)
            now = pd.Timestamp.now()
            yield {
                "doc_id": doc_id,
                "url": str(url),
                "domain": url.host,
                "main_domain": self.get_main_domain(url),
                "depth": 0,
                "priority": Priority.high.value,
                "status": Status.pending.value,
                "created": now,
                "updated": now,
                "root": doc_id,
                "random_sort_key": random.random(),
                "features_tubingen": False,
                "features_english": False,
            }

    def add_to_frontier(
        self, url: URL, priority: Priority, depth: int, root: str
    ) -> None:
        doc_id = self.create_id_for_url(url)

        try:
            entry = self.frontier.loc[doc_id]
        except KeyError:
            now = pd.Timestamp.now()
            self.frontier.loc[doc_id] = {
                "url": str(url),
                "domain": url.host,
                "main_domain": self.get_main_domain(url),
                "depth": depth,
                "priority": priority.value,
                "status": Status.pending.value,
                "created": now,
                "updated": now,
                "root": root,
                "random_sort_key": random.random(),
                "features_tubingen": False,
                "features_english": False,
            }
            return None

        if entry["status"] == Status.pending and (
            entry["depth"] > depth or entry["priority"] < priority.value
        ):
            self.update_doc(
                doc_id,
                depth=min(entry["depth"], depth),
                priority=max(entry["priority"], priority.value),
            )

        return None

    def fetch_next_doc_batch(self) -> list[ScrapingRequest]:
        pending_docs = (
            self.frontier.query(
                "status == 'pending' and depth < @self.config.max_depth"
            )
            .sort_values(
                ["priority", "depth", "random_sort_key"], ascending=[False, True, True]
            )
            .drop_duplicates(subset=["main_domain"], keep="first")
            .head(self.config.batch_size)
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
        tree = self.parse_html(content=response.content)
        if not bool(TUBINGEN_PATTERN.search(tree.text_content())):
            return None

        if self.is_english(tree=tree):
            priority = Priority.high
            self.update_doc(request.doc_id, features_tubingen=True, features_english=True)
        else:
            self.update_doc(request.doc_id, features_tubingen=True)
            priority = Priority.low
            url_en = response.url.copy_with(path="/en", query=None, fragment=None)
            if self.is_url_valid(url_en):
                self.add_to_frontier(
                    url=url_en,
                    priority=Priority.high,
                    depth=request.depth + 1,
                    root=request.root,
                )

        for href in tree.xpath("//a/@href"):
            if url := self.extract_url(url=href, base_url=response.url):
                self.add_to_frontier(
                    url=url,
                    priority=priority,
                    depth=request.depth + 1,
                    root=request.root,
                )

    def extract_url(self, url: str, base_url: URL) -> URL or None:
        try:
            url = URL(url)
        except httpx.InvalidURL:
            self.logger.debug(f"Invalid URL: {url}")
            return None

        if url.is_relative_url:
            url = base_url.join(url)

        url = self.clean_url(url)
        return url if self.is_url_valid(url) else None

    def update_doc(self, doc_id: str, **update_dict) -> None:
        update_dict["updated"] = pd.Timestamp.now()
        self.frontier.loc[doc_id, list(update_dict.keys())] = list(update_dict.values())

    def save_response(self, doc_id: str, content: bytes) -> None:
        (self.html_dir / f"{doc_id}.html").write_bytes(content)

    def save_frontier(self) -> None:
        self.frontier.to_csv(self.base_dir / f"{self.run_id}.csv")

    async def close(self) -> None:
        if self._client:
            self.logger.info("Closing the HTTP client")
            await self._client.aclose()
            self._client = None

    async def fetch_response(self, req: ScrapingRequest) -> Response or None:
        try:
            return await self._get(req.url)
        except tenacity.RetryError:
            self.logger.debug(f"Failed to fetch: {req.url}")
            return None

    async def _run(self) -> None:
        pbar = tqdm(total=None)
        while req_batch := self.fetch_next_doc_batch():
            pbar.set_description(f"Fetching {len(req_batch)} URLs")
            responses = await asyncio.gather(
                *(self.fetch_response(r) for r in req_batch)
            )
            for req, response in zip(req_batch, responses):
                if response is None or not self.is_html(response):
                    self.update_doc(req.doc_id, status=Status.failed.value)
                    continue

                pbar.set_description(f"Processing: {req.doc_id}")
                self.save_response(req.doc_id, response.content)
                self.add_new_links_to_frontier(req, response)
                self.update_doc(
                    req.doc_id,
                    status=Status.completed.value,
                    url=str(response.url),
                    domain=response.url.host,
                )
                pbar.update(1)

            self.save_frontier()

        pbar.close()

    async def run(self) -> None:
        try:
            await self._run()
        finally:
            await self.close()


if __name__ == "__main__":
    c = Crawler("latest")
    asyncio.run(c.run())
