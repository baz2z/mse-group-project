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

# Regular expressions to match the Tübingen in the text
TUBINGEN_PATTERN = re.compile(r"t(ü|ue|u)binge([nr])", re.IGNORECASE)

# Regular expression to match the English language code
ENGLISH_PATTERN = re.compile(r"^en([-_](us|gb|de))?$", re.IGNORECASE)


class Crawler:
    """
    Main crawler class to fetch the web pages.
    """

    def __init__(self, run_id: str = "", config: CrawlerConfig | None = None):
        """
        Initialize the crawler.

        Args:
            run_id: Run ID for the crawler
            config: Crawler configuration

        Returns:
            None
        """
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
        """
        Get the HTTP client. Create a new one if it does not exist.

        Returns:
            HTTP client
        """
        if self._client is None:
            self.logger.info("Creating a new HTTP client")
            self._client = AsyncClient(headers=self.config.headers)
        return self._client

    def html_too_large(self, content: bytes) -> bool:
        """
        Check if the HTML content is too large.

        Args:
            content: HTML content

        Returns:
            True if the content is too large, False otherwise
        """
        return len(content) > self.config.max_filesize

    @staticmethod
    def get_main_domain(url: URL) -> str:
        """
        Get the main domain from the URL.

        Args:
            url: URL object

        Returns:
            Main domain of the URL
        """
        return (
            ".".join(url.host.split(".")[-2:]) if url.host.count(".") > 1 else url.host
        )

    @staticmethod
    def clean_url(url: URL) -> URL:
        """
        Clean the URL by removing the query, fragment, and parameters.

        Args:
            url: URL object

        Returns:
            Cleaned URL with no query, fragment, or parameters
        """
        return url.copy_with(query=None, fragment=None, params=None)

    @staticmethod
    def create_id_for_url(url: URL) -> str:
        """
        Create hash of the URL to use as a unique ID.

        Args:
            url: URL object

        Returns:
            Unique ID for the URL
        """
        return hashlib.md5(str(url).encode()).hexdigest()

    @staticmethod
    def is_html(response: Response) -> bool:
        """
        Check if the response is HTML.

        Args:
            response: HTTP response

        Returns:
            True if the response is HTML, False otherwise
        """
        return (
            response.headers.get("content-type", "")
            .lower()
            .strip()
            .startswith("text/html")
        )

    @staticmethod
    def is_english(tree: html.HtmlElement) -> bool:
        """
        Check if the HTML content is in English.

        Args:
            tree: HTML tree object

        Returns:
            True if the content is in English, False otherwise
        """
        return any(
            bool(ENGLISH_PATTERN.search(lang.strip()))
            for lang in tree.xpath("//html/@lang")
        )

    def is_url_valid(self, url: URL) -> bool:
        """
        Check if the URL is valid.

        Args:
            url: URL object

        Returns:
            True if the URL is valid, False otherwise
        """
        try:
            return (
                url.is_absolute_url
                and url.scheme in {"http", "https"}
                and not any(d.search(url.host) for d in self.config.denied_domains)
            )
        except Exception as e:
            self.logger.debug(f"Invalid URL: {url}, {e}")
            return False

    def parse_html(self, content: bytes) -> html.HtmlElement:
        """
        Parse the HTML content.

        Args:
            content: HTML content

        Returns:
            HTML tree object
        """
        try:
            return html.fromstring(content)
        except ParserError as e:
            self.logger.debug(f"Failed to parse HTML: {e})")
            return html.fromstring("<html></html>")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _get(self, url: URL) -> Response:
        """
        Fetch the URL using the HTTP client with retries.

        Args:
            url: URL object

        Returns:
            HTTP response
        """
        async with asyncio.timeout(self.config.timeout):
            response = await self.client.get(url, follow_redirects=True)

        response.raise_for_status()
        return response

    def load_frontier(self) -> pd.DataFrame:
        """
        Load the frontier from the file or create a new one.

        Returns:
            Frontier DataFrame
        """
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
        """
        Convert the seed URLs to the frontier format.

        Returns:
            Iterator of dictionaries with the frontier format
        """
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
        """
        Add the URL to the frontier.

        Args:
            url: URL object
            priority: Priority of the URL
            depth: Depth of the URL
            root: ID of the root document

        Returns:
            None
        """
        doc_id = self.create_id_for_url(url)

        try:
            # test if the URL is already in the frontier
            entry = self.frontier.loc[doc_id]
        except KeyError:
            # add the URL to the frontier if it is not there
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

        # update the existing entry if the URL is already in the frontier
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
        """
        Fetch the next batch of documents to scrape.

        Returns:
            List of ScrapingRequest objects
        """
        # fetch the next batch using the priority, depth, and random sort key
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
        """
        Parse the HTML content and add the new links to the frontier.

        Args:
            request: ScrapingRequest object
            response: HTTP response

        Returns:
            None
        """
        tree = self.parse_html(content=response.content)
        if not bool(TUBINGEN_PATTERN.search(tree.text_content())):
            # don't add any new links if the page does not contain "Tübingen"
            return None

        if self.is_english(tree=tree):
            # set priority to high if the page is in English
            priority = Priority.high
            self.update_doc(
                request.doc_id, features_tubingen=True, features_english=True
            )
        else:
            self.update_doc(request.doc_id, features_tubingen=True)
            priority = Priority.low

            # test if the page has an English version
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
        """
        Extract the URL from the href attribute.

        Args:
            url: URL string
            base_url: Base URL object

        Returns:
            URL object or None
        """
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
        """
        Update the document in the frontier.

        Args:
            doc_id: Document ID
            update_dict: Dictionary with the updates

        Returns:
            None
        """
        update_dict["updated"] = pd.Timestamp.now()
        self.frontier.loc[doc_id, list(update_dict.keys())] = list(update_dict.values())

    def save_response(self, doc_id: str, content: bytes) -> None:
        """
        Save the HTML content to the file.

        Args:
            doc_id: Document ID
            content: HTML content

        Returns:
            None
        """
        (self.html_dir / f"{doc_id}.html").write_bytes(content)

    def save_frontier(self) -> None:
        """
        Save the frontier to the file.

        Returns:
            None
        """
        self.frontier.to_csv(self.base_dir / f"{self.run_id}.csv")

    async def close(self) -> None:
        """
        Close the HTTP client.

        Returns:
            None
        """
        if self._client:
            self.logger.info("Closing the HTTP client")
            await self._client.aclose()
            self._client = None

    async def fetch_response(self, req: ScrapingRequest) -> Response or None:
        """
        Fetch the response for the URL.

        Args:
            req: ScrapingRequest object

        Returns:
            HTTP response or None
        """
        try:
            return await self._get(req.url)
        except tenacity.RetryError:
            self.logger.debug(f"Failed to fetch: {req.url}")
            return None

    async def _run(self) -> None:
        """
        Main loop to fetch the documents.

        Returns:
            None
        """
        pbar = tqdm(total=None)

        while req_batch := self.fetch_next_doc_batch():
            pbar.set_description(f"Fetching {len(req_batch)} URLs")

            # fetch the responses in parallel
            responses = await asyncio.gather(
                *(self.fetch_response(r) for r in req_batch)
            )
            for req, response in zip(req_batch, responses):
                if (
                    response is None
                    or not self.is_html(response)
                    or self.html_too_large(response.content)
                ):
                    # update the document status to failed if the response is invalid
                    self.update_doc(req.doc_id, status=Status.failed.value)
                    continue

                pbar.set_description(f"Processing: {req.doc_id}")
                # save the response and add the new links to the frontier
                self.save_response(req.doc_id, response.content)
                self.add_new_links_to_frontier(req, response)
                self.update_doc(
                    req.doc_id,
                    status=Status.completed.value,
                    url=str(response.url),
                    domain=response.url.host,
                )
                pbar.update(1)

            # save the frontier after each batch
            self.save_frontier()

        pbar.close()

    async def run(self) -> None:
        """
        Run the crawler.

        Returns:
            None
        """
        try:
            await self._run()
        finally:
            await self.close()


if __name__ == "__main__":
    c = Crawler("latest")
    asyncio.run(c.run())
