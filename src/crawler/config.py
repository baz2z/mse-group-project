import re
from dataclasses import dataclass, field
from pathlib import Path

from httpx import URL

HARD_DRIVE = Path("D://")
assert HARD_DRIVE.exists(), "Please connect the hard drive used for the project."


def get_seed_urls() -> set[URL]:
    """
    Get the seed URLs from the query results.

    Returns:
        Set of seed URLs.
    """
    return {
        URL("https://allevents.in/tubingen"),
        URL("https://en.wikipedia.org/wiki/T%C3%BCbingen"),
        URL("https://en.wikivoyage.org/wiki/T%C3%BCbingen"),
        URL("https://historicgermany.travel/historic-germany/tubingen/"),
        URL("https://kunsthalle-tuebingen.de/en/"),
        URL(
            "https://theculturetrip.com/europe/germany/articles/the-best-things-to-see-and-do-in-tubingen-germany"
        ),
        URL("https://tuebingen.ai/"),
        URL("https://tuebingenresearchcampus.com/"),
        URL("https://uni-tuebingen.de/"),
        URL("https://velvetescape.com/things-to-do-in-tubingen/"),
        URL("https://www.booking.com/accommodation/city/de/tubingen.html"),
        URL("https://www.germansights.com/tubingen/"),
        URL("https://www.germany.travel/en/cities-culture/tuebingen.html"),
        URL("https://www.komoot.com/guide/210692/attractions-around-tuebingen"),
        URL(
            "https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/"
        ),
        URL("https://www.tourism-bw.com/attractions/old-town-of-tuebingen-592d513a97"),
        URL(
            "https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html"
        ),
        URL("https://www.tuebingen.de/en/"),
        URL("https://www.tuebingen.mpg.de/en"),
    }


def get_headers() -> dict[str, str]:
    """
    Get the headers for the HTTP requests.

    Returns:
        Headers for the HTTP requests.
    """
    # HTTP headers to use for the requests to simulate a browser
    return {
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


def get_denied_domains():
    return {
        re.compile(r"(?!en|de)\.wiki\w*\.org"),
        re.compile(r"web.archive.org"),
        re.compile(r"facebook.com"),
        re.compile(r"twitter.com"),
        re.compile(r"youtube.com"),
        re.compile(r"instagram.com"),
        re.compile(r"linkedin.com"),
        re.compile(r"reddit.com"),
    }


@dataclass(frozen=True)
class CrawlerConfig:
    DIR: Path = HARD_DRIVE
    batch_size: int = 256
    max_depth: int = 10
    timeout: float = 10
    max_filesize: int = 10_000_000
    seed_urls: set[URL] = field(default_factory=get_seed_urls)
    headers: dict[str, str] = field(default_factory=get_headers)
    denied_domains: set[re.Pattern] = field(default_factory=get_denied_domains)
