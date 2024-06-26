import requests
from bs4 import BeautifulSoup
import csv
import hashlib
import os
from urllib.parse import urlparse
from queue import PriorityQueue
from urllib.parse import urljoin


def is_valid_url(url):
    """Check if the URL has a valid scheme and netloc."""
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)


#Crawl the web. You need (at least) two parameters:
#frontier: The frontier of known URLs to crawl. You will initially populate this with your seed set of URLs and later maintain all discovered (but not yet crawled) URLs here.
#index: The location of the local index storing the discovered documents. 
def crawl(frontier, index):
	retrieved_pages = {}
	visited = set()
	pq = PriorityQueue()
	for url in frontier:
		pq.put((0, url))
	while not pq.empty() and len(visited) < 5:
		priority, current_url = pq.get()
		if current_url in visited or not is_valid_url(current_url):
			continue
		print(current_url)
		response = requests.get(current_url)
		soup = BeautifulSoup(response.content, "html.parser")
		link_elements = soup.select("a[href]")
		for link_element in link_elements:
			url = link_element["href"]
			if not url.startswith(('http://', 'https://')):
				url = urljoin(current_url, url)
			#print("internal link" + url)
			pq.put((priority + 1, url))
		retrieved_pages[current_url] = response.content
		visited.add(current_url)
	save_data(retrieved_pages, index)
                                                                
def save_data(data, index):
	# Assuming this code is inside a function or somewhere relevant in your script
	data_directory = os.path.join(os.path.dirname(__file__), '..', 'dat')
	csv_file_path = os.path.join(data_directory, index)

	# Ensure the data directory exists
	os.makedirs(data_directory, exist_ok=True)

	with open(csv_file_path, "w") as f:
		for url, content in data.items():
			f.write(f"{url}\t{content}\n")


if __name__ == "__main__":
    urls = ["https://www.tuebingen.de/"]
    crawl(urls, "documents.csv")