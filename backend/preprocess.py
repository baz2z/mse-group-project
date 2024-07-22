""" Preprocessing functions for the backend. """
from bs4 import BeautifulSoup


def sanitize_html(html_content):
    """
    Sanitize HTML content by removing script, style, and link tags,
    and adjusting image sources and stylesheet links.

    :param html_content: The HTML content to sanitize.
    :return: The sanitized HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and link tags
    for script in soup(['script', 'style']):
        script.decompose()

    # Adjust or remove image src attributes
    for img in soup.find_all('img'):
        img['src'] = 'about:blank'  # Placeholder for missing images

    # Adjust or remove stylesheet links
    for link in soup.find_all('link', rel='stylesheet'):
        link.decompose()

    return str(soup)


def extract_title(html_content):
    """
    Extract the title from HTML content.

    :param html_content: The HTML content from which to extract the title.
    :return: The extracted title or 'No Title' if no title tag is found.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    title_tag = soup.find('title')
    return title_tag.get_text() if title_tag else 'No Title'


def extract_description(html_content):
    """
    Extract the meta description from HTML content.

    :param html_content: The HTML content from which to extract the description.
    :return: The extracted description or 'No description available' if no meta description is found.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_description = soup.find('meta', attrs={'name': 'description'})
    return meta_description['content'] if meta_description else 'No description available'


def extract_first_paragraph(html_content):
    """
    Extract the first paragraph from HTML content.

    :param html_content: The HTML content from which to extract the first paragraph.
    :return: The first 300 characters of the first paragraph or 'No description available' if no paragraph is found.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    first_paragraph = soup.find('p')
    return first_paragraph.get_text()[:300] if first_paragraph else 'No description available'


def preprocess_dist_string(dist_string):
    """
    Preprocess a distance string to ensure it's in a valid JSON array format.

    :param dist_string: The distance string to preprocess.
    :return: The preprocessed distance string in JSON array format.
    """
    # Remove leading and trailing spaces and brackets
    dist_string = dist_string.strip().strip('[]')
    
    # Replace multiple spaces with a single comma
    dist_string = ','.join(part.strip() for part in dist_string.split())
    
    # Add brackets around the string to ensure it's a valid JSON array format
    return f'[{dist_string}]'
