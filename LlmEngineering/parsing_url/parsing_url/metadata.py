import requests
from bs4 import BeautifulSoup

def redacted_metadata(url: str,headers):
    """Fetch common metadata (title, description, keywords) from the given URL."""
    response = requests.get(url,headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    metadata = {}
    # Page title
    metadata['title'] = soup.title.string if soup.title else None
    # Meta description
    desc_tag = soup.find('meta', attrs={'name':'description'})
    metadata['description'] = desc_tag['content'] if desc_tag else None
    # Meta keywords
    keywords_tag = soup.find('meta', attrs={'name':'keywords'})
    metadata['keywords'] = keywords_tag['content'] if keywords_tag else None
    return metadata