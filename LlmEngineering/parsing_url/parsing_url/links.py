import requests
from bs4 import BeautifulSoup

def redacted_links(url: str,headers):
    response = requests.get(url,headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    return [a['href'] for a in soup.find_all('a', href=True)]