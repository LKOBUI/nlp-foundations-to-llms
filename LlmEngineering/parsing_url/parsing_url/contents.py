import requests
from bs4 import BeautifulSoup

def redacted_contents(url: str, headers):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    return [p.get_text(strip=True) for p in soup.find_all('p')]

