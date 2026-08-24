import requests
from bs4 import BeautifulSoup

def redacted_headings(url: str,headers):
    response = requests.get(url,headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    return [h.get_text(strip=True) for h in soup.find_all(['h1','h2','h3'])]
