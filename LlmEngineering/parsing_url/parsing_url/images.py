import requests
from bs4 import BeautifulSoup

def redacted_images(url: str, headers):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    return [img['src'] for img in soup.find_all('img', src=True)]
