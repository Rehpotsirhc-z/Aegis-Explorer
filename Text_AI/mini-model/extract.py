import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Glossary_of_video_game_terms"

# Fetch the content from the URL
response = requests.get(url)
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find all <dfn> tags
    dfn_tags = soup.find_all("dfn")
    
    # Extract and clean the text from each <dfn> tag
    terms = [tag.get_text(strip=True) for tag in dfn_tags]
    
    # Print the extracted terms
    for term in terms:
        print(term)
else:
    print("Failed to retrieve the page. Status code:", response.status_code)
