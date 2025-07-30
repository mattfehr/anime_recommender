# backend/scraper.py
import requests
from bs4 import BeautifulSoup
import re
import csv
import os
from dotenv import load_dotenv

load_dotenv() 

SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")

def scrape_user_ratings(username: str, output_path="data/user_ratings.csv"):
    mal_url = f"https://myanimelist.net/animelist/{username}?status=7&order=4&order2=0"

    scrape_url = f"http://api.scraperapi.com/?api_key={SCRAPER_API_KEY}&url={mal_url}"

    response = requests.get(scrape_url)
    if response.status_code != 200:
        raise Exception("Failed to scrape user profile.")

    soup = BeautifulSoup(response.text, "html.parser")

    # check for private or nonexistent list
    if "This list is private" in soup.text or "404" in soup.title.string:
        raise Exception("Profile is private or does not exist.")

    anime_entries = soup.select("tbody.list-item")
    results = []

    for entry in anime_entries:
        title_tag = entry.select_one("td.data.title a")
        score_tag = entry.select_one("td.data.score span.score-label")
        if title_tag and score_tag:
            score_text = score_tag.text.strip()
            rating = -1 if score_text == "-" else int(score_text)
            match = re.search(r"/anime/(\d+)", title_tag["href"])
            anime_id = int(match.group(1)) if match else None
            if anime_id:
                results.append({
                    "user_id": username,
                    "anime_id": anime_id,
                    "rating": rating
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "anime_id", "rating"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Scraped {len(results)} ratings for user '{username}'")
