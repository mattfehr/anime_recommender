import requests
from bs4 import BeautifulSoup
import re
import csv
import os
import base64
from dotenv import load_dotenv

load_dotenv()
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")

def scrape_user_ratings(username: str, output_path="data/user_ratings.csv"):
    if not SCRAPINGBEE_API_KEY:
        raise Exception("Missing SCRAPINGBEE_API_KEY")

    target_url = f"https://myanimelist.net/animelist/{username}?status=7"

    # JavaScript to scroll and load all lazy-loaded list items
    scroll_script = """
    (async () => {
        let totalHeight = 0;
        const distance = 100;
        const delay = ms => new Promise(res => setTimeout(res, ms));
        while (totalHeight < document.body.scrollHeight) {
            window.scrollBy(0, distance);
            await delay(200);
            totalHeight += distance;
        }
    })();
    """

    encoded_script = base64.b64encode(scroll_script.encode("utf-8")).decode("utf-8")

    api_url = "https://app.scrapingbee.com/api/v1/"
    params = {
        "api_key": SCRAPINGBEE_API_KEY,
        "url": target_url,
        "render_js": "true",
        "js_snippet": encoded_script
    }

    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        raise Exception(f"ScrapingBee error: {response.status_code} - {response.text}")

    soup = BeautifulSoup(response.text, "html.parser")

    if "This list is private" in soup.text or "404" in soup.title.string:
        raise Exception("Profile is private or does not exist.")

    anime_entries = soup.select("tbody.list-item")
    results = []

    for entry in anime_entries:
        title_tag = entry.select_one("td.data.title a")
        score_tag = entry.select_one("td.data.score span.score-label")

        if title_tag and score_tag:
            score_text = score_tag.text.strip()

            try:
                rating = -1 if score_text == "-" else int(score_text)
            except ValueError:
                print(f"⚠️ Skipping invalid rating: {score_text}")
                continue

            match = re.search(r"/anime/(\d+)", title_tag["href"])
            anime_id = int(match.group(1)) if match else None

            if anime_id:
                results.append({
                    "user_id": username,
                    "anime_id": anime_id,
                    "rating": rating
                })

    if not results:
        raise Exception("No valid ratings found — user may have an empty list or scraping failed.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "anime_id", "rating"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Scraped {len(results)} ratings for user '{username}'")
    return len(results)
