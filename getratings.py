from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import re
import csv

# --- User input ---
username = input("Enter MyAnimeList username: ").strip()
mal_url = f"https://myanimelist.net/animelist/{username}"

# --- Set up Selenium ---
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Adjust path to chromedriver if necessary
driver = webdriver.Chrome(options=options)

# --- Load the MAL page ---
driver.get(mal_url)
time.sleep(5)  # wait for JS to render the anime list

# --- Parse HTML with BeautifulSoup ---
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# --- Extract anime data ---
anime_entries = soup.select("tbody.list-item")
results = []

for entry in anime_entries:
    title_tag = entry.select_one("td.data.title a")
    score_tag = entry.select_one("td.data.score span.score-label")

    if title_tag and score_tag:
        title = title_tag.text.strip()
        url = "https://myanimelist.net" + title_tag["href"]
        score = score_tag.text.strip()

        # Extract anime ID from URL: /anime/31646/3-gatsu_no_Lion
        match = re.search(r"/anime/(\d+)", title_tag["href"])
        anime_id = int(match.group(1)) if match else None

        results.append({
            "user_name": username,
            "anime_id": anime_id,
            "title": title,
            "url": url,
            "score": score
        })

# --- Save to CSV ---
csv_filename = f"user_ratings.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["user_name", "anime_id", "title", "url", "score"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Scraped {len(results)} anime entries for user '{username}'")
print(f"📁 Saved to: {csv_filename}")
