from selenium import webdriver
from selenium.webdriver.chrome.options import Options
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

driver = webdriver.Chrome(options=options)

# --- Load the MAL page ---
driver.get(mal_url)
time.sleep(5)  # initial wait for page load

# --- Scroll to bottom to load all entries ---
SCROLL_PAUSE_TIME = 3
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

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
        score_text = score_tag.text.strip()
        rating = -1 if score_text == "-" else int(score_text)

        match = re.search(r"/anime/(\d+)", title_tag["href"])
        anime_id = int(match.group(1)) if match else None

        if anime_id:
            results.append({
                "user_id": username,  # MAL usernames serve as IDs here
                "anime_id": anime_id,
                "rating": rating
            })

# --- Save to CSV ---
csv_filename = f"user_ratings.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["user_id", "anime_id", "rating"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Scraped {len(results)} anime entries for user '{username}'")
print(f"📁 Saved to: {csv_filename}")
