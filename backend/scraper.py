# backend/scraper.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import re
import csv
import os

#function to get a MAL username, go to their ratings page and webscrape their ratings
def scrape_user_ratings(username: str, output_path="data/user_ratings.csv"):
    mal_url = f"https://myanimelist.net/animelist/{username}"           #MAL uses this url format for everyone

    #configures chrome to run headless with no GUI
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)

    #opens the url and wait for initial page load
    driver.get(mal_url)
    time.sleep(5)

    #scroll until all content is loaded, MAL lists are lazy loaded
    SCROLL_PAUSE_TIME = 3
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    #soup gets the full html page and closes the browser
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    #stop if account list is private or errors out
    if "This list is private" in soup.text or "404" in soup.title.string:
        raise Exception("Profile is private or does not exist.")

    #select each anime entry row with CSS selector
    anime_entries = soup.select("tbody.list-item")
    results = []

    #for every entry get the title link and score
    for entry in anime_entries:
        title_tag = entry.select_one("td.data.title a")
        score_tag = entry.select_one("td.data.score span.score-label")
        if title_tag and score_tag:
            score_text = score_tag.text.strip()
            rating = -1 if score_text == "-" else int(score_text)         #if unrated deault to -1
            match = re.search(r"/anime/(\d+)", title_tag["href"])         #extract the anime id from the title link url
            anime_id = int(match.group(1)) if match else None
            if anime_id:
                results.append({
                    "user_id": username,
                    "anime_id": anime_id,
                    "rating": rating
                })

    #write the output to a users rating csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "anime_id", "rating"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Scraped {len(results)} ratings for user '{username}'")
