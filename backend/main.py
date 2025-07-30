# backend/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import pandas as pd
import os
import traceback
from recommender import hybrid_recommendation
from scraper import scrape_user_ratings

#initialize the FastAPI and add CORS middleware to allow requests from any origin
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[                                     #should be https://anime-recommender-ebon.vercel.app/ for when deployed
        "http://localhost:3000",
        "https://anime-recommender-ebon.vercel.app"
    ],        
    allow_credentials=True,
    allow_methods=["*"],        #allows all HTTP methods like GET, POST, PUT etc
    allow_headers=["*"],
)

#structure POST request to /recommend (JSON with username field)
class UsernameInput(BaseModel):
    username: str

#load the anime data and fill missing values
anime_df = pd.read_csv("data/anime.csv")
anime_df['genre'] = anime_df['genre'].fillna('')
anime_df['type'] = anime_df['type'].fillna('')
anime_df['rating'] = anime_df['rating'].fillna(0).astype(str)
anime_df['members'] = anime_df['members'].fillna(0).astype(int).astype(str)

#create POST /recommend endpoint that takes username input
@app.post("/recommend")
async def recommend(username_input: UsernameInput):
    username = username_input.username
    try:
        print(f"📥 Scraping ratings for: {username}")
        count = scrape_user_ratings(username)
        print(f"✅ Scraped {count} ratings")

        from recommender import run_recommender
        results = run_recommender(username)
        print(f"🎯 Generated {len(results)} recommendations")

        return {"results": results.to_dict(orient="records")}

    except Exception as e:
        print("❌ Exception occurred:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
#add health check route
@app.get("/")
def health_check():
    return {"status": "Backend is alive"}
