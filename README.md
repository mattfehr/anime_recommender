# 🎌 Anime Recommender

A hybrid anime recommendation system that combines collaborative filtering and content-based filtering to generate personalized recommendations for anime fans.

🔗 **Web Deployment**: [anime-recommender-ebon.vercel.app](https://anime-recommender-ebon.vercel.app/)
- Note this was taken down

🔗 **Live Demo**: [YouTube Local Run Demo](https://youtu.be/LMskEjUUIP0)
- Note this was taken down

---

## 🚀 Features

- **Hybrid Recommendation Engine**:
  - **Collaborative Filtering** using SVD based on MyAnimeList user ratings
  - **Content-Based Filtering** using TF-IDF and cosine similarity on anime descriptions and metadata
- **Interactive Web Interface** to explore and receive anime suggestions
- **FastAPI Backend** serving recommendations through a REST API
- **Drag-and-Drop UI** for user interaction and dynamic filtering

---

## 🛠️ Tech Stack

- **Frontend**: React.js
- **Backend**: Python, FastAPI, scikit-learn, pandas, NumPy
- **Modeling**: SVD, TF-IDF Vectorizer, Cosine Similarity
- **Deployment**: Vercel (Frontend), Render/Localhost (Backend)

---

## ⚙️ Getting Started
To run locally, clone the repository and run the respective commands below in /frontend and /backed. Use cd (backend or frontend) in bash

### 🖥️ Frontend

```bash
cd client
npm install
npm start
```

### 🧠 Backend

```bash
cd server
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

---

## 📂 Project Structure

```
anime_recommender/
│
├── backend/
│   ├── __pycache__/
│   ├── data/
│   │   ├── anime.csv
│   │   ├── rating.csv
│   │   └── user_ratings.csv
│   ├── main.py
│   ├── recommender.py
│   ├── requirements.txt
│   ├── runtime.txt
│   └── scraper.py
│
├── frontend/
│   ├── node_modules/
│   ├── public/
│   └── src/
│       ├── App.css
│       ├── App.jsx
```
- note the rest is unimportant or was used for initial testing

---

## 📈 How It Works

1. **Collaborative Filtering** learns latent features from user–anime interactions using SVD.
2. **Content-Based Filtering** computes similarities based on anime synopses and genres.
3. Hybrid logic merges both scores to generate final ranked recommendations.

---

## 📦 Requirements

- Python 3.8+
- Node.js 16+
- Dependencies listed in `requirements.txt` and `package.json`

