// frontend/src/App.jsx
import React, { useState } from 'react';
import './App.css';

export default function App() {
  const [username, setUsername] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setRecommendations([]);
    try {
      const res = await fetch('http://localhost:8000/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Unknown error');
      setRecommendations(data.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">Anime Recommender</h1>
      <form onSubmit={handleSubmit} className="form">
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Enter MyAnimeList username"
          className="input"
        />
        <button type="submit" className="button">Get Recommendations</button>
      </form>

      {loading && <p className="message">Loading recommendations...</p>}
      {error && <p className="error">Error: {error}</p>}

      <div>
        {recommendations.map((anime) => (
          <div key={anime.anime_id} className="card">
            <img src={anime.image_url} alt={anime.name} />
            <div className="card-content">
              <a href={anime.mal_url} target="_blank" rel="noopener noreferrer" className="card-title">
                {anime.name}
              </a>
              <div className="card-meta">
                {anime.genre} | {anime.type} | Score: {anime.score}
              </div>
              <div className="card-description">
                {anime.synopsis || "No description available."}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
