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

      {recommendations.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>Title</th>
              <th>Genre</th>
              <th>Type</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {recommendations.map((anime) => (
              <tr key={anime.anime_id}>
                <td>{anime.name}</td>
                <td>{anime.genre}</td>
                <td>{anime.type}</td>
                <td>{anime.score.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}