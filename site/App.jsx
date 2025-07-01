// frontend/src/App.jsx
import React, { useState } from 'react';

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
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-6">
      <h1 className="text-3xl font-bold mb-6">Anime Recommender</h1>
      <form onSubmit={handleSubmit} className="flex gap-4 mb-6">
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Enter MyAnimeList username"
          className="px-4 py-2 rounded border border-gray-300 shadow-sm"
        />
        <button
          type="submit"
          className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Get Recommendations
        </button>
      </form>

      {loading && <p>Loading recommendations...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}

      <div className="w-full max-w-3xl">
        {recommendations.length > 0 && (
          <table className="w-full bg-white shadow rounded">
            <thead>
              <tr className="bg-gray-200 text-left">
                <th className="p-3">Title</th>
                <th className="p-3">Genre</th>
                <th className="p-3">Type</th>
                <th className="p-3">Score</th>
              </tr>
            </thead>
            <tbody>
              {recommendations.map((anime) => (
                <tr key={anime.anime_id} className="border-t">
                  <td className="p-3 font-medium">{anime.name}</td>
                  <td className="p-3">{anime.genre}</td>
                  <td className="p-3">{anime.type}</td>
                  <td className="p-3">{anime.score.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
