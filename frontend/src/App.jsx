// frontend/src/App.jsx
import React, { useState } from 'react';
import './App.css';

//App compoenent is the root of the frontend
export default function App() {
  //state variables
  const [username, setUsername] = useState('');                   //input text from user
  const [recommendations, setRecommendations] = useState([]);     //results from backend
  const [loading, setLoading] = useState(false);                  //loading for in progress
  const [error, setError] = useState(null);                       //display error 

  //submits the form, ensures form doesnt reset page and resets all for new request
  const handleSubmit = async (e) => {
    e.preventDefault();       //prevent page reload
    setLoading(true);         //show loading
    setError(null);           //clear previous errors
    setRecommendations([]);   //clear previous results
    try {
      // const res = await fetch('http://localhost:8000/recommend', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ username })
      // });
      // const res = await fetch('https://anime-recommender-api-cbif.onrender.com/recommend', {  //send POST FastAPI backend with username
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ username })
      // });
      const res = await fetch('https://animerecommender-production.up.railway.app/recommend', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username })
      });

      //wait for api response and parse it in JSON format then save results in state
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
      {/*ttile header, username input box and button to call handlesubmit*/}
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

      {/*displays loading and errors*/}
      {loading && <p className="message">Loading recommendations...</p>}
      {error && <p className="error">Error: {error}</p>}

      <div>
        {/*each recommendation becomes a card to show image, links to page and has metadata*/}
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
