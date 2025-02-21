// File: src/components/TrackSummary.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';

function TrackSummary() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios
      .get('http://localhost:5000/summary')
      .then((response) => {
        setSummary(response.data.summary);
        setLoading(false);
      })
      .catch((err) => {
        setError('Failed to fetch summary.');
        setLoading(false);
      });
  }, []);

  return (
    <div>
      <h2>Track Summary</h2>
      {loading && <p>Loading summary...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {summary && (
        <div>
          <p>Total Pings: {summary.totalPings}</p>
          <p>Average TS: {summary.averageTS}</p>
          <p>Average Speed: {summary.averageSpeed}</p>
          <h3>Species Distribution:</h3>
          <ul>
            {summary.speciesDistribution &&
              Object.entries(summary.speciesDistribution).map(([species, count]) => (
                <li key={species}>
                  {species}: {count}
                </li>
              ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default TrackSummary;
