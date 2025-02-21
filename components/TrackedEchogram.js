// File: src/components/TrackedEchogram.js
import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

function TrackedEchogram() {
  const [plotData, setPlotData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/plot/echogram', formData);
      // Extract tracked echogram data from response
      const { trackedEchogram } = response.data;
      setPlotData(trackedEchogram);
    } catch (err) {
      setError('Failed to fetch tracked echogram.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Tracked Echogram</h2>
      <input type="file" onChange={handleFileUpload} />
      {loading && <p>Loading plot...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {plotData && (
        <Plot
          data={plotData.data}
          layout={plotData.layout}
          style={{ width: '100%', height: '400px' }}
        />
      )}
    </div>
  );
}

export default TrackedEchogram;
