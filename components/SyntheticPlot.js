import React, { useEffect, useState } from 'react';
import axios from 'axios';

function SyntheticPlot() {
  const [image, setImage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/plot/synthetic')
      .then(response => {
        if (response.data.image) {
          setImage(response.data.image);
        } else {
          setError("No image returned from backend.");
        }
      })
      .catch(err => {
        console.error("Error fetching synthetic plot:", err);
        setError("Failed to load synthetic plot.");
      });
  }, []);

  return (
    <div>
      <h2>Synthetic Data Plot</h2>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {image ? (
        <img
          src={`data:image/png;base64,${image}`}
          alt="Synthetic Plot"
          style={{ maxWidth: '80%' }}
        />
      ) : (
        <p>Loading plot...</p>
      )}
    </div>
  );
}

export default SyntheticPlot;
