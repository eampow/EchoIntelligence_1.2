// src/components/TrainingVisualizations.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Paper, Typography, Box } from '@mui/material';

function TrainingVisualizations() {
  const [plotImage, setPlotImage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/plot/training-visualization')
      .then(response => {
        if (response.data.image) {
          setPlotImage(response.data.image);
        } else {
          setError("No image returned.");
        }
      })
      .catch(err => {
        console.error("Error fetching training visualization:", err);
        setError("Failed to load training visualization.");
      });
  }, []);

  return (
    <Paper sx={{ p: 2, textAlign: 'center' }}>
      <Typography variant="h6">Training Visualizations</Typography>
      {error && <Typography color="error">{error}</Typography>}
      {plotImage ? (
        <Box sx={{ mt: 2 }}>
          <img src={`data:image/png;base64,${plotImage}`} alt="Training Visualization" style={{ maxWidth: '100%' }} />
        </Box>
      ) : (
        <Typography>Loading training visualization...</Typography>
      )}
    </Paper>
  );
}

export default TrainingVisualizations;
