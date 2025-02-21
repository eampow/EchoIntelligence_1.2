// File: src/components/SpatialPlot.js
import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';
import { Slider, Typography, Box, Paper } from '@mui/material';

function SpatialPlot() {
  const [plotData, setPlotData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);

  useEffect(() => {
    axios
      .get('http://localhost:5000/plot/3d-spatial')
      .then((response) => {
        setPlotData(response.data);
        setLoading(false);
      })
      .catch(() => {
        setError('Failed to fetch 3D spatial plot.');
        setLoading(false);
      });
  }, []);

  const layout = plotData
    ? {
        ...plotData.layout,
        font: { family: 'Roboto, sans-serif', color: '#333' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        scene: {
          ...plotData.layout.scene,
          camera: {
            eye: { x: 1 * zoomLevel, y: 1 * zoomLevel, z: 1.5 * zoomLevel },
          },
        },
      }
    : {};

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6" gutterBottom>
        Interactive 3D Spatial Plot
      </Typography>

      <Box sx={{ width: 300, my: 2 }}>
        <Typography variant="body1" gutterBottom>
          Zoom Level: {zoomLevel.toFixed(1)}
        </Typography>
        <Slider
          value={zoomLevel}
          onChange={(e, newValue) => setZoomLevel(newValue)}
          step={0.1}
          min={0.5}
          max={3}
          valueLabelDisplay="auto"
        />
      </Box>

      {loading && <Typography>Loading 3D plot...</Typography>}
      {error && <Typography color="error">{error}</Typography>}

      {plotData && (
        <Plot
          data={plotData.data}
          layout={layout}
          style={{ width: '100%', height: '500px' }}
        />
      )}
    </Paper>
  );
}

export default SpatialPlot;
