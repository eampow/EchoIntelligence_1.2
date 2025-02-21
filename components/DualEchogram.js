// src/components/DualEchogram.js
import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Box, Typography, Button, Paper, Grid } from '@mui/material';

function DualEchogram() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [rawPlot, setRawPlot] = useState(null);
  const [trackedPlot, setTrackedPlot] = useState(null);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = () => {
    if (!selectedFile) {
      alert("No file selected!");
      return;
    }
    const formData = new FormData();
    formData.append('file', selectedFile);

    axios.post('http://127.0.0.1:5000/plot/echogram', formData)
      .then(response => {
        const data = response.data;
        if (data.error) {
          setError(data.error);
          return;
        }
        setRawPlot(data.rawEchogram);
        setTrackedPlot(data.trackedEchogram);
        setError('');
      })
      .catch(err => {
        console.error("Error generating echograms:", err);
        setError("Error generating echograms: " + (err.response?.data?.error || err.message));
      });
  };

  return (
    <Box sx={{ textAlign: 'center', p: 2 }}>
      <Typography variant="h4" gutterBottom>Dual Echogram</Typography>
      <input type="file" accept=".RAW" onChange={handleFileChange} />
      <Button variant="contained" sx={{ ml: 2 }} onClick={handleUpload}>
        Upload & Generate
      </Button>
      {error && <Typography color="error" sx={{ mt: 2 }}>{error}</Typography>}
      
      <Grid container spacing={2} sx={{ mt: 2 }}>
        {rawPlot && (
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6">Raw Echogram</Typography>
              <Plot
                data={rawPlot.data}
                layout={rawPlot.layout}
                style={{ width: '100%', height: '400px' }}
                useResizeHandler={true}
              />
            </Paper>
          </Grid>
        )}
        {trackedPlot && (
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6">Tracked Echogram</Typography>
              <Plot
                data={trackedPlot.data}
                layout={trackedPlot.layout}
                style={{ width: '100%', height: '400px' }}
                useResizeHandler={true}
              />
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default DualEchogram;
