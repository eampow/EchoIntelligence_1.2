// src/components/FileProcessor.js
import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Box, Typography, Button, Paper, Grid } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

function FileProcessor() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [rawPlot, setRawPlot] = useState(null);
  const [trackedPlot, setTrackedPlot] = useState(null);
  const [summary, setSummary] = useState([]);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleProcess = () => {
    if (!selectedFile) {
      alert("No file selected!");
      return;
    }
    const formData = new FormData();
    formData.append('file', selectedFile);

    axios.post('http://127.0.0.1:5000/plot/echogram', formData)
      .then((response) => {
        const data = response.data;
        if (data.error) {
          setError(data.error);
          return;
        }
        setRawPlot(data.rawEchogram);
        setTrackedPlot(data.trackedEchogram);
        setSummary(data.summary || []);
        setError('');
      })
      .catch((err) => {
        console.error("Error processing file:", err);
        setError("Error processing file: " + (err.response?.data?.error || err.message));
      });
  };

  // Define columns for the DataGrid summary table.
  const columns = [
    { field: 'TrackID', headerName: 'Track ID', width: 90 },
    { field: 'NumPoints', headerName: 'Num Points', width: 110 },
    { field: 'AvgTS', headerName: 'Avg TS (dB)', width: 120 },
    { field: 'AvgSpeed', headerName: 'Avg Speed', width: 120 },
    { field: 'PredSpecies', headerName: 'Predicted Species', width: 150 }
  ];

  // Convert summary array to rows that DataGrid requires (each row needs a unique id)
  const rows = summary.map((row, index) => ({ id: index, ...row }));

  return (
    <Box sx={{ textAlign: 'center', p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Process RAW File & View Echograms
      </Typography>
      <input type="file" accept=".RAW" onChange={handleFileChange} />
      <Button variant="contained" sx={{ ml: 2 }} onClick={handleProcess}>
        Process File
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
                useResizeHandler
              />
            </Paper>
          </Grid>
        )}
        {trackedPlot && (
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6">Tracked Echogram (Predicted Species)</Typography>
              <Plot
                data={trackedPlot.data}
                layout={trackedPlot.layout}
                style={{ width: '100%', height: '400px' }}
                useResizeHandler
              />
            </Paper>
          </Grid>
        )}
      </Grid>

      {summary.length > 0 && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>Track Summary</Typography>
          <div style={{ height: 300, width: '100%' }}>
            <DataGrid
              rows={rows}
              columns={columns}
              pageSize={5}
              rowsPerPageOptions={[5, 10, 20]}
              disableSelectionOnClick
            />
          </div>
        </Paper>
      )}
    </Box>
  );
}

export default FileProcessor;
