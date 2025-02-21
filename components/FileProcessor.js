// src/components/FileProcessor.js
import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Box, Typography, Button, Paper } from '@mui/material';
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

    axios.post('http://127.0.0.1:5000/process-file', formData)
      .then(response => {
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
      .catch(err => {
        console.error("Error processing file:", err);
        setError("Error processing file: " + (err.response?.data?.error || err.message));
      });
  };

  // Define columns for the DataGrid summary
  // Example columns: TrackID, NumPoints, AvgTS, AvgSpeed, PredSpecies
  const columns = [
    { field: 'TrackID', headerName: 'Track ID', width: 90 },
    { field: 'NumPoints', headerName: 'Num Points', width: 110 },
    { field: 'AvgTS', headerName: 'Avg TS (dB)', width: 120 },
    { field: 'AvgSpeed', headerName: 'Avg Speed', width: 120 },
    { field: 'PredSpecies', headerName: 'Species', width: 150 }
  ];

  // Convert summary array to rows that DataGrid can read
  // DataGrid expects a unique 'id' for each row
  const rows = summary.map((row, index) => ({ id: index, ...row }));

  return (
    <Box sx={{ textAlign: 'center', p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Process RAW File
      </Typography>
      <input type="file" accept=".RAW" onChange={handleFileChange} />
      <Button variant="contained" sx={{ ml: 2 }} onClick={handleProcess}>
        Process File
      </Button>
      {error && <Typography color="error" sx={{ mt: 2 }}>{error}</Typography>}

      {rawPlot && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6">Raw Echogram (Interactive)</Typography>
          <Plot
            data={rawPlot.data}
            layout={rawPlot.layout}
            style={{ width: '100%', height: '400px' }}
            useResizeHandler={true}
          />
        </Paper>
      )}

      {trackedPlot && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6">Tracked Echogram (Colored by Species)</Typography>
          <Plot
            data={trackedPlot.data}
            layout={trackedPlot.layout}
            style={{ width: '100%', height: '400px' }}
            useResizeHandler={true}
          />
        </Paper>
      )}

      {summary.length > 0 && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>Summary of Tracks</Typography>
          <div style={{ height: 400, width: '100%' }}>
            <DataGrid
              rows={rows}
              columns={columns}
              pageSize={5}
              rowsPerPageOptions={[5, 10, 20]}
            />
          </div>
        </Paper>
      )}
    </Box>
  );
}

export default FileProcessor;
