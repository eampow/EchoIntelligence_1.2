// src/components/Summary.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Paper, Typography } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

function Summary() {
  const [summary, setSummary] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/summary')
      .then(response => {
        setSummary(response.data);
      })
      .catch(err => {
        console.error("Error fetching summary:", err);
        setError("Error fetching summary: " + (err.response?.data?.error || err.message));
      });
  }, []);

  // Format the summary data into rows for the DataGrid
  const rows = [
    { id: 1, label: "Total Pings", value: summary.totalPings || 'N/A' },
    { id: 2, label: "Average TS", value: summary.averageTS ? summary.averageTS.toFixed(2) : 'N/A' },
    { id: 3, label: "Average Speed", value: summary.averageSpeed ? summary.averageSpeed.toFixed(2) : 'N/A' }
    // You can add more rows as needed.
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>Summary Statistics</Typography>
      {error && <Typography color="error">{error}</Typography>}
      {!error && (
        <div style={{ height: 300, width: '100%' }}>
          <DataGrid
            rows={rows}
            columns={[
              { field: 'label', headerName: 'Metric', width: 200 },
              { field: 'value', headerName: 'Value', width: 200 }
            ]}
            pageSize={5}
            rowsPerPageOptions={[5]}
            disableSelectionOnClick
          />
        </div>
      )}
    </Paper>
  );
}

export default Summary;
