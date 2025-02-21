// File: src/components/RawEchogram.js
import React, { useState } from 'react';
import { Paper, Typography, Box, Button } from '@mui/material';

function RawEchogram() {
  const [fileName, setFileName] = useState('');

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileName(file.name);
    }
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Raw Echogram
      </Typography>
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
        <Button variant="contained" component="label">
          Choose File
          <input type="file" hidden onChange={handleFileSelect} />
        </Button>
        <Typography>{fileName || 'No file chosen'}</Typography>
      </Box>
    </Paper>
  );
}

export default RawEchogram;
