// File: src/components/TrainButton.js
import React, { useState } from 'react';
import axios from 'axios';
import Button from '@mui/material/Button';
import Notification from './Notification';

function TrainButton() {
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info',
  });

  const handleClose = () => {
    setNotification({ ...notification, open: false });
  };

  const handleTrain = () => {
    axios
      .post('http://127.0.0.1:5000/train')
      .then((response) => {
        setNotification({
          open: true,
          message: response.data.status || 'Training completed successfully!',
          severity: 'success',
        });
      })
      .catch((err) => {
        console.error('Training error:', err);
        setNotification({
          open: true,
          message:
            'Training failed: ' +
            (err.response?.data?.error || err.message),
          severity: 'error',
        });
      });
  };

  return (
    <div>
      <Button variant="contained" color="primary" onClick={handleTrain}>
        Train Models
      </Button>
      <Notification
        open={notification.open}
        onClose={handleClose}
        message={notification.message}
        severity={notification.severity}
      />
    </div>
  );
}

export default TrainButton;
