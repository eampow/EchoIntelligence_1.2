import React, { useState } from 'react';
import axios from 'axios';

function EchogramPlot() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [image, setImage] = useState('');
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
        if (response.data.image) {
          setImage(response.data.image);
          setError('');
        } else {
          setError(response.data.error || 'Unknown error occurred.');
          setImage('');
        }
      })
      .catch(err => {
        console.error("Echogram generation error:", err);
        setError("Echogram generation failed: " + (err.response?.data?.error || err.message));
        setImage('');
      });
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <h2>Echogram from RAW File</h2>
      <input type="file" accept=".RAW" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginLeft: '10px' }}>
        Upload & Generate Echogram
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {image && (
        <div style={{ marginTop: '20px' }}>
          <img
            src={`data:image/png;base64,${image}`}
            alt="Echogram"
            style={{ maxWidth: '80%', border: '1px solid #ccc' }}
          />
        </div>
      )}
    </div>
  );
}

export default EchogramPlot;
