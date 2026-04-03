import React, { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import { predictWaste } from '../api/client';
import './WasteClassifier.css';

const WasteClassifier = ({ onPrediction }) => {
  const [mode, setMode] = useState('upload'); // 'upload' or 'webcam'
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const captureWebcam = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      setImagePreview(imageSrc);
      // Convert base64 to file
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });
          setSelectedImage(file);
          setError(null);
        });
    }
  };

  const handleClassify = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await predictWaste(selectedImage);
      onPrediction(result, imagePreview, selectedImage);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="waste-classifier">
      <div className="classifier-header">
        <h2>Classify Waste Item</h2>
        <div className="mode-selector">
          <button
            className={mode === 'upload' ? 'active' : ''}
            onClick={() => setMode('upload')}
          >
            📁 Upload Image
          </button>
          <button
            className={mode === 'webcam' ? 'active' : ''}
            onClick={() => setMode('webcam')}
          >
            📷 Webcam
          </button>
        </div>
      </div>

      <div className="classifier-content">
        {mode === 'upload' ? (
          <div className="upload-section">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
              id="file-input"
            />
            <label htmlFor="file-input" className="upload-button">
              Choose Image
            </label>
          </div>
        ) : (
          <div className="webcam-section">
            {!imagePreview ? (
              <>
                <Webcam
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  className="webcam-view"
                />
                <button onClick={captureWebcam} className="capture-button">
                  📸 Capture Photo
                </button>
              </>
            ) : null}
          </div>
        )}

        {imagePreview && (
          <div className="preview-section">
            <h3>Image Preview</h3>
            <img src={imagePreview} alt="Preview" className="image-preview" />
            <div className="action-buttons">
              <button
                onClick={handleClassify}
                disabled={loading}
                className="classify-button"
              >
                {loading ? '⏳ Classifying...' : '🔍 Classify'}
              </button>
              <button onClick={handleReset} className="reset-button">
                🔄 Reset
              </button>
            </div>
          </div>
        )}

        {error && <div className="error-message">❌ {error}</div>}
      </div>
    </div>
  );
};

export default WasteClassifier;
