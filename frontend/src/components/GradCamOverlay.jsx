import React from 'react';
import './GradCamOverlay.css';

const GradCamOverlay = ({ originalImage, heatmap, showHeatmap }) => {
  if (!originalImage) {
    return null;
  }

  return (
    <div className="grad-cam-overlay">
      <img src={originalImage} alt="Original waste sample" className="grad-cam-base-image" />
      {showHeatmap && heatmap ? (
        <img
          src={`data:image/png;base64,${heatmap}`}
          alt="Grad-CAM attention map"
          className="grad-cam-heatmap-image"
        />
      ) : null}
    </div>
  );
};

export default GradCamOverlay;
