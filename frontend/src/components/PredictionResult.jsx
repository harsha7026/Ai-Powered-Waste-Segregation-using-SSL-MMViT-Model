import React, { useEffect, useState } from 'react';
import { generateGradCam, getDisposalRules } from '../api/client';
import GradCamOverlay from './GradCamOverlay';
import './PredictionResult.css';

const PredictionResult = ({ result, sourceImagePreview, sourceImageFile }) => {
  const [gradCamData, setGradCamData] = useState(null);
  const [showAttentionMap, setShowAttentionMap] = useState(true);
  const [loadingAttentionMap, setLoadingAttentionMap] = useState(false);
  const [attentionMapError, setAttentionMapError] = useState(null);
  const [disposalRules, setDisposalRules] = useState({});
  const [rulesLoaded, setRulesLoaded] = useState(false);

  // Load disposal rules on mount
  useEffect(() => {
    const loadRules = async () => {
      try {
        const rules = await getDisposalRules();
        setDisposalRules(rules);
        setRulesLoaded(true);
      } catch (error) {
        console.error('Failed to load disposal rules:', error);
        // Use fallback rules
        setDisposalRules({});
        setRulesLoaded(true);
      }
    };

    loadRules();
  }, []);

  useEffect(() => {
    setGradCamData(null);
    setShowAttentionMap(true);
    setAttentionMapError(null);
  }, [result, sourceImagePreview]);

  if (!result) {
    return (
      <div className="prediction-result empty">
        <p>Upload an image to see classification results</p>
      </div>
    );
  }

  const { predicted_class, probabilities } = result;
  
  // Get disposal info from dynamic rules or fallback
  const disposalInfo = disposalRules[predicted_class] || {
    title: 'General Waste Guidance',
    description: 'Handle with care and check your local municipality guidelines for correct disposal.'
  };

  const classIcons = {
    organic: '🍎',
    plastic: '♻️',
    paper: '📄',
    metal: '🔩',
    glass: '🍶'
  };

  const classColors = {
    organic: '#4caf50',
    plastic: '#2196f3',
    paper: '#ff9800',
    metal: '#9e9e9e',
    glass: '#00acc1'
  };

  const handleViewAttentionMap = async () => {
    if (!sourceImageFile) {
      setAttentionMapError('Source image is not available for Grad-CAM. Please classify again.');
      return;
    }

    setLoadingAttentionMap(true);
    setAttentionMapError(null);

    try {
      const response = await generateGradCam(sourceImageFile);
      setGradCamData(response);
      setShowAttentionMap(true);
    } catch (error) {
      setAttentionMapError(error.message);
    } finally {
      setLoadingAttentionMap(false);
    }
  };

  return (
    <div className="prediction-result">
      <h2>Classification Result</h2>
      
      <div className="predicted-class" style={{ borderColor: classColors[predicted_class] }}>
        <span className="class-icon">{classIcons[predicted_class] || '📦'}</span>
        <span className="class-name">{predicted_class.toUpperCase()}</span>
      </div>

      <div className="probabilities">
        <h3>Confidence Scores</h3>
        {Object.entries(probabilities).map(([className, probability]) => (
          <div key={className} className="probability-item">
            <div className="probability-label">
              <span>{classIcons[className]} {className}</span>
              <span className="probability-value">
                {(probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="probability-bar-container">
              <div
                className="probability-bar"
                style={{
                  width: `${probability * 100}%`,
                  backgroundColor: classColors[className]
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="disposal-card">
        <h3>Disposal Recommendation</h3>
        <p className="disposal-title">{disposalInfo.title}</p>
        <p className="disposal-body">{disposalInfo.description}</p>
      </div>

      <div className="attention-map-section">
        <div className="attention-map-header">
          <h3>Model Explainability</h3>
          <button onClick={handleViewAttentionMap} disabled={loadingAttentionMap} className="attention-button">
            {loadingAttentionMap ? 'Generating...' : 'View Attention Map'}
          </button>
        </div>

        {attentionMapError ? <p className="attention-error">{attentionMapError}</p> : null}

        {gradCamData && sourceImagePreview ? (
          <>
            <GradCamOverlay
              originalImage={sourceImagePreview}
              heatmap={gradCamData.heatmap}
              showHeatmap={showAttentionMap}
            />
            <div className="attention-controls">
              <label>
                <input
                  type="checkbox"
                  checked={showAttentionMap}
                  onChange={(event) => setShowAttentionMap(event.target.checked)}
                />
                Show heatmap overlay
              </label>
              <p>
                Focus class: <strong>{gradCamData.predicted_class}</strong> | Confidence:{' '}
                <strong>{(gradCamData.confidence * 100).toFixed(1)}%</strong>
              </p>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
};

export default PredictionResult;
