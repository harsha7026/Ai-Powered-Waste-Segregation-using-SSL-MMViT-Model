import React from 'react';
import './HistoryList.css';

const HistoryList = ({ history }) => {
  if (!history || history.length === 0) {
    return (
      <div className="history-list empty">
        <h3>Recent Predictions</h3>
        <p>No predictions yet</p>
      </div>
    );
  }

  const classIcons = {
    organic: '🍎',
    plastic: '♻️',
    paper: '📄',
    metal: '🔩'
  };

  return (
    <div className="history-list">
      <h3>Recent Predictions ({history.length})</h3>
      <div className="history-items">
        {history.map((item, index) => (
          <div key={index} className="history-item">
            <img src={item.image} alt={`Prediction ${index + 1}`} className="history-thumbnail" />
            <div className="history-info">
              <span className="history-class">
                {classIcons[item.predicted_class]} {item.predicted_class}
              </span>
              <span className="history-time">{item.timestamp}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HistoryList;
