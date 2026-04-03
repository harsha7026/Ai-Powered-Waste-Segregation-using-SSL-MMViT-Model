import React, { useEffect, useMemo, useState } from 'react';
import { getClassDistribution, getStatsSummary } from '../api/client';
import './Dashboard.css';

const Dashboard = () => {
  const [summary, setSummary] = useState(null);
  const [distribution, setDistribution] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadDashboardData = async () => {
      setLoading(true);
      setError(null);

      try {
        const [summaryResponse, distributionResponse] = await Promise.all([
          getStatsSummary(),
          getClassDistribution()
        ]);
        setSummary(summaryResponse);
        setDistribution(distributionResponse || {});
      } catch (fetchError) {
        setError(fetchError.message || 'Failed to load analytics data');
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
  }, []);

  const chartData = useMemo(() => {
    const entries = Object.entries(distribution || {});
    const maxValue = entries.reduce((max, [, value]) => Math.max(max, value), 0);

    return entries.map(([label, value]) => ({
      label,
      value,
      width: maxValue > 0 ? (value / maxValue) * 100 : 0
    }));
  }, [distribution]);

  if (loading) {
    return (
      <section className="dashboard-card">
        <h2>Analytics Dashboard</h2>
        <p>Loading analytics...</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="dashboard-card">
        <h2>Analytics Dashboard</h2>
        <p className="dashboard-error">{error}</p>
      </section>
    );
  }

  const totalPredictions = summary?.total_predictions || 0;
  const averageConfidence = Math.round((summary?.avg_confidence || 0) * 100);
  const lastPredictionText = summary?.last_prediction_time
    ? new Date(summary.last_prediction_time).toLocaleString()
    : 'No predictions yet';

  return (
    <section className="dashboard-card">
      <h2>Analytics Dashboard</h2>

      <div className="dashboard-section">
        <h3>Usage Summary</h3>
        <div className="summary-grid">
          <div className="summary-item">
            <span className="summary-label">Total Predictions</span>
            <span className="summary-value">{totalPredictions}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Avg. Confidence</span>
            <span className="summary-value">{averageConfidence}%</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Last Prediction</span>
            <span className="summary-value summary-small">{lastPredictionText}</span>
          </div>
        </div>
      </div>

      <div className="dashboard-section">
        <h3>Class Distribution</h3>
        {chartData.length === 0 ? (
          <p>No predictions available yet.</p>
        ) : (
          <div className="distribution-chart">
            {chartData.map((item) => (
              <div key={item.label} className="chart-row">
                <span className="chart-label">{item.label}</span>
                <div className="chart-bar-track">
                  <div className="chart-bar" style={{ width: `${item.width}%` }} />
                </div>
                <span className="chart-value">{item.value}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
};

export default Dashboard;
