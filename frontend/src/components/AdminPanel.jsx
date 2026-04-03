import React, { useEffect, useState } from 'react';
import { getDisposalRules, saveDisposalRules } from '../api/client';
import './AdminPanel.css';

const AdminPanel = ({ onNavigateBack }) => {
  const [rules, setRules] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState('');

  // Waste classes we support
  const wasteClasses = ['plastic', 'paper', 'organic', 'glass', 'metal', 'e-waste'];

  useEffect(() => {
    loadRules();
  }, []);

  const loadRules = async () => {
    setLoading(true);
    setError(null);

    try {
      const rulesData = await getDisposalRules();
      setRules(rulesData);
    } catch (err) {
      setError(err.message || 'Failed to load disposal rules');
    } finally {
      setLoading(false);
    }
  };

  const handleFieldChange = (wasteClass, field, value) => {
    setRules((prevRules) => ({
      ...prevRules,
      [wasteClass]: {
        ...prevRules[wasteClass],
        [field]: value
      }
    }));
  };

  const handleSaveAll = async () => {
    setSaving(true);
    setError(null);
    setSuccessMessage('');

    try {
      await saveDisposalRules(rules);
      setSuccessMessage('✓ All disposal rules saved successfully!');
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccessMessage('');
      }, 3000);
    } catch (err) {
      setError(err.message || 'Failed to save disposal rules');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <section className="admin-panel">
        <h2>Admin Panel - Loading...</h2>
      </section>
    );
  }

  return (
    <section className="admin-panel">
      <div className="admin-header">
        <div>
          <h2>🔧 Admin Panel</h2>
          <p className="admin-subtitle">Manage disposal rules and guidelines</p>
        </div>
        {onNavigateBack && (
          <button onClick={onNavigateBack} className="back-button">
            ← Back to Classifier
          </button>
        )}
      </div>

      {error && <div className="admin-error">{error}</div>}
      {successMessage && <div className="admin-success">{successMessage}</div>}

      <div className="rules-grid">
        {wasteClasses.map((wasteClass) => {
          const rule = rules[wasteClass] || { title: '', description: '' };
          return (
            <div key={wasteClass} className="rule-card">
              <h3 className="rule-class-name">{wasteClass.toUpperCase()}</h3>
              
              <div className="form-group">
                <label htmlFor={`${wasteClass}-title`}>Title</label>
                <input
                  id={`${wasteClass}-title`}
                  type="text"
                  value={rule.title || ''}
                  onChange={(e) => handleFieldChange(wasteClass, 'title', e.target.value)}
                  placeholder={`e.g., ${wasteClass.charAt(0).toUpperCase() + wasteClass.slice(1)} (Dry Waste)`}
                  className="rule-input"
                />
              </div>

              <div className="form-group">
                <label htmlFor={`${wasteClass}-description`}>Description</label>
                <textarea
                  id={`${wasteClass}-description`}
                  value={rule.description || ''}
                  onChange={(e) => handleFieldChange(wasteClass, 'description', e.target.value)}
                  placeholder="Enter disposal instructions..."
                  rows={3}
                  className="rule-textarea"
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="admin-actions">
        <button
          onClick={handleSaveAll}
          disabled={saving}
          className="save-button"
        >
          {saving ? '💾 Saving...' : '💾 Save All Changes'}
        </button>
      </div>
    </section>
  );
};

export default AdminPanel;
