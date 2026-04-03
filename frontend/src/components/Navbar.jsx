import React from 'react';
import './Navbar.css';

const Navbar = ({ currentView, onNavigate }) => {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div>
          <h1 className="navbar-title">AI Waste Segregation</h1>
          <p className="navbar-subtitle">
            Self-Supervised MMViT for Smart Waste Classification
          </p>
        </div>

        <div className="navbar-actions">
          <button
            className={`nav-button ${currentView === 'home' ? 'active' : ''}`}
            onClick={() => onNavigate('home')}
          >
            🔍 Classifier
          </button>
          <button
            className={`nav-button ${currentView === 'dashboard' ? 'active' : ''}`}
            onClick={() => onNavigate('dashboard')}
          >
            📊 Analytics
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
