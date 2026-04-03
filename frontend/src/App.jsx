import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import WasteClassifier from './components/WasteClassifier';
import PredictionResult from './components/PredictionResult';
import HistoryList from './components/HistoryList';
import Dashboard from './components/Dashboard';
import './styles/App.css';

function App() {
  const [currentResult, setCurrentResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [currentImagePreview, setCurrentImagePreview] = useState(null);
  const [currentImageFile, setCurrentImageFile] = useState(null);
  const [view, setView] = useState(() => {
    const path = window.location.pathname;
    if (path === '/dashboard') return 'dashboard';
    return 'home';
  });

  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname;
      if (path === '/dashboard') setView('dashboard');
      else setView('home');
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const navigateTo = (targetView) => {
    const pathMap = {
      dashboard: '/dashboard',
      home: '/'
    };
    const path = pathMap[targetView] || '/';
    
    if (window.location.pathname !== path) {
      window.history.pushState({}, '', path);
    }
    setView(targetView);
  };

  const handlePrediction = (result, imagePreview, imageFile) => {
    setCurrentResult(result);
    setCurrentImagePreview(imagePreview);
    setCurrentImageFile(imageFile);
    
    // Add to history
    const newHistoryItem = {
      predicted_class: result.predicted_class,
      image: imagePreview,
      timestamp: new Date().toLocaleTimeString()
    };

    setHistory(prev => [newHistoryItem, ...prev].slice(0, 10)); // Keep last 10
  };

  return (
    <div className="app">
      <Navbar currentView={view} onNavigate={navigateTo} />
      
      <main className="main-container">
        {view === 'dashboard' ? (
          <Dashboard />
        ) : (
          <>
            <div className="content-grid">
              <div className="classifier-section">
                <WasteClassifier onPrediction={handlePrediction} />
              </div>

              <div className="results-section">
                <PredictionResult
                  result={currentResult}
                  sourceImagePreview={currentImagePreview}
                  sourceImageFile={currentImageFile}
                />
              </div>
            </div>

            <div className="history-section">
              <HistoryList history={history} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
