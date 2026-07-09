import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Dashboard from './Dashboard';
import Results from './Results';

// A small helper component to highlight the active tab
function NavigationBar() {
  const location = useLocation();
  
  const getStyle = (path) => ({
    textDecoration: 'none',
    fontSize: '1.2rem',
    fontWeight: 'bold',
    color: location.pathname === path ? '#0070f3' : '#666',
    borderBottom: location.pathname === path ? '3px solid #0070f3' : 'none',
    paddingBottom: '0.5rem'
  });

  return (
    <nav style={{ display: 'flex', gap: '2rem', marginBottom: '2rem', borderBottom: '2px solid #eee' }}>
      <Link to="/" style={getStyle('/')}>Control Panel</Link>
      <Link to="/results" style={getStyle('/results')}>Benchmark Results</Link>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div style={{ maxWidth: '1000px', margin: '0 auto', padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
        <NavigationBar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/results" element={<Results />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;