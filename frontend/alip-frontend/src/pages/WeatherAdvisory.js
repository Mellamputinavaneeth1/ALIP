import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getWeatherAdvisory } from '../services/api';
import toast from 'react-hot-toast';

export default function WeatherAdvisory() {
  const navigate            = useNavigate();
  const [location, setLocation] = useState('');
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);

  const handleSubmit = async () => {
    if (!location) {
      toast.error('Please enter a location!');
      return;
    }
    setLoading(true);
    try {
      const res = await getWeatherAdvisory(location);
      setResult(res.data);
      toast.success('Weather data loaded!');
    } catch (err) {
      toast.error('Location not found. Try another city!');
    }
    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <button style={styles.backBtn} onClick={() => navigate('/farmer')}>
          ← Back
        </button>
        <h1 style={styles.headerTitle}>🌦️ Weather Advisory</h1>
      </div>

      <div style={styles.content}>
        {/* Search */}
        <div style={styles.searchCard}>
          <h2 style={styles.cardTitle}>Enter Your Location</h2>
          <div style={styles.searchRow}>
            <input
              style={styles.input}
              placeholder="e.g. Hyderabad, Chennai, Mumbai..."
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
            />
            <button
              style={{...styles.searchBtn, opacity: loading ? 0.7 : 1}}
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? '⏳' : '🔍 Get Advisory'}
            </button>
          </div>
        </div>

        {/* Results */}
        {result && (
          <>
            {/* Weather Stats */}
            <div style={styles.weatherGrid}>
              {[
                { icon: '🌡️', label: 'Temperature', value: `${result.weather.temperature}°C` },
                { icon: '💧', label: 'Humidity',    value: `${result.weather.humidity}%` },
                { icon: '💨', label: 'Wind Speed',  value: `${result.weather.wind_speed.toFixed(1)} km/h` },
                { icon: '🌤️', label: 'Condition',   value: result.weather.condition },
                { icon: '🌧️', label: 'Rainfall',    value: `${result.weather.rainfall}mm` },
                { icon: '📍', label: 'Location',    value: result.weather.location }
              ].map((item, idx) => (
                <div key={idx} style={styles.weatherCard}>
                  <span style={styles.weatherIcon}>{item.icon}</span>
                  <p style={styles.weatherValue}>{item.value}</p>
                  <p style={styles.weatherLabel}>{item.label}</p>
                </div>
              ))}
            </div>

            {/* Advisories */}
            <div style={styles.advisoryCard}>
              <h2 style={styles.cardTitle}>
                🌾 Farming Advisories ({result.total_advisories})
              </h2>
              {result.advisories.map((advisory, idx) => (
                <div key={idx} style={styles.advisoryItem}>
                  <p style={styles.advisoryText}>
                    {typeof advisory === 'string' ? advisory : advisory.message}
                  </p>
                  {typeof advisory !== 'string' && (
                    <p style={styles.advisoryMeta}>
                      {advisory.severity} severity
                      {advisory.confidence !== undefined && ` • ${advisory.confidence}% confidence`}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: { minHeight: '100vh', background: '#f5f6fa' },
  header: {
    background:  'linear-gradient(135deg, #1a472a, #2d6a4f)',
    padding:     '16px 32px',
    display:     'flex',
    alignItems:  'center',
    gap:         '16px'
  },
  backBtn: {
    padding: '8px 16px', background: 'rgba(255,255,255,0.2)',
    color: 'white', border: 'none', borderRadius: '8px',
    cursor: 'pointer', fontSize: '14px'
  },
  headerTitle: { color: 'white', margin: '0', fontSize: '22px' },
  content:     { padding: '24px 32px' },
  searchCard: {
    background: 'white', borderRadius: '16px',
    padding: '24px', marginBottom: '24px',
    boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
  },
  cardTitle: { fontSize: '20px', fontWeight: 'bold', color: '#2c3e50', margin: '0 0 16px' },
  searchRow: { display: 'flex', gap: '12px' },
  input: {
    flex: 1, padding: '12px 16px',
    border: '2px solid #e0e0e0', borderRadius: '8px',
    fontSize: '15px', outline: 'none'
  },
  searchBtn: {
    padding: '12px 24px',
    background: 'linear-gradient(135deg, #3498db, #2980b9)',
    color: 'white', border: 'none', borderRadius: '8px',
    fontSize: '15px', fontWeight: 'bold', cursor: 'pointer'
  },
  weatherGrid: {
    display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '16px', marginBottom: '24px'
  },
  weatherCard: {
    background: 'white', borderRadius: '12px',
    padding: '20px', textAlign: 'center',
    boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
  },
  weatherIcon:  { fontSize: '32px', display: 'block', marginBottom: '8px' },
  weatherValue: { fontSize: '18px', fontWeight: 'bold', color: '#2c3e50', margin: '0 0 4px' },
  weatherLabel: { fontSize: '12px', color: '#666', margin: '0' },
  advisoryCard: {
    background: 'white', borderRadius: '16px',
    padding: '24px', boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
  },
  advisoryItem: {
    background: '#f0fff4', borderRadius: '8px',
    padding: '12px 16px', marginBottom: '10px',
    borderLeft: '4px solid #2d6a4f'
  },
  advisoryText: { margin: '0', color: '#2c3e50', fontSize: '14px' },
  advisoryMeta: { margin: '6px 0 0', color: '#666', fontSize: '12px', textTransform: 'capitalize' }
};
