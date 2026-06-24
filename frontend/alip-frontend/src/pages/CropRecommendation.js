import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getCropRecommendation } from '../services/api';
import toast from 'react-hot-toast';

export default function CropRecommendation() {
  const navigate          = useNavigate();
  const [form, setForm]   = useState({ location: '', season: 'kharif' });
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!form.location) {
      toast.error('Please enter your location!');
      return;
    }
    setLoading(true);
    try {
      const res = await getCropRecommendation(form);
      setResult(res.data);
      toast.success('Recommendation ready!');
    } catch (err) {
      toast.error('Failed to get recommendation!');
    }
    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <button style={styles.backBtn} onClick={() => navigate('/farmer')}>
          ← Back
        </button>
        <h1 style={styles.headerTitle}>🌱 Crop Recommendation</h1>
      </div>

      <div style={styles.content}>
        {/* Input Card */}
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>Get Crop Recommendation</h2>
          <p style={styles.cardDesc}>
            Enter your location and season to get the best crop suggestion
          </p>

          <div style={styles.inputGroup}>
            <label style={styles.label}>📍 Your Location</label>
            <input
              style={styles.input}
              placeholder="e.g. Hyderabad, Punjab, Kerala..."
              value={form.location}
              onChange={(e) => setForm({...form, location: e.target.value})}
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>🗓️ Growing Season</label>
            <div style={styles.seasonGrid}>
              {[
                { value: 'kharif', label: 'Kharif', icon: '🌧️', desc: 'June - Nov' },
                { value: 'rabi',   label: 'Rabi',   icon: '❄️', desc: 'Nov - Apr' },
                { value: 'zaid',   label: 'Zaid',   icon: '☀️', desc: 'Apr - Jun' }
              ].map(season => (
                <div
                  key={season.value}
                  style={{
                    ...styles.seasonCard,
                    border: form.season === season.value
                      ? '2px solid #2d6a4f'
                      : '2px solid #e0e0e0',
                    background: form.season === season.value
                      ? '#f0fff4' : 'white'
                  }}
                  onClick={() => setForm({...form, season: season.value})}
                >
                  <span style={styles.seasonIcon}>{season.icon}</span>
                  <p style={styles.seasonLabel}>{season.label}</p>
                  <p style={styles.seasonDesc}>{season.desc}</p>
                </div>
              ))}
            </div>
          </div>

          <button
            style={{...styles.submitBtn, opacity: loading ? 0.7 : 1}}
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? '⏳ Analyzing...' : '🌱 Get Recommendation'}
          </button>
        </div>

        {/* Results */}
        {result && (
          <div style={styles.resultCard}>
            <h2 style={styles.cardTitle}>🎯 Recommendation Results</h2>

            {/* Soil Info */}
            <div style={styles.soilBox}>
              <h3 style={styles.soilTitle}>🪱 Soil Information</h3>
              <div style={styles.soilGrid}>
                {[
                  { label: 'Soil Type',    value: result.soil_type },
                  { label: 'Nitrogen',     value: result.soil_summary.nitrogen },
                  { label: 'Phosphorous',  value: result.soil_summary.phosphorous },
                  { label: 'Potassium',    value: result.soil_summary.potassium },
                  { label: 'pH Level',     value: result.soil_summary.ph }
                ].map((item, idx) => (
                  <div key={idx} style={styles.soilItem}>
                    <p style={styles.soilValue}>{item.value}</p>
                    <p style={styles.soilLabel}>{item.label}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Recommendations */}
            <h3 style={styles.recTitle}>🏆 Top Crop Recommendations</h3>
            {result.recommendations.map((rec, idx) => (
              <div
                key={idx}
                style={{
                  ...styles.recCard,
                  border: idx === 0
                    ? '2px solid #27ae60'
                    : '2px solid #e0e0e0',
                  background: idx === 0 ? '#f0fff4' : 'white'
                }}
              >
                <div style={styles.recLeft}>
                  <span style={styles.recRank}>#{idx + 1}</span>
                  <div>
                    <h4 style={styles.recCrop}>
                      {idx === 0 ? '🥇' : idx === 1 ? '🥈' : '🥉'}{' '}
                      {rec.crop.charAt(0).toUpperCase() + rec.crop.slice(1)}
                    </h4>
                    <p style={styles.recDesc}>{rec.description}</p>
                  </div>
                </div>
                <div style={styles.recConfidence}>
                  <p style={styles.confValue}>{rec.confidence}%</p>
                  <p style={styles.confLabel}>Match</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: { minHeight: '100vh', background: '#f5f6fa' },
  header: {
    background: 'linear-gradient(135deg, #1a472a, #2d6a4f)',
    padding: '16px 32px', display: 'flex',
    alignItems: 'center', gap: '16px'
  },
  backBtn: {
    padding: '8px 16px', background: 'rgba(255,255,255,0.2)',
    color: 'white', border: 'none', borderRadius: '8px',
    cursor: 'pointer', fontSize: '14px'
  },
  headerTitle: { color: 'white', margin: '0', fontSize: '22px' },
  content: {
    display: 'grid', gridTemplateColumns: '1fr 1fr',
    gap: '24px', padding: '24px 32px'
  },
  card: {
    background: 'white', borderRadius: '16px',
    padding: '28px', boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
  },
  cardTitle: { fontSize: '20px', fontWeight: 'bold', color: '#2c3e50', margin: '0 0 8px' },
  cardDesc:  { color: '#666', fontSize: '14px', margin: '0 0 20px' },
  inputGroup: { marginBottom: '20px' },
  label: { display: 'block', marginBottom: '8px', fontWeight: '600', color: '#333', fontSize: '14px' },
  input: {
    width: '100%', padding: '12px 16px',
    border: '2px solid #e0e0e0', borderRadius: '8px',
    fontSize: '15px', outline: 'none', boxSizing: 'border-box'
  },
  seasonGrid: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' },
  seasonCard: {
    borderRadius: '10px', padding: '16px',
    textAlign: 'center', cursor: 'pointer'
  },
  seasonIcon:  { fontSize: '28px', display: 'block', marginBottom: '4px' },
  seasonLabel: { fontWeight: 'bold', color: '#2c3e50', margin: '0 0 2px', fontSize: '14px' },
  seasonDesc:  { color: '#666', margin: '0', fontSize: '11px' },
  submitBtn: {
    width: '100%', padding: '14px',
    background: 'linear-gradient(135deg, #27ae60, #2ecc71)',
    color: 'white', border: 'none', borderRadius: '8px',
    fontSize: '16px', fontWeight: 'bold', cursor: 'pointer'
  },
  resultCard: {
    background: 'white', borderRadius: '16px',
    padding: '28px', boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
  },
  soilBox: {
    background: '#fff8e1', borderRadius: '12px',
    padding: '16px', marginBottom: '20px'
  },
  soilTitle: { margin: '0 0 12px', color: '#2c3e50', fontSize: '16px' },
  soilGrid:  { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' },
  soilItem:  { textAlign: 'center', background: 'white', borderRadius: '8px', padding: '10px' },
  soilValue: { fontWeight: 'bold', color: '#2c3e50', margin: '0 0 2px', fontSize: '14px' },
  soilLabel: { color: '#666', margin: '0', fontSize: '11px' },
  recTitle:  { fontSize: '16px', color: '#2c3e50', margin: '0 0 12px' },
  recCard: {
    borderRadius: '10px', padding: '16px',
    marginBottom: '10px', display: 'flex',
    alignItems: 'center', justifyContent: 'space-between'
  },
  recLeft:      { display: 'flex', alignItems: 'center', gap: '12px' },
  recRank:      { fontSize: '20px', fontWeight: 'bold', color: '#666' },
  recCrop:      { margin: '0 0 4px', color: '#2c3e50', fontSize: '16px' },
  recDesc:      { margin: '0', color: '#666', fontSize: '12px' },
  recConfidence:{ textAlign: 'center' },
  confValue:    { fontWeight: 'bold', color: '#27ae60', margin: '0', fontSize: '20px' },
  confLabel:    { color: '#666', margin: '0', fontSize: '11px' }
};