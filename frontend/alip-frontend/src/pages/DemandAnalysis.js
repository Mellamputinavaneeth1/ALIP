import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getDemandAnalysis } from '../services/api';
import toast from 'react-hot-toast';

export default function DemandAnalysis() {
  const navigate            = useNavigate();
  const [region, setRegion] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!region) { toast.error('Enter a region!'); return; }
    setLoading(true);
    try {
      const res = await getDemandAnalysis(region);
      setResult(res.data);
      toast.success('Analysis complete!');
    } catch (err) {
      toast.error('Failed to analyze!');
    }
    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <button style={styles.backBtn} onClick={() => navigate('/broker')}>
          ← Back
        </button>
        <h1 style={styles.headerTitle}>📊 Demand Analysis</h1>
      </div>

      <div style={styles.content}>
        {/* Search */}
        <div style={styles.searchCard}>
          <h2 style={styles.cardTitle}>Analyze Regional Demand</h2>
          <div style={styles.searchRow}>
            <input
              style={styles.input}
              placeholder="Enter region e.g. Hyderabad"
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
            />
            <button
              style={{...styles.searchBtn, opacity: loading ? 0.7 : 1}}
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? '⏳' : '🔍 Analyze'}
            </button>
          </div>
        </div>

        {result && (
          <>
            {/* Alerts */}
            {result.alerts.length > 0 && (
              <div style={styles.alertsCard}>
                <h2 style={styles.cardTitle}>
                  ⚠️ Market Alerts ({result.total_alerts})
                </h2>
                {result.alerts.map((alert, idx) => (
                  <div
                    key={idx}
                    style={{
                      ...styles.alertItem,
                      borderLeft: `4px solid ${
                        alert.type === 'oversupply' ? '#e74c3c' : '#f39c12'
                      }`,
                      background: alert.type === 'oversupply'
                        ? '#fff5f5' : '#fffbf0'
                    }}
                  >
                    <h4 style={styles.alertTitle}>{alert.message}</h4>
                    <p style={styles.alertAction}>
                      💡 Action: {alert.action}
                    </p>
                    <span style={{
                      ...styles.alertBadge,
                      background: alert.type === 'oversupply'
                        ? '#e74c3c' : '#f39c12'
                    }}>
                      {alert.type === 'oversupply'
                        ? '📈 Oversupply' : '📉 Undersupply'}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Supply Summary */}
            <div style={styles.summaryCard}>
              <h2 style={styles.cardTitle}>
                📦 Supply Summary — {result.region}
              </h2>
              {Object.entries(result.analysis).map(([crop, qty], idx) => (
                <div key={idx} style={styles.supplyRow}>
                  <span style={styles.cropName}>
                    🌾 {crop.charAt(0).toUpperCase() + crop.slice(1)}
                  </span>
                  <span style={styles.cropQty}>{qty} kg</span>
                </div>
              ))}

              {/* Recommendations */}
              {result.recommendations.map((rec, idx) => (
                <div key={idx} style={styles.recItem}>
                  <p style={styles.recText}>{rec.message}</p>
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
    background: 'linear-gradient(135deg, #1a237e, #283593)',
    padding: '16px 32px', display: 'flex',
    alignItems: 'center', gap: '16px'
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
  cardTitle:  { fontSize: '20px', fontWeight: 'bold', color: '#2c3e50', margin: '0 0 16px' },
  searchRow:  { display: 'flex', gap: '12px' },
  input: {
    flex: 1, padding: '12px 16px',
    border: '2px solid #e0e0e0', borderRadius: '8px',
    fontSize: '15px', outline: 'none'
  },
  searchBtn: {
    padding: '12px 24px',
    background: 'linear-gradient(135deg, #9b59b6, #8e44ad)',
    color: 'white', border: 'none', borderRadius: '8px',
    fontSize: '15px', fontWeight: 'bold', cursor: 'pointer'
  },
  alertsCard: {
    background: 'white', borderRadius: '16px',
    padding: '24px', marginBottom: '24px',
    boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
  },
  alertItem: {
    borderRadius: '10px', padding: '16px',
    marginBottom: '12px'
  },
  alertTitle:  { margin: '0 0 8px', color: '#2c3e50', fontSize: '14px' },
  alertAction: { margin: '0 0 8px', color: '#555', fontSize: '13px' },
  alertBadge: {
    display: 'inline-block', padding: '4px 12px',
    borderRadius: '12px', color: 'white',
    fontSize: '12px', fontWeight: 'bold'
  },
  summaryCard: {
    background: 'white', borderRadius: '16px',
    padding: '24px', boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
  },
  supplyRow: {
    display: 'flex', justifyContent: 'space-between',
    alignItems: 'center', padding: '12px 16px',
    background: '#f8f9fa', borderRadius: '8px', marginBottom: '8px'
  },
  cropName:  { fontWeight: '600', color: '#2c3e50', fontSize: '15px' },
  cropQty:   { fontWeight: 'bold', color: '#27ae60', fontSize: '15px' },
  recItem: {
    background: '#f0fff4', borderRadius: '8px',
    padding: '12px 16px', marginTop: '8px',
    borderLeft: '4px solid #27ae60'
  },
  recText: { margin: '0', color: '#2c3e50', fontSize: '14px' }
};