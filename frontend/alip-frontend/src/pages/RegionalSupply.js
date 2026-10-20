import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getRegionalSupply } from '../services/api';
import toast from 'react-hot-toast';

export default function RegionalSupply() {
  const navigate              = useNavigate();
  const [region, setRegion]   = useState('');
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!region) { toast.error('Please enter a region!'); return; }
    setLoading(true);
    try {
      const res = await getRegionalSupply(region);
      setResult(res.data);
      toast.success('Supply data loaded!');
    } catch (err) {
      toast.error('Failed to load supply data!');
    }
    setLoading(false);
  };

  const getSupplyStatus = (quantity) => {
    if (quantity > 500) return { label: 'High',     color: '#e74c3c' };
    if (quantity > 200) return { label: 'Medium',   color: '#f39c12' };
    return                     { label: 'Low',      color: '#27ae60' };
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <button style={styles.backBtn} onClick={() => navigate('/broker')}>
          ← Back
        </button>
        <h1 style={styles.headerTitle}>🗺️ Regional Supply</h1>
      </div>

      <div style={styles.content}>
        {/* Search Card */}
        <div style={styles.searchCard}>
          <h2 style={styles.cardTitle}>Check Regional Crop Supply</h2>
          <p style={styles.cardDesc}>
            Enter a region to view all available crop supplies
          </p>
          <div style={styles.searchRow}>
            <input
              style={styles.input}
              placeholder="Enter region e.g. Hyderabad, Chennai..."
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
            />
            <button
              style={{...styles.searchBtn, opacity: loading ? 0.7 : 1}}
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? '⏳' : '🔍 Search'}
            </button>
          </div>

          {/* Quick Region Buttons */}
          <div style={styles.quickRegions}>
            <p style={styles.quickLabel}>Quick Select:</p>
            {['Hyderabad', 'Chennai', 'Mumbai', 'Delhi', 'Bangalore'].map(r => (
              <button
                key={r}
                style={styles.quickBtn}
                onClick={() => {
                  setRegion(r);
                }}
              >
                📍 {r}
              </button>
            ))}
          </div>
        </div>

        {/* Results */}
        {result && (
          <>
            {/* Summary Stats */}
            <div style={styles.statsRow}>
              <div style={styles.statCard}>
                <p style={styles.statValue}>{result.region}</p>
                <p style={styles.statLabel}>📍 Region</p>
              </div>
              <div style={styles.statCard}>
                <p style={styles.statValue}>{result.crops_available}</p>
                <p style={styles.statLabel}>🌾 Crop Types</p>
              </div>
              <div style={styles.statCard}>
                <p style={styles.statValue}>
                  {Object.values(result.supply)
                    .reduce((a, b) => a + b.total_quantity_kg, 0)
                    .toFixed(0)} kg
                </p>
                <p style={styles.statLabel}>📦 Total Supply</p>
              </div>
              <div style={styles.statCard}>
                <p style={styles.statValue}>
                  {Object.values(result.supply)
                    .reduce((a, b) => a + b.listings_count, 0)}
                </p>
                <p style={styles.statLabel}>📋 Total Listings</p>
              </div>
            </div>

            {/* Supply Table */}
            {result.crops_available > 0 ? (
              <div style={styles.tableCard}>
                <h2 style={styles.cardTitle}>
                  🌾 Crop Supply in {result.region}
                </h2>

                {/* Table Header */}
                <div style={styles.tableHeader}>
                  <span style={styles.th}>Crop Name</span>
                  <span style={styles.th}>Total Quantity</span>
                  <span style={styles.th}>Listings</span>
                  <span style={styles.th}>Avg Price/kg</span>
                  <span style={styles.th}>Supply Level</span>
                </div>

                {/* Table Rows */}
                {Object.entries(result.supply).map(([crop, data], idx) => {
                  const status = getSupplyStatus(data.total_quantity_kg);
                  return (
                    <div
                      key={idx}
                      style={{
                        ...styles.tableRow,
                        background: idx % 2 === 0 ? '#f8f9fa' : 'white'
                      }}
                    >
                      <span style={styles.td}>
                        🌾 {crop.charAt(0).toUpperCase() + crop.slice(1)}
                      </span>
                      <span style={styles.td}>
                        {data.total_quantity_kg} kg
                      </span>
                      <span style={styles.td}>
                        {data.listings_count}
                      </span>
                      <span style={styles.td}>
                        ₹{data.avg_price_per_kg}/kg
                      </span>
                      <span style={styles.td}>
                        <span style={{
                          ...styles.badge,
                          background: status.color
                        }}>
                          {status.label}
                        </span>
                      </span>
                    </div>
                  );
                })}

                {/* Supply Chart Bars */}
                <h3 style={styles.chartTitle}>📊 Supply Distribution</h3>
                {Object.entries(result.supply).map(([crop, data], idx) => {
                  const maxQty = Math.max(
                    ...Object.values(result.supply)
                      .map(d => d.total_quantity_kg)
                  );
                  const percentage = (data.total_quantity_kg / maxQty) * 100;
                  const status     = getSupplyStatus(data.total_quantity_kg);

                  return (
                    <div key={idx} style={styles.barRow}>
                      <span style={styles.barLabel}>
                        {crop.charAt(0).toUpperCase() + crop.slice(1)}
                      </span>
                      <div style={styles.barTrack}>
                        <div style={{
                          ...styles.barFill,
                          width:      `${percentage}%`,
                          background: status.color
                        }} />
                      </div>
                      <span style={styles.barValue}>
                        {data.total_quantity_kg} kg
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={styles.emptyCard}>
                <span style={styles.emptyIcon}>📭</span>
                <h3 style={styles.emptyTitle}>No Supply Data</h3>
                <p style={styles.emptyText}>
                  No crop listings found for {result.region}.
                  Farmers need to add listings first.
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: { minHeight: '100vh', background: '#f5f6fa' },
  header: {
    background:  'linear-gradient(135deg, #1a237e, #283593)',
    padding:     '16px 32px',
    display:     'flex',
    alignItems:  'center',
    gap:         '16px'
  },
  backBtn: {
    padding:      '8px 16px',
    background:   'rgba(255,255,255,0.2)',
    color:        'white',
    border:       'none',
    borderRadius: '8px',
    cursor:       'pointer',
    fontSize:     '14px'
  },
  headerTitle: { color: 'white', margin: '0', fontSize: '22px' },
  content:     { padding: '24px 32px' },
  searchCard: {
    background:   'white',
    borderRadius: '16px',
    padding:      '24px',
    marginBottom: '24px',
    boxShadow:    '0 4px 20px rgba(0,0,0,0.08)'
  },
  cardTitle: {
    fontSize:   '20px',
    fontWeight: 'bold',
    color:      '#2c3e50',
    margin:     '0 0 8px'
  },
  cardDesc: { color: '#666', fontSize: '14px', margin: '0 0 16px' },
  searchRow: { display: 'flex', gap: '12px', marginBottom: '16px' },
  input: {
    flex:         1,
    padding:      '12px 16px',
    border:       '2px solid #e0e0e0',
    borderRadius: '8px',
    fontSize:     '15px',
    outline:      'none'
  },
  searchBtn: {
    padding:      '12px 24px',
    background:   'linear-gradient(135deg, #2980b9, #3498db)',
    color:        'white',
    border:       'none',
    borderRadius: '8px',
    fontSize:     '15px',
    fontWeight:   'bold',
    cursor:       'pointer'
  },
  quickRegions: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' },
  quickLabel:   { color: '#666', fontSize: '13px', margin: '0' },
  quickBtn: {
    padding:      '6px 12px',
    background:   '#f0f4ff',
    border:       '1px solid #c5cae9',
    borderRadius: '20px',
    cursor:       'pointer',
    fontSize:     '12px',
    color:        '#3949ab',
    fontWeight:   '600'
  },
  statsRow: {
    display:             'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap:                 '16px',
    marginBottom:        '24px'
  },
  statCard: {
    background:   'white',
    borderRadius: '12px',
    padding:      '20px',
    textAlign:    'center',
    boxShadow:    '0 4px 20px rgba(0,0,0,0.08)'
  },
  statValue: {
    fontSize:   '22px',
    fontWeight: 'bold',
    color:      '#2c3e50',
    margin:     '0 0 4px'
  },
  statLabel: { color: '#666', margin: '0', fontSize: '13px' },
  tableCard: {
    background:   'white',
    borderRadius: '16px',
    padding:      '24px',
    boxShadow:    '0 4px 20px rgba(0,0,0,0.08)'
  },
  tableHeader: {
    display:         'grid',
    gridTemplateColumns: 'repeat(5, 1fr)',
    background:      '#f0f4ff',
    borderRadius:    '8px',
    padding:         '12px 16px',
    marginBottom:    '8px',
    marginTop:       '16px'
  },
  th: { fontWeight: 'bold', color: '#3949ab', fontSize: '13px' },
  tableRow: {
    display:             'grid',
    gridTemplateColumns: 'repeat(5, 1fr)',
    padding:             '12px 16px',
    borderRadius:        '8px',
    marginBottom:        '4px',
    alignItems:          'center'
  },
  td: { color: '#2c3e50', fontSize: '14px' },
  badge: {
    display:      'inline-block',
    padding:      '4px 10px',
    borderRadius: '12px',
    color:        'white',
    fontSize:     '11px',
    fontWeight:   'bold'
  },
  chartTitle: {
    fontSize:   '16px',
    color:      '#2c3e50',
    margin:     '24px 0 16px'
  },
  barRow: {
    display:     'flex',
    alignItems:  'center',
    gap:         '12px',
    marginBottom:'10px'
  },
  barLabel: {
    width:      '120px',
    fontSize:   '13px',
    color:      '#2c3e50',
    fontWeight: '600'
  },
  barTrack: {
    flex:         1,
    height:       '24px',
    background:   '#f0f0f0',
    borderRadius: '12px',
    overflow:     'hidden'
  },
  barFill: {
    height:       '100%',
    borderRadius: '12px',
    transition:   'width 0.5s ease'
  },
  barValue: { width: '80px', fontSize: '13px', color: '#666', textAlign: 'right' },
  emptyCard: {
    background:   'white',
    borderRadius: '16px',
    padding:      '48px',
    textAlign:    'center',
    boxShadow:    '0 4px 20px rgba(0,0,0,0.08)'
  },
  emptyIcon:  { fontSize: '64px', display: 'block', marginBottom: '16px' },
  emptyTitle: { fontSize: '20px', color: '#2c3e50', margin: '0 0 8px' },
  emptyText:  { color: '#666', fontSize: '14px', margin: '0' }
};