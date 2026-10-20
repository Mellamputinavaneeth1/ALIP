import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getNotifications } from '../services/api';
import toast from 'react-hot-toast';

export default function Notifications() {
  const navigate                  = useNavigate();
  const [alerts, setAlerts]       = useState([]);
  const [loading, setLoading]     = useState(true);
  const [filter, setFilter]       = useState('all');
  const [summary, setSummary]     = useState({});

  useEffect(() => {
    loadAlerts();
  }, []);

  const loadAlerts = async () => {
    try {
      const res = await getNotifications();
      setAlerts(res.data.alerts);
      setSummary({
        total:   res.data.total,
        high:    res.data.high_alerts,
        medium:  res.data.medium_alerts
      });
    } catch (err) {
      toast.error('Failed to load notifications!');
    }
    setLoading(false);
  };

  const filteredAlerts = filter === 'all'
    ? alerts
    : alerts.filter(a => a.type === filter);

  const getAlertStyle = (type) => {
    if (type === 'oversupply')  return { border: '#e74c3c', bg: '#fff5f5', icon: '📈' };
    if (type === 'undersupply') return { border: '#f39c12', bg: '#fffbf0', icon: '📉' };
    return                             { border: '#27ae60', bg: '#f0fff4', icon: '✅' };
  };

  if (loading) {
    return (
      <div style={styles.loadingContainer}>
        <span style={styles.loadingIcon}>🔔</span>
        <p>Loading notifications...</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <button style={styles.backBtn} onClick={() => navigate('/broker')}>
          ← Back
        </button>
        <h1 style={styles.headerTitle}>
          🔔 Notifications
          {summary.high > 0 && (
            <span style={styles.badge}>{summary.high}</span>
          )}
        </h1>
        <button style={styles.refreshBtn} onClick={() => { setLoading(true); loadAlerts(); }}>
          🔄 Refresh
        </button>
      </div>

      <div style={styles.content}>
        {/* Summary Cards */}
        <div style={styles.summaryRow}>
          <div style={{...styles.summaryCard, borderTop: '4px solid #3498db'}}>
            <p style={styles.summaryValue}>{summary.total}</p>
            <p style={styles.summaryLabel}>🔔 Total Alerts</p>
          </div>
          <div style={{...styles.summaryCard, borderTop: '4px solid #e74c3c'}}>
            <p style={styles.summaryValue}>{summary.high}</p>
            <p style={styles.summaryLabel}>🚨 High Priority</p>
          </div>
          <div style={{...styles.summaryCard, borderTop: '4px solid #f39c12'}}>
            <p style={styles.summaryValue}>{summary.medium}</p>
            <p style={styles.summaryLabel}>⚠️ Medium Priority</p>
          </div>
          <div style={{...styles.summaryCard, borderTop: '4px solid #27ae60'}}>
            <p style={styles.summaryValue}>
              {summary.total - summary.high - summary.medium}
            </p>
            <p style={styles.summaryLabel}>✅ Normal</p>
          </div>
        </div>

        {/* Filter Tabs */}
        <div style={styles.filterRow}>
          {[
            { value: 'all',         label: '🔔 All' },
            { value: 'oversupply',  label: '📈 Oversupply' },
            { value: 'undersupply', label: '📉 Undersupply' },
            { value: 'normal',      label: '✅ Normal' }
          ].map(f => (
            <button
              key={f.value}
              style={{
                ...styles.filterBtn,
                background: filter === f.value
                  ? '#1a237e' : 'white',
                color: filter === f.value
                  ? 'white' : '#333'
              }}
              onClick={() => setFilter(f.value)}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* Alerts List */}
        {filteredAlerts.length > 0 ? (
          <div style={styles.alertsList}>
            {filteredAlerts.map((alert, idx) => {
              const style = getAlertStyle(alert.type);
              return (
                <div
                  key={idx}
                  style={{
                    ...styles.alertCard,
                    borderLeft: `5px solid ${style.border}`,
                    background: style.bg
                  }}
                >
                  <div style={styles.alertHeader}>
                    <div style={styles.alertLeft}>
                      <span style={styles.alertIcon}>{style.icon}</span>
                      <div>
                        <h3 style={styles.alertTitle}>{alert.title}</h3>
                        <p style={styles.alertTime}>🕐 {alert.time}</p>
                      </div>
                    </div>
                    <span style={{
                      ...styles.severityBadge,
                      background: alert.severity === 'high'
                        ? '#e74c3c'
                        : alert.severity === 'medium'
                        ? '#f39c12' : '#27ae60'
                    }}>
                      {alert.severity.toUpperCase()}
                    </span>
                  </div>

                  <p style={styles.alertMessage}>{alert.message}</p>

                  <div style={styles.alertFooter}>
                    <div style={styles.alertMeta}>
                      <span style={styles.metaItem}>📍 {alert.region}</span>
                      <span style={styles.metaItem}>🌾 {alert.crop}</span>
                      <span style={styles.metaItem}>⚖️ {alert.quantity}kg</span>
                    </div>
                    <div style={styles.actionBox}>
                      <p style={styles.actionLabel}>💡 Recommended Action:</p>
                      <p style={styles.actionText}>{alert.action}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div style={styles.emptyCard}>
            <span style={styles.emptyIcon}>🔕</span>
            <h3 style={styles.emptyTitle}>No Notifications</h3>
            <p style={styles.emptyText}>
              No alerts found. Market supply is balanced!
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: { minHeight: '100vh', background: '#f5f6fa' },
  loadingContainer: {
    minHeight: '100vh', display: 'flex',
    flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', fontSize: '18px', color: '#666'
  },
  loadingIcon: { fontSize: '48px', marginBottom: '16px' },
  header: {
    background:     'linear-gradient(135deg, #1a237e, #283593)',
    padding:        '16px 32px',
    display:        'flex',
    alignItems:     'center',
    gap:            '16px'
  },
  backBtn: {
    padding: '8px 16px', background: 'rgba(255,255,255,0.2)',
    color: 'white', border: 'none', borderRadius: '8px',
    cursor: 'pointer', fontSize: '14px'
  },
  headerTitle: {
    color: 'white', margin: '0',
    fontSize: '22px', flex: 1,
    display: 'flex', alignItems: 'center', gap: '10px'
  },
  badge: {
    background: '#e74c3c', color: 'white',
    borderRadius: '50%', width: '24px', height: '24px',
    display: 'flex', alignItems: 'center',
    justifyContent: 'center', fontSize: '12px', fontWeight: 'bold'
  },
  refreshBtn: {
    padding: '8px 16px', background: 'rgba(255,255,255,0.2)',
    color: 'white', border: '1px solid rgba(255,255,255,0.4)',
    borderRadius: '8px', cursor: 'pointer', fontSize: '13px'
  },
  content:    { padding: '24px 32px' },
  summaryRow: {
    display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '16px', marginBottom: '24px'
  },
  summaryCard: {
    background: 'white', borderRadius: '12px',
    padding: '20px', textAlign: 'center',
    boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
  },
  summaryValue: { fontSize: '28px', fontWeight: 'bold', color: '#2c3e50', margin: '0 0 4px' },
  summaryLabel: { color: '#666', margin: '0', fontSize: '13px' },
  filterRow: {
    display: 'flex', gap: '10px',
    marginBottom: '20px', flexWrap: 'wrap'
  },
  filterBtn: {
    padding: '10px 20px', border: '2px solid #e0e0e0',
    borderRadius: '25px', cursor: 'pointer',
    fontSize: '14px', fontWeight: '600', transition: 'all 0.2s'
  },
  alertsList: { display: 'flex', flexDirection: 'column', gap: '16px' },
  alertCard: {
    background: 'white', borderRadius: '12px',
    padding: '20px', boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
  },
  alertHeader: {
    display: 'flex', justifyContent: 'space-between',
    alignItems: 'flex-start', marginBottom: '12px'
  },
  alertLeft:   { display: 'flex', alignItems: 'center', gap: '12px' },
  alertIcon:   { fontSize: '32px' },
  alertTitle:  { fontSize: '16px', fontWeight: 'bold', color: '#2c3e50', margin: '0 0 4px' },
  alertTime:   { color: '#999', fontSize: '12px', margin: '0' },
  severityBadge: {
    padding: '4px 12px', borderRadius: '12px',
    color: 'white', fontSize: '11px', fontWeight: 'bold'
  },
  alertMessage: { color: '#555', fontSize: '14px', margin: '0 0 16px', lineHeight: '1.5' },
  alertFooter:  { borderTop: '1px solid #f0f0f0', paddingTop: '12px' },
  alertMeta: {
    display: 'flex', gap: '16px',
    marginBottom: '10px', flexWrap: 'wrap'
  },
  metaItem:    { color: '#666', fontSize: '13px', fontWeight: '600' },
  actionBox: {
    background: 'rgba(255,255,255,0.7)',
    borderRadius: '8px', padding: '10px'
  },
  actionLabel: { color: '#666', fontSize: '12px', margin: '0 0 4px', fontWeight: '600' },
  actionText:  { color: '#2c3e50', fontSize: '13px', margin: '0', fontWeight: '500' },
  emptyCard: {
    background: 'white', borderRadius: '16px',
    padding: '48px', textAlign: 'center',
    boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
  },
  emptyIcon:  { fontSize: '64px', display: 'block', marginBottom: '16px' },
  emptyTitle: { fontSize: '20px', color: '#2c3e50', margin: '0 0 8px' },
  emptyText:  { color: '#666', fontSize: '14px', margin: '0' }
};