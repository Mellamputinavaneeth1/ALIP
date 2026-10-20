import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getFullDemandAnalysis, initiateShipment } from '../services/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart
} from 'recharts';
import toast from 'react-hot-toast';
import './DemandAnalysis.css';

/* ─── Constants ──────────────────────────────────────── */
const REGIONS = [
  'Hyderabad','Warangal','Karimnagar','Nizamabad','Khammam',
  'Nalgonda','Mahabubnagar','Adilabad','Medak','Rangareddy',
  'Vijayawada','Guntur','Kurnool','Anantapur','Tirupati',
  'Visakhapatnam','Rajahmundry','Eluru','Nellore','Ongole',
  'Kadapa','Srikakulam','Kakinada','Chittoor','Machilipatnam',
  'Mumbai','Pune','Nagpur','Nashik','Aurangabad',
  'Chennai','Coimbatore','Madurai',
  'Bangalore','Mysore',
  'Delhi','Lucknow','Jaipur','Indore','Bhopal',
  'Kolkata','Patna','Raipur','Ahmedabad','Surat'
];

const CROPS = [
  'rice','wheat','maize','cotton','sugarcane',
  'groundnut','soybean','tomato','banana','potato','onion'
];

const TRUCKS = [
  { value: 'small',  label: '🚛 Small — ₹25/km' },
  { value: 'medium', label: '🚚 Medium — ₹35/km' },
  { value: 'large',  label: '🚜 Large — ₹50/km' },
];

/* ─── Helpers ────────────────────────────────────────── */
function urgencyClass(label) {
  if (!label) return '';
  const l = label.toLowerCase();
  if (l === 'watch') return 'da-urgency-watch';
  if (l === 'high') return 'da-urgency-high';
  if (l === 'critical') return 'da-urgency-critical';
  return 'da-urgency-emergency';
}

function urgencyTagClass(label) {
  if (!label) return '';
  return label.toLowerCase();
}

function scoreColor(score) {
  if (score >= 75) return '#22c55e';
  if (score >= 55) return '#52b788';
  if (score >= 40) return '#f59e0b';
  return '#ef4444';
}

function statusClass(level) {
  if (!level) return 'low';
  return level.toLowerCase();
}

function roiClass(roi) {
  if (roi >= 1.5) return 'good';
  if (roi >= 1) return 'ok';
  return 'bad';
}

/* ─── Main Component ─────────────────────────────────── */
export default function DemandAnalysis() {
  const navigate = useNavigate();

  /* Form state */
  const [region, setRegion]       = useState('');
  const [crop, setCrop]           = useState('');
  const [supplyKg, setSupplyKg]   = useState('');
  const [truckType, setTruckType] = useState('medium');

  /* Result state */
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);

  /* Modal */
  const [shipModal, setShipModal]       = useState(null);
  const [shipping, setShipping]         = useState(false);

  /* ─── Fetch ──────────────────────────────────────── */
  const handleAnalyze = async () => {
    if (!region || !crop || !supplyKg) {
      toast.error('Fill in region, crop, and supply quantity');
      return;
    }
    setLoading(true);
    setData(null);
    try {
      const res = await getFullDemandAnalysis(region, crop, supplyKg, {
        truck_type: truckType,
        top_n: 5
      });
      setData(res.data);
      toast.success('Analysis complete!');
    } catch (err) {
      const msg = err?.response?.data?.detail || 'Analysis failed';
      toast.error(typeof msg === 'string' ? msg : 'Analysis failed');
    }
    setLoading(false);
  };

  /* ─── Ship ───────────────────────────────────────── */
  const handleShip = async () => {
    if (!shipModal) return;
    setShipping(true);
    try {
      const payload = {
        broker_id: localStorage.getItem('userId') || 'demo',
        source_region: data.region,
        destination_region: shipModal.region,
        crop: data.crop,
        quantity_kg: shipModal.quantity_to_send,
        truck_type: truckType,
        expected_delivery_days: 2
      };
      await initiateShipment(payload);
      toast.success(`Shipment to ${shipModal.region} initiated!`);
      setShipModal(null);
    } catch (err) {
      toast.error('Failed to initiate shipment');
    }
    setShipping(false);
  };

  /* ─── Chart Data ─────────────────────────────────── */
  const chartData = data?.price_trajectory?.day_prices?.map((p, i) => ({
    day: `Day ${i}`,
    price: p,
    loss: data.price_trajectory.day_losses_per_100kg[i]
  })) || [];

  /* ═══════════════ RENDER ════════════════ */
  return (
    <div className="da-page">
      {/* ─── Header ─────────────────────────── */}
      <div className="da-header">
        <button className="da-back" onClick={() => navigate('/broker')}>← Back</button>
        <h1>📊 Market Intelligence Dashboard</h1>
        <span className="da-badge">AI Powered</span>
      </div>

      <div className="da-body">
        {/* ─── Input Card ───────────────────── */}
        <div className="da-input-card">
          <h2>Run Demand Analysis</h2>
          <div className="da-input-row">
            <div className="da-input-group">
              <label>Region</label>
              <select value={region} onChange={e => setRegion(e.target.value)}>
                <option value="">Select region...</option>
                {REGIONS.map(r => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            <div className="da-input-group">
              <label>Crop</label>
              <select value={crop} onChange={e => setCrop(e.target.value)}>
                <option value="">Select crop...</option>
                {CROPS.map(c => (
                  <option key={c} value={c}>
                    {c.charAt(0).toUpperCase() + c.slice(1)}
                  </option>
                ))}
              </select>
            </div>
            <div className="da-input-group">
              <label>Supply (kg)</label>
              <input
                type="number" placeholder="e.g. 1500"
                value={supplyKg}
                onChange={e => setSupplyKg(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleAnalyze()}
              />
            </div>
            <div className="da-input-group">
              <label>Truck Type</label>
              <select value={truckType} onChange={e => setTruckType(e.target.value)}>
                {TRUCKS.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
            </div>
            <button
              className="da-analyze-btn"
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? '⏳ Analyzing...' : '🔍 Analyze'}
            </button>
          </div>
        </div>

        {/* ─── Loading ──────────────────────── */}
        {loading && (
          <div className="da-loading">
            <div className="da-spinner" />
            <span className="da-loading-text">
              Running 7-factor intelligence analysis...
            </span>
          </div>
        )}

        {/* ─── Empty ────────────────────────── */}
        {!loading && !data && (
          <div className="da-empty">
            <div className="da-empty-icon">📊</div>
            <h3>No Analysis Yet</h3>
            <p>Select a region, crop, and supply quantity to run the intelligence engine.</p>
          </div>
        )}

        {/* ─── Results ──────────────────────── */}
        {!loading && data && data.success && (
          <>
            {/* ── Healthy Market Banner ────── */}
            {data.message && (
              <div className="da-panel" style={{
                marginBottom: 28,
                borderColor: 'rgba(34,197,94,0.25)',
                background: 'rgba(34,197,94,0.06)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{ fontSize: 36 }}>✅</span>
                  <div>
                    <h3 style={{ margin: '0 0 4px', color: '#22c55e', fontSize: 18 }}>
                      Market is Healthy
                    </h3>
                    <p style={{ margin: 0, color: '#94a3b8', fontSize: 13, lineHeight: 1.5 }}>
                      {data.urgency?.reasoning || data.message}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* ── Metric Cards ─────────────── */}
            <div className="da-metrics">
              {/* Urgency */}
              <div className="da-metric-card">
                <div className="da-metric-label">Urgency Score</div>
                <div className={`da-metric-value ${urgencyClass(data.urgency?.label)}`}>
                  {data.urgency?.score || 0}/10
                </div>
                <div className="da-metric-sub">
                  Act within {data.urgency?.act_within_days || '—'} days
                </div>
                <span className={`da-urgency-tag ${urgencyTagClass(data.urgency?.label)}`}>
                  {data.urgency?.label || 'UNKNOWN'}
                </span>
              </div>

              {/* Oversupply Ratio */}
              <div className="da-metric-card">
                <div className="da-metric-label">Oversupply Ratio</div>
                <div className="da-metric-value" style={{
                  color: (data.current_state?.oversupply_ratio || 0) > 1.5
                    ? '#ef4444' : (data.current_state?.oversupply_ratio || 0) > 1
                    ? '#f59e0b' : '#22c55e'
                }}>
                  {data.current_state?.oversupply_ratio || '—'}x
                </div>
                <div className="da-metric-sub">
                  Supply: {data.current_state?.current_supply_kg?.toLocaleString() || '—'} kg
                </div>
                <div className="da-metric-sub">
                  Threshold: {data.current_state?.dynamic_threshold_kg?.toLocaleString() || '—'} kg
                </div>
              </div>

              {/* Z-Score */}
              <div className="da-metric-card">
                <div className="da-metric-label">Z-Score Anomaly</div>
                <div className="da-metric-value" style={{
                  color: (data.current_state?.anomaly_level === 'critical')
                    ? '#ef4444' : (data.current_state?.anomaly_level === 'significant')
                    ? '#f59e0b' : '#22c55e'
                }}>
                  {typeof data.current_state?.z_score === 'number'
                    ? data.current_state.z_score.toFixed(1) : '—'}
                </div>
                <div className="da-metric-sub">
                  Level: {data.current_state?.anomaly_level || 'unknown'}
                </div>
                <div className="da-metric-sub">
                  Trend: {data.current_state?.supply_trend || '—'}
                </div>
              </div>

              {/* Price Day-7 Loss */}
              <div className="da-metric-card">
                <div className="da-metric-label">7-Day Price Loss</div>
                <div className="da-metric-value" style={{ color: '#ef4444' }}>
                  ₹{data.price_trajectory?.total_projected_loss_7d_per_100kg?.toFixed(0) || '0'}
                </div>
                <div className="da-metric-sub">per 100 kg if no action</div>
                <div className="da-metric-sub">
                  Break-even day: {data.price_trajectory?.break_even_day ?? 'N/A'}
                </div>
              </div>
            </div>

            {/* ── Two-Panel Row ─────────────── */}
            <div className="da-panels">
              {/* Price Trajectory Chart */}
              <div className="da-panel">
                <div className="da-panel-head">
                  <div className="da-icon" style={{ background: 'rgba(239,68,68,0.12)' }}>📉</div>
                  <h3>Price Trajectory (7 Days)</h3>
                </div>
                <div className="da-chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ef4444" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{
                          background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)',
                          borderRadius: 8, color: '#e2e8f0', fontSize: 12
                        }}
                        formatter={(value, name) => [
                          `₹${value}`,
                          name === 'price' ? 'Price/kg' : 'Loss/100kg'
                        ]}
                      />
                      {data.price_trajectory?.break_even_day != null && (
                        <ReferenceLine
                          x={`Day ${data.price_trajectory.break_even_day}`}
                          stroke="#f59e0b" strokeDasharray="5 5"
                          label={{ value: 'Break-even', fill: '#f59e0b', fontSize: 10 }}
                        />
                      )}
                      <Area
                        type="monotone" dataKey="price"
                        stroke="#ef4444" strokeWidth={2}
                        fill="url(#priceGrad)"
                        dot={{ r: 3, fill: '#ef4444' }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                {data.price_trajectory?.recommendation && (
                  <div className="da-trajectory-rec">
                    💡 {data.price_trajectory.recommendation}
                  </div>
                )}
              </div>

              {/* Seasonal Context */}
              <div className="da-panel">
                <div className="da-panel-head">
                  <div className="da-icon" style={{ background: 'rgba(59,130,246,0.12)' }}>🌾</div>
                  <h3>Seasonal Context</h3>
                </div>
                {data.seasonal_context ? (
                  <>
                    <span className={`da-seasonal-badge ${
                      data.seasonal_context.is_peak_month ? 'peak' : 'off-peak'
                    }`}>
                      {data.seasonal_context.is_peak_month
                        ? `📅 Peak Month — ${data.seasonal_context.season} season`
                        : `${data.seasonal_context.month_name} — ${data.seasonal_context.season} season`
                      }
                    </span>
                    <p className="da-seasonal-text">
                      {data.seasonal_context.reasoning}
                    </p>
                    {data.seasonal_context.self_correction_likely && (
                      <div className="da-seasonal-rec" style={{
                        background: 'rgba(34,197,94,0.08)',
                        borderColor: 'rgba(34,197,94,0.15)',
                        color: '#22c55e'
                      }}>
                        ✅ Self-correction likely — {data.seasonal_context.self_correction_note}
                      </div>
                    )}
                    {!data.seasonal_context.self_correction_likely && data.seasonal_context.has_pattern && (
                      <div className="da-seasonal-rec" style={{
                        background: 'rgba(245,158,11,0.08)',
                        borderColor: 'rgba(245,158,11,0.15)',
                        color: '#f59e0b'
                      }}>
                        ⚠️ Self-correction unlikely — active redistribution needed
                      </div>
                    )}
                    <div className="da-seasonal-rec" style={{ marginTop: 8 }}>
                      📋 {data.seasonal_context.recommendation}
                    </div>
                  </>
                ) : (
                  <p className="da-seasonal-text">No seasonal data available.</p>
                )}
              </div>
            </div>

            {/* ── Destination Ranking ──────── */}
            <div className="da-panel full" style={{ marginBottom: 28 }}>
              <div className="da-panel-head">
                <div className="da-icon" style={{ background: 'rgba(82,183,136,0.12)' }}>🎯</div>
                <h3>
                  Top Destinations — Weighted Ranking
                  {data.destinations?.length > 0 && (
                    <span style={{ fontWeight: 400, color: '#64748b', fontSize: 12, marginLeft: 8 }}>
                      ({data.destinations.length} results)
                    </span>
                  )}
                </h3>
              </div>
              {data.destinations?.length > 0 ? (
                <div style={{ overflowX: 'auto' }}>
                  <table className="da-dest-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Market</th>
                        <th>Distance</th>
                        <th>Score</th>
                        <th>Send Qty</th>
                        <th>Transport</th>
                        <th>ROI</th>
                        <th>Saturation</th>
                        <th>Weather</th>
                        <th>Timing</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.destinations.map((dest, i) => (
                        <tr key={i}>
                          <td style={{ fontWeight: 700, color: '#52b788' }}>
                            {i + 1}
                          </td>
                          <td>
                            <div style={{ fontWeight: 600 }}>{dest.region}</div>
                            <div style={{ fontSize: 10, color: '#64748b' }}>{dest.state}</div>
                          </td>
                          <td>{Math.round(dest.distance_km)} km</td>
                          <td>
                            <div className="da-score-bar-wrap">
                              <div className="da-score-bar-bg">
                                <div
                                  className="da-score-bar-fill"
                                  style={{
                                    width: `${dest.composite_score}%`,
                                    background: scoreColor(dest.composite_score)
                                  }}
                                />
                              </div>
                              <span className="da-score-num" style={{ color: scoreColor(dest.composite_score) }}>
                                {dest.composite_score}
                              </span>
                            </div>
                            <div className="da-stars">
                              {'★'.repeat(dest.star_rating)}{'☆'.repeat(5 - dest.star_rating)}
                            </div>
                          </td>
                          <td style={{ fontWeight: 600 }}>
                            {dest.quantity_to_send?.toLocaleString()} kg
                          </td>
                          <td>
                            <div>₹{dest.transport_cost_per_100kg}/100kg</div>
                            <div style={{ fontSize: 10, color: '#64748b' }}>
                              Total: ₹{dest.transport_total?.toLocaleString()}
                            </div>
                          </td>
                          <td>
                            <span className={`da-roi ${roiClass(dest.roi)}`}>
                              {dest.roi}x
                            </span>
                          </td>
                          <td>
                            <span className={`da-status ${statusClass(dest.saturation)}`}>
                              {dest.saturation}
                            </span>
                          </td>
                          <td>
                            <span className={`da-status ${statusClass(dest.weather_risk)}`}>
                              {dest.weather_risk}
                            </span>
                          </td>
                          <td style={{ fontSize: 11, maxWidth: 140 }}>
                            {dest.timing}
                          </td>
                          <td>
                            <button
                              className="da-ship-btn"
                              onClick={() => setShipModal(dest)}
                            >
                              🚛 Ship
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="da-empty" style={{ padding: '30px 0' }}>
                  <p>No suitable destinations found within range.</p>
                </div>
              )}

              {/* Excluded destinations */}
              {data.excluded_destinations?.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, fontWeight: 600 }}>
                    EXCLUDED DESTINATIONS
                  </div>
                  {data.excluded_destinations.map((ex, i) => (
                    <div key={i} style={{
                      display: 'inline-block', padding: '4px 10px',
                      background: 'rgba(239,68,68,0.06)',
                      border: '1px solid rgba(239,68,68,0.1)',
                      borderRadius: 6, fontSize: 11, color: '#ef4444',
                      marginRight: 8, marginBottom: 6
                    }}>
                      ✕ {ex.region} — {ex.reason}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* ── Allocation Summary ──────── */}
            {data.allocation_summary && (
              <div className="da-panel full">
                <div className="da-panel-head">
                  <div className="da-icon" style={{ background: 'rgba(82,183,136,0.12)' }}>📦</div>
                  <h3>Allocation Summary</h3>
                </div>
                <div className="da-alloc">
                  <div className="da-alloc-item">
                    <div className="da-alloc-val" style={{ color: '#ef4444' }}>
                      {data.allocation_summary.total_excess_kg?.toLocaleString()} kg
                    </div>
                    <div className="da-alloc-label">Total Excess</div>
                  </div>
                  <div className="da-alloc-item">
                    <div className="da-alloc-val">
                      {data.allocation_summary.recommended_send_kg?.toLocaleString()} kg
                    </div>
                    <div className="da-alloc-label">Recommended Send</div>
                  </div>
                  <div className="da-alloc-item">
                    <div className="da-alloc-val" style={{ color: '#60a5fa' }}>
                      {data.allocation_summary.keep_local_kg?.toLocaleString()} kg
                    </div>
                    <div className="da-alloc-label">Keep Local</div>
                  </div>
                  <div className="da-alloc-item">
                    <div className="da-alloc-val">
                      ₹{data.allocation_summary.net_savings_vs_holding?.toLocaleString()}
                    </div>
                    <div className="da-alloc-label">Net Savings vs Holding</div>
                  </div>
                </div>

                {/* Urgency reasoning */}
                {data.urgency?.reasoning && (
                  <div style={{
                    marginTop: 16, padding: '10px 14px',
                    background: 'rgba(82,183,136,0.06)',
                    border: '1px solid rgba(82,183,136,0.1)',
                    borderRadius: 8, fontSize: 12, color: '#94a3b8',
                    lineHeight: 1.6
                  }}>
                    🤖 <strong style={{ color: '#52b788' }}>AI Reasoning:</strong>{' '}
                    {data.urgency.reasoning}
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* ─── Error state ─────────────────── */}
        {!loading && data && !data.success && (
          <div className="da-empty">
            <div className="da-empty-icon">❌</div>
            <h3>Analysis Failed</h3>
            <p>{data.error || 'Unknown error occurred'}</p>
          </div>
        )}
      </div>

      {/* ─── Ship Confirmation Modal ─────── */}
      {shipModal && (
        <div className="da-modal-overlay" onClick={() => !shipping && setShipModal(null)}>
          <div className="da-modal" onClick={e => e.stopPropagation()}>
            <h3>🚛 Confirm Shipment</h3>
            <div className="da-modal-row">
              <span className="label">From</span>
              <span className="value">{data?.region}</span>
            </div>
            <div className="da-modal-row">
              <span className="label">To</span>
              <span className="value">{shipModal.region} ({shipModal.state})</span>
            </div>
            <div className="da-modal-row">
              <span className="label">Crop</span>
              <span className="value">{data?.crop}</span>
            </div>
            <div className="da-modal-row">
              <span className="label">Quantity</span>
              <span className="value">{shipModal.quantity_to_send?.toLocaleString()} kg</span>
            </div>
            <div className="da-modal-row">
              <span className="label">Distance</span>
              <span className="value">{Math.round(shipModal.distance_km)} km</span>
            </div>
            <div className="da-modal-row">
              <span className="label">Transport Cost</span>
              <span className="value">₹{shipModal.transport_total?.toLocaleString()}</span>
            </div>
            <div className="da-modal-row">
              <span className="label">ROI</span>
              <span className="value" style={{
                color: shipModal.roi >= 1.5 ? '#22c55e' : shipModal.roi >= 1 ? '#f59e0b' : '#ef4444'
              }}>
                {shipModal.roi}x
              </span>
            </div>
            <div className="da-modal-row" style={{ borderBottom: 'none' }}>
              <span className="label">Timing</span>
              <span className="value">{shipModal.timing}</span>
            </div>
            <div className="da-modal-actions">
              <button
                className="da-modal-cancel"
                onClick={() => setShipModal(null)}
                disabled={shipping}
              >
                Cancel
              </button>
              <button
                className="da-modal-confirm"
                onClick={handleShip}
                disabled={shipping}
              >
                {shipping ? '⏳ Sending...' : '✅ Confirm & Ship'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}