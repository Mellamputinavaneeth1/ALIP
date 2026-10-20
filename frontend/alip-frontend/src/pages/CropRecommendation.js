import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getCropRecommendation, getCropRecommendationManual, getCropRecommendationRegional } from '../services/api';
import toast from 'react-hot-toast';
import './CropRecommendation.css';

// ─── Constants ────────────────────────────────────────────────────────────────
const SEASONS = [
  { value: 'kharif', label: 'Kharif',  icon: '🌧️', period: 'Jun – Nov', rainfall: 200 },
  { value: 'rabi',   label: 'Rabi',    icon: '❄️', period: 'Nov – Apr', rainfall: 50  },
  { value: 'zaid',   label: 'Zaid',    icon: '☀️', period: 'Apr – Jun', rainfall: 30  },
];

const REG_SEASONS = [
  { value: 'kharif', label: 'Kharif', icon: '🌧️', period: 'Jun – Nov' },
  { value: 'rabi',   label: 'Rabi',   icon: '❄️', period: 'Nov – Apr' },
];

const FIELD_RANGES = {
  nitrogen:    { min: 0,   max: 140, unit: 'mg/kg', label: 'Nitrogen (N)',     color: '#22c55e', icon: '🟢' },
  phosphorous: { min: 0,   max: 145, unit: 'mg/kg', label: 'Phosphorous (P)',  color: '#3b82f6', icon: '🔵' },
  potassium:   { min: 0,   max: 205, unit: 'mg/kg', label: 'Potassium (K)',    color: '#a855f7', icon: '🟣' },
  temperature: { min: 8,   max: 44,  unit: '°C',    label: 'Temperature',      color: '#f97316', icon: '🌡️' },
  humidity:    { min: 14,  max: 100, unit: '%',      label: 'Humidity',         color: '#06b6d4', icon: '💧' },
  ph:          { min: 3.5, max: 9.5, unit: 'pH',     label: 'Soil pH',          color: '#eab308', icon: '⚗️' },
  rainfall:    { min: 20,  max: 300, unit: 'mm',     label: 'Rainfall',         color: '#0ea5e9', icon: '🌧️' },
};

const DEFAULT_MANUAL = {
  nitrogen: 50, phosphorous: 25, potassium: 30,
  temperature: 28, humidity: 65, ph: 6.8, rainfall: 100,
};

const CROP_EMOJIS = {
  rice: '🌾', wheat: '🌾', maize: '🌽', soybean: '🌱', cotton: '🌿',
  groundnut: '🥜', pigeonpeas: '🫘', mungbean: '🫛', chickpea: '🫘',
  mustard: '🌼', lentil: '🫘', barley: '🌾', jowar: '🌾', bajra: '🌾', ragi: '🌾',
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
const pct = (val, min, max) => Math.min(100, Math.max(0, ((val - min) / (max - min)) * 100));

const demandClass = (d = '') => {
  const key = d.toLowerCase().replace(' ', '-');
  if (key.includes('very')) return 'very-high';
  if (key.includes('high')) return 'high';
  if (key.includes('medium')) return 'medium';
  return 'low';
};

const shapImpactClass = (impact = '') => impact.toLowerCase().replace(' ', '-');
const confDeg = (conf) => `${(conf / 100) * 360}deg`;
const capitalize = (s) => s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
const confBandClass = (band = 'medium') => band.toLowerCase();

const FACTOR_STYLES = {
  rainfall:    { icon: '🌧️', color: '#38bdf8' },
  temperature: { icon: '🌡️', color: '#fb923c' },
  soil_ph:     { icon: '⚗️',  color: '#facc15' },
  soil_oc:     { icon: '🌱',  color: '#4ade80' },
  soil_clay:   { icon: '🪨',  color: '#a78bfa' },
  soil_cec:    { icon: '🔬',  color: '#c084fc' },
  irrigation:  { icon: '💧',  color: '#818cf8' },
  default_f:   { icon: '📊',  color: '#8b5cf6' },
};

const getFactorStyle = (reason = '') => {
  const lower = reason.toLowerCase();
  if (lower.includes('rain') || lower.includes('precipitat')) return FACTOR_STYLES.rainfall;
  if (lower.includes('temp')) return FACTOR_STYLES.temperature;
  if (lower.includes('ph'))   return FACTOR_STYLES.soil_ph;
  if (lower.includes('organic') || lower.includes(' oc')) return FACTOR_STYLES.soil_oc;
  if (lower.includes('clay'))  return FACTOR_STYLES.soil_clay;
  if (lower.includes('cec'))   return FACTOR_STYLES.soil_cec;
  if (lower.includes('irrig')) return FACTOR_STYLES.irrigation;
  return FACTOR_STYLES.default_f;
};

// ─── Sub-components (existing) ────────────────────────────────────────────────

function SoilBar({ value, field }) {
  const r = FIELD_RANGES[field];
  const p = pct(value, r.min, r.max);
  return (
    <div className="cr-soil-item">
      <div className="cr-soil-label">{r.icon} {r.label}</div>
      <div className="cr-soil-value">{value}<span className="cr-soil-unit"> {r.unit}</span></div>
      <div className="cr-soil-bar-wrap">
        <div className="cr-soil-bar" style={{ width: `${p}%`, background: `linear-gradient(90deg, ${r.color}, ${r.color}aa)` }} />
      </div>
    </div>
  );
}

function ShapPanel({ shap }) {
  if (!shap?.top_factors?.length) return null;
  return (
    <div className="cr-shap-panel">
      <div className="cr-shap-title">🧠 Why This Crop? — AI Explanation (SHAP)</div>
      {shap.top_factors.map((f, i) => (
        <div key={i} className="cr-shap-row">
          <div className="cr-shap-row-top">
            <span className="cr-shap-feat">{f.feature}</span>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span className="cr-shap-val">{f.raw_value} {FIELD_RANGES[f.feature_key]?.unit || ''}</span>
              <span className={`cr-shap-impact ${shapImpactClass(f.impact)}`}>{f.impact}</span>
            </div>
          </div>
          <div className="cr-shap-bar-bg">
            <div className="cr-shap-bar" style={{ width: `${f.bar_pct}%` }} />
          </div>
        </div>
      ))}
      <p style={{ margin: '10px 0 0', fontSize: 11, color: '#475569' }}>
        These are the top 3 soil/climate factors that most influenced this recommendation.
      </p>
    </div>
  );
}

function HeroCard({ rec, season }) {
  const agro = rec.agronomy || {};
  const confD = confDeg(rec.confidence);

  return (
    <div className="cr-hero">
      <div className="cr-hero-top">
        <span className="cr-hero-emoji">{agro.emoji || '🌿'}</span>
        <div className="cr-hero-info">
          <div className="cr-hero-label">🏆 Top Recommendation</div>
          <h2 className="cr-hero-crop">{rec.crop}</h2>
          <p className="cr-hero-desc">{rec.description}</p>
        </div>
        <div className="cr-hero-confidence">
          <div className="cr-conf-ring" style={{ '--conf-deg': confD }}>
            <span className="cr-conf-val">{rec.confidence.toFixed(0)}%</span>
          </div>
          <div className="cr-conf-lbl">Match Score</div>
        </div>
      </div>

      {agro.sowing_months && (
        <div className="cr-hero-agro">
          <div className="cr-agro-item">
            <div className="cr-agro-icon">📅</div>
            <div className="cr-agro-val">{agro.sowing_months}</div>
            <div className="cr-agro-lbl">Sowing Period</div>
          </div>
          <div className="cr-agro-item">
            <div className="cr-agro-icon">💧</div>
            <div className="cr-agro-val">{agro.water_need || '—'}</div>
            <div className="cr-agro-lbl">Water Need</div>
          </div>
          <div className="cr-agro-item">
            <div className="cr-agro-icon">🌾</div>
            <div className="cr-agro-val">{agro.yield_qtl_per_acre || '—'}</div>
            <div className="cr-agro-lbl">Expected Yield</div>
          </div>
          <div className="cr-agro-item">
            <div className="cr-agro-icon">💰</div>
            <div className="cr-agro-val">
              {typeof agro.msp_per_qtl === 'number'
                ? `₹${agro.msp_per_qtl}/qtl`
                : agro.msp_per_qtl || '—'}
            </div>
            <div className="cr-agro-lbl">MSP / Market</div>
          </div>
        </div>
      )}
    </div>
  );
}

function AltCard({ rec, rank }) {
  const agro = rec.agronomy || {};
  return (
    <div className="cr-alt-card">
      <div className="cr-alt-top">
        <span className="cr-alt-emoji">{agro.emoji || '🌿'}</span>
        <div>
          <div className="cr-alt-crop">#{rank} {rec.crop}</div>
          <div className="cr-alt-conf">Match: <span>{rec.confidence.toFixed(1)}%</span></div>
        </div>
      </div>
      <div className="cr-alt-divider" />
      <div className="cr-alt-info">
        <div className="cr-alt-info-item">
          <div className="cr-alt-info-lbl">Season</div>
          <div className="cr-alt-info-val">{agro.season || '—'}</div>
        </div>
        <div className="cr-alt-info-item">
          <div className="cr-alt-info-lbl">Yield</div>
          <div className="cr-alt-info-val">{agro.yield_qtl_per_acre || '—'}</div>
        </div>
        <div className="cr-alt-info-item">
          <div className="cr-alt-info-lbl">Water</div>
          <div className={`cr-water ${agro.water_need_level || 'medium'}`}>
            {agro.water_need?.split('(')[0]?.trim() || '—'}
          </div>
        </div>
        <div className="cr-alt-info-item">
          <div className="cr-alt-info-lbl">Demand</div>
          <span className={`cr-demand ${demandClass(agro.market_demand)}`}>
            {agro.market_demand || '—'}
          </span>
        </div>
      </div>
    </div>
  );
}

function AgronomyDetail({ rec }) {
  const agro = rec.agronomy || {};
  if (!agro.sowing_months) return null;

  return (
    <div className="cr-agro-panel">
      <div className="cr-panel-head">
        <div className="cr-panel-icon">📋</div>
        <h3>Agronomy Guide — {rec.crop.charAt(0).toUpperCase() + rec.crop.slice(1)}</h3>
      </div>
      <div className="cr-agro-grid">
        <div className="cr-agro-card">
          <div className="cr-agro-card-icon">📅</div>
          <div className="cr-agro-card-label">Sowing Period</div>
          <div className="cr-agro-card-value">{agro.sowing_months}</div>
        </div>
        <div className="cr-agro-card">
          <div className="cr-agro-card-icon">🌾</div>
          <div className="cr-agro-card-label">Harvest Period</div>
          <div className="cr-agro-card-value">{agro.harvest_months || '—'}</div>
        </div>
        <div className="cr-agro-card">
          <div className="cr-agro-card-icon">⏱️</div>
          <div className="cr-agro-card-label">Crop Duration</div>
          <div className="cr-agro-card-value">{agro.duration_days || '—'}</div>
        </div>
        <div className="cr-agro-card">
          <div className="cr-agro-card-icon">💧</div>
          <div className="cr-agro-card-label">Water Requirement</div>
          <div className="cr-agro-card-value">{agro.water_need || '—'}</div>
        </div>
        <div className="cr-agro-card">
          <div className="cr-agro-card-icon">📊</div>
          <div className="cr-agro-card-label">Expected Yield</div>
          <div className="cr-agro-card-value">{agro.yield_qtl_per_acre || '—'}</div>
        </div>
        <div className="cr-agro-card">
          <div className="cr-agro-card-icon">💰</div>
          <div className="cr-agro-card-label">MSP / Market Price</div>
          <div className="cr-agro-card-value">
            {typeof agro.msp_per_qtl === 'number' ? `₹${agro.msp_per_qtl} / qtl` : (agro.msp_per_qtl || '—')}
          </div>
        </div>
        <div className="cr-agro-card">
          <div className="cr-agro-card-icon">🌡️</div>
          <div className="cr-agro-card-label">Ideal Temperature</div>
          <div className="cr-agro-card-value">{agro.ideal_temp || '—'}</div>
        </div>
        <div className="cr-agro-card">
          <div className="cr-agro-card-icon">⚗️</div>
          <div className="cr-agro-card-label">Ideal Soil pH</div>
          <div className="cr-agro-card-value">{agro.ideal_ph || '—'}</div>
        </div>
        <div className="cr-agro-card">
          <div className="cr-agro-card-icon">📈</div>
          <div className="cr-agro-card-label">Market Demand</div>
          <div className="cr-agro-card-value">
            <span className={`cr-demand ${demandClass(agro.market_demand)}`}>{agro.market_demand || '—'}</span>
          </div>
        </div>
      </div>
      {agro.tips && (
        <div className="cr-tip-box">
          💡 <strong>Farming Tip:</strong> {agro.tips}
        </div>
      )}
      {agro.fertilizer_hint && (
        <div className="cr-tip-box" style={{ marginTop: 8 }}>
          🌿 <strong>Fertilizer Guide:</strong> {agro.fertilizer_hint}
        </div>
      )}
    </div>
  );
}

// ─── Regional AI Sub-components ──────────────────────────────────────────────

function RegionalEmptyState() {
  return (
    <div className="cr-panel cr-reg-empty">
      <div className="cr-reg-empty-icon">🗺️</div>
      <h3>Regional AI Recommendation</h3>
      <p>
        District-level crop ranking using historical agricultural records, climate normals, soil profiles, and irrigation data.
      </p>
      <div className="cr-reg-stats">
        <div className="cr-reg-stat">
          <div className="cr-reg-stat-val">AI</div>
          <div className="cr-reg-stat-lbl">Powered</div>
        </div>
        <div className="cr-reg-stat">
          <div className="cr-reg-stat-val">Rankings</div>
          <div className="cr-reg-stat-lbl">Regional Model</div>
        </div>
      </div>
    </div>
  );
}

function ContextStrip({ ctx }) {
  if (!ctx) return null;
  return (
    <div className="cr-ctx-strip">
      <div className="cr-ctx-item">
        <div className="cr-ctx-icon">🌡️</div>
        <div className="cr-ctx-val">
          {ctx.avg_temperature != null ? `${parseFloat(ctx.avg_temperature).toFixed(1)}°C` : '—'}
        </div>
        <div className="cr-ctx-lbl">10-YR AVG TEMP</div>
      </div>
      <div className="cr-ctx-item">
        <div className="cr-ctx-icon">🌧️</div>
        <div className="cr-ctx-val">
          {ctx.seasonal_rainfall != null ? `${Math.round(ctx.seasonal_rainfall)} mm` : '—'}
        </div>
        <div className="cr-ctx-lbl">10-YR SEASONAL RAIN</div>
      </div>
      <div className="cr-ctx-item">
        <div className="cr-ctx-icon">⚗️</div>
        <div className="cr-ctx-val">
          {ctx.soil_ph != null ? `pH ${parseFloat(ctx.soil_ph).toFixed(2)}` : '—'}
        </div>
        <div className="cr-ctx-lbl">SOIL pH</div>
      </div>
      <div className="cr-ctx-item">
        <div className="cr-ctx-icon">💧</div>
        <div className="cr-ctx-val">
          {ctx.irrigation_share != null ? `${(ctx.irrigation_share * 100).toFixed(1)}%` : '—'}
        </div>
        <div className="cr-ctx-lbl">IRRIGATED AREA</div>
      </div>
    </div>
  );
}

function RegionalHeroCard({ rec }) {
  const emoji = CROP_EMOJIS[rec.crop] || '🌿';
  const band  = rec.confidence_band || 'medium';
  const hybridScore = rec.hybrid_rank_score || rec.score || 0;
  const mlProb = rec.ml_probability || 0;
  const priorScore = rec.prior_score || 0;
  const score = Math.round(hybridScore * 100);
  const deg   = `${Math.min(score, 100) * 3.6}deg`;

  return (
    <div className="cr-regional-hero">
      <div className="cr-hero-top">
        <span className="cr-hero-emoji" style={{ fontSize: 52 }}>{emoji}</span>
        <div className="cr-hero-info">
          <div className="cr-hero-label" style={{ color: '#8b5cf6' }}>🏆 #{rec.rank || 1} RECOMMENDATION</div>
          <h2 className="cr-hero-crop">{capitalize(rec.crop)}</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 6 }}>
            <span className={`cr-conf-band ${confBandClass(band)}`}>
              ● {capitalize(band)} Confidence
            </span>
            <span style={{ fontSize: 12, color: '#475569' }}>
              Rank Score: {hybridScore.toFixed(3)}
            </span>
          </div>
        </div>
        <div className="cr-hero-confidence">
          <div className="cr-conf-ring" style={{ '--conf-deg': deg }}>
            <span className="cr-conf-val" style={{ color: '#a78bfa', fontSize: 14 }}>{hybridScore.toFixed(2)}</span>
          </div>
          <div className="cr-conf-lbl">Rank Score</div>
          <div style={{ fontSize: 10, color: '#64748b', marginTop: 4, textAlign: 'center' }}>
            ML: {mlProb.toFixed(2)} | Prior: {priorScore.toFixed(2)}
          </div>
        </div>
      </div>
      <div className="cr-reg-score-disclaimer">
        Rank Score combines calibrated ML probability with historical district-season crop frequency. It is not a yield prediction or farming success guarantee.
      </div>

      {rec.guardrail_warnings && rec.guardrail_warnings.length > 0 && (
        <div style={{ marginTop: 10, padding: '8px 12px', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)', borderRadius: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#f59e0b', marginBottom: 4 }}>Guardrail Notes:</div>
          {rec.guardrail_warnings.map((w, i) => (
            <div key={i} style={{ fontSize: 11, color: '#94a3b8', paddingLeft: 8 }}>{typeof w === 'string' ? w : w.reason || JSON.stringify(w)}</div>
          ))}
        </div>
      )}

      {rec.reasons && rec.reasons.length > 0 && (
        <div className="cr-reg-reasons">
          <div className="cr-reg-reasons-title">🧠 WHY THIS CROP — AI FACTORS (SHAP)</div>
          {rec.reasons.map((reason, i) => {
            const fs = getFactorStyle(reason);
            return (
              <div key={i} className="cr-reg-reason-row">
                <span className="cr-reg-factor-icon" style={{ color: fs.color }}>{fs.icon}</span>
                <div className="cr-reg-reason-bar" style={{ borderLeftColor: fs.color }} />
                <div className="cr-reg-reason-text">{reason}</div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function RegionalAltCard({ rec, rank }) {
  const emoji = CROP_EMOJIS[rec.crop] || '🌿';
  const band  = rec.confidence_band || 'low';
  const hybridScore = rec.hybrid_rank_score || rec.score || 0;

  return (
    <div className="cr-reg-alt-card">
      <div className="cr-alt-top">
        <span className="cr-alt-emoji">{emoji}</span>
        <div>
          <div className="cr-alt-rank">#{rec.rank || rank} RECOMMENDATION</div>
          <div className="cr-alt-crop-name">{capitalize(rec.crop)}</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
            <span style={{ fontSize: 12, color: '#8b5cf6', fontWeight: 600 }}>
              Rank Score: {hybridScore.toFixed(3)}
            </span>
            <span className={`cr-conf-band ${confBandClass(band)}`} style={{ fontSize: 10, padding: '2px 7px' }}>
              {capitalize(band)}
            </span>
          </div>
        </div>
      </div>
      <div className="cr-alt-divider" />
      {rec.reasons && rec.reasons.length > 0 && (
        <div className="cr-reg-alt-reasons">
          {rec.reasons.slice(0, 2).map((r, i) => (
            <div key={i} style={{ display: 'flex', gap: 6, marginBottom: 5, alignItems: 'flex-start' }}>
              <span style={{ color: '#8b5cf6', fontSize: 10, marginTop: 3 }}>✔</span>
              <span style={{ fontSize: 11, color: '#64748b', lineHeight: 1.5 }}>{r}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AbstainBanner({ reason, recs }) {
  return (
    <div className="cr-abstain">
      <div style={{ fontSize: 28 }}>⚠️</div>
      <div>
        <div className="cr-abstain-title">Low Confidence — Showing Best Estimates</div>
        <div className="cr-abstain-msg" style={{ marginBottom: recs?.length ? 12 : 0 }}>
          {reason || 'Insufficient district data for a confident recommendation.'}
        </div>
        {recs && recs.length > 0 && (
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {recs.map((r, i) => (
              <div key={r.crop} className="cr-abstain-crop-chip">
                {CROP_EMOJIS[r.crop] || '🌿'} {capitalize(r.crop)}
                <span style={{ marginLeft: 6, fontSize: 10, opacity: 0.7 }}>
                  {(r.hybrid_rank_score || r.score || 0).toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ConfBandLegend() {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="cr-band-legend-wrap">
      <button
        className="cr-band-legend-toggle"
        onClick={() => setOpen(o => !o)}
      >
        {open ? '▲' : 'ℹ️'} What do these confidence bands mean?
      </button>
      {open && (
        <div className="cr-band-legend-popup">
          <div className="cr-band-legend-title">Confidence Band Definitions</div>
          <div className="cr-band-legend-row">
            <span className="cr-conf-band high">High</span>
            <span>Score 0.70+ — Crop dominates this district-season historically</span>
          </div>
          <div className="cr-band-legend-row">
            <span className="cr-conf-band medium">Medium</span>
            <span>Score 0.40–0.70 — Widely cultivated but secondary to dominant crops</span>
          </div>
          <div className="cr-band-legend-row">
            <span className="cr-conf-band low">Low</span>
            <span>Score below 0.40 — Marginal or infrequent in this district</span>
          </div>
          <div className="cr-band-legend-note">
            Scores are calibrated historical probabilities, not yield guarantees.
          </div>
        </div>
      )}
    </div>
  );
}

function AllCropsAccordion({ result }) {
  const [open, setOpen] = React.useState(false);
  const candidates = result.all_candidates || result.recommendations || [];
  if (candidates.length <= 3) return null;

  const remaining = candidates.slice(3);

  return (
    <div className="cr-all-crops-wrap">
      <button
        className="cr-all-crops-toggle"
        onClick={() => setOpen(o => !o)}
      >
        {open ? '▲ Hide' : '▼ View'} All {candidates.length} Evaluated Field Crops
      </button>
      {open && (
        <div className="cr-all-crops-table">
          <div className="cr-all-crops-head">
            <span>Rank</span><span>Crop</span><span>Score</span><span>Band</span>
          </div>
          {remaining.map((rec, i) => {
            const band = rec.confidence_band || 'low';
            return (
              <div key={rec.crop} className="cr-all-crops-row">
                <span className="cr-all-crops-rank">#{i + 4}</span>
                <span className="cr-all-crops-name">
                  {CROP_EMOJIS[rec.crop] || '🌿'} {capitalize(rec.crop)}
                </span>
                <span className="cr-all-crops-score">
                  {(rec.hybrid_rank_score || rec.score || 0).toFixed(3)}
                </span>
                <span className={`cr-conf-band ${confBandClass(band)}`}>{capitalize(band)}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function RegionalResultHeader({ result }) {
  const modelLabel = result?.provenance?.model_version || 'v2-regional';
  return (
    <>
      <div className="cr-district-confirm">
        <div className="cr-district-confirm-left">
          <span className="cr-district-pin">{'📍'}</span>
          <div>
            <div className="cr-district-confirm-title">
              Active District: <strong>{result.district}</strong>
              {result.state ? `, ${result.state}` : ''}
            </div>
            <div className="cr-district-confirm-sub">
              {capitalize(result.season)} Season &middot; Model: {modelLabel}
            </div>
          </div>
        </div>
        <span className={`cr-reg-status-chip ${result.abstain ? 'uncertain' : 'confident'}`}>
          {result.abstain ? 'Uncertain' : 'Confident'}
        </span>
      </div>
      {result.abstain && (
        <div className="cr-ood-warning">
          <strong>Boundary Notice:</strong> This district has limited historical data. Verify
          recommendations with your local Krishi Vigyan Kendra (KVK).
        </div>
      )}
    </>
  );
}

function RegionalDataNote({ result }) {
  const prov = result?.provenance || {};
  const cropSource = typeof prov.crop_data === 'object' ? prov.crop_data.source : (prov.crop_data || 'Unknown');
  const climateSource = typeof prov.climate === 'object' ? prov.climate.source : (prov.climate || 'Unknown');
  const climateRes = typeof prov.climate === 'object' ? prov.climate.resolution : (prov.climate_resolution || '');
  const soilSource = typeof prov.soil === 'object' ? prov.soil.source : (prov.soil || 'Unknown');
  const soilRes = typeof prov.soil === 'object' ? prov.soil.resolution : (prov.soil_resolution || '');
  const irrSource = typeof prov.irrigation === 'object' ? prov.irrigation.source : (prov.irrigation || 'Unknown');
  const irrRes = typeof prov.irrigation === 'object' ? prov.irrigation.resolution : (prov.irrigation_resolution || '');
  const hybridNote = prov.hybrid_alpha !== undefined ? ` · Hybrid Ranker (alpha=${prov.hybrid_alpha})` : '';

  return (
    <div className="cr-data-note">
      <span>📡</span>
      <span>
        Data Provenance: {cropSource} · Climate: {climateSource} {climateRes ? `(${climateRes})` : ''} · Soil: {soilSource} {soilRes ? `(${soilRes})` : ''} · Irrigation: {irrSource} {irrRes ? `(${irrRes})` : ''}{hybridNote} · pH source: {result.ph_source || 'regional baseline'}
      </span>
      {result.ph_source === 'farmer_measured' && (
        <div style={{ marginTop: 4, color: '#a78bfa', fontSize: 11 }}>
          pH Source: Your lab-measured value (overrides regional baseline)
        </div>
      )}
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────
export default function CropRecommendation() {
  const navigate = useNavigate();

  // ── Active tab: 'auto' | 'manual' | 'regional'
  const [activeTab, setActiveTab] = useState('auto');

  // Auto / Manual shared state
  const [mode, setMode]       = useState('auto');
  const [location, setLocation] = useState('');
  const [season, setSeason]     = useState('kharif');
  const [manual, setManual]     = useState({ ...DEFAULT_MANUAL });
  const [data, setData]         = useState(null);
  const [loading, setLoading]   = useState(false);

  // Regional-specific state
  const [regLocation, setRegLocation] = useState('');
  const [regSeason, setRegSeason]     = useState('kharif');
  const [phEnabled, setPhEnabled]     = useState(false);
  const [measuredPH, setMeasuredPH]   = useState('');
  const [regLoading, setRegLoading]   = useState(false);
  const [regResult, setRegResult]     = useState(null);
  const [regLocError, setRegLocError] = useState('');
  const [regPhError,  setRegPhError]  = useState('');

  // ── Handlers ───────────────────────────────────────────────
  const handleAnalyze = async () => {
    setLoading(true);
    setData(null);
    try {
      let res;
      if (mode === 'auto') {
        if (!location.trim()) { toast.error('Please enter your location'); setLoading(false); return; }
        res = await getCropRecommendation({ location: location.trim(), season });
      } else {
        res = await getCropRecommendationManual({ ...manual, season, location: location.trim() });
      }
      setData(res.data);
      toast.success('Recommendation ready!');
    } catch (err) {
      const msg = err?.response?.data?.detail || 'Analysis failed';
      toast.error(typeof msg === 'string' ? msg : 'Analysis failed');
    }
    setLoading(false);
  };

  const handleManualChange = (field, value) => {
    setManual(prev => ({ ...prev, [field]: parseFloat(value) || 0 }));
  };

  const handleRegional = async () => {
    if (!regLocation.trim() || regLocation.trim().length < 2) {
      setRegLocError('Please enter at least a 2-character district name.');
      return;
    }
    setRegLocError('');
    if (phEnabled && measuredPH) {
      const phVal = parseFloat(measuredPH);
      if (isNaN(phVal) || phVal < 3.0 || phVal > 10.0) {
        setRegPhError('Soil pH must be between 3.0 and 10.0.');
        return;
      }
    }
    setRegPhError('');

    setRegLoading(true);
    setRegResult(null);
    try {
      const payload = {
        location:    regLocation.trim(),
        season:      regSeason,
        measured_ph: phEnabled && measuredPH ? parseFloat(measuredPH) : null,
      };
      const res = await getCropRecommendationRegional(payload);
      setRegResult(res.data);
      if (res.data.success) toast.success('Regional recommendation ready!');
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Regional recommendation failed');
    } finally {
      setRegLoading(false);
    }
  };

  const selectedSeason   = SEASONS.find(s => s.value === season);
  const displayRainfall  = selectedSeason?.rainfall || 100;

  // Sync mode state with activeTab for Auto/Manual
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === 'auto')   setMode('auto');
    if (tab === 'manual') setMode('manual');
  };

  // ── Render ─────────────────────────────────────────────────
  return (
    <div className="cr-page">
      {/* Header */}
      <div className="cr-header">
        <button className="cr-back" onClick={() => navigate('/farmer')}>← Back</button>
        <h1>🌱 Crop Intelligence Engine</h1>
        <span className="cr-badge">AI Powered</span>
      </div>

      <div className="cr-body">
        {/* ── Left: Input Panel ─────────────────────────────── */}
        <div>
          <div className="cr-panel">
            <div className="cr-panel-head">
              <div className="cr-panel-icon" style={activeTab === 'regional' ? { background: 'rgba(139,92,246,0.15)' } : {}}>
                {activeTab === 'regional' ? '🗺️' : '📍'}
              </div>
              <h3>{activeTab === 'regional' ? 'Regional AI Setup' : 'Crop Analysis Setup'}</h3>
            </div>

            {/* ── Tab Bar ── */}
            <div className="cr-tab-bar">
              <button
                className={`cr-tab ${activeTab === 'auto' ? 'active' : ''}`}
                onClick={() => handleTabChange('auto')}
                title="Quick AI mode — benchmark model using location + live weather"
              >
                🌐 Quick AI (Benchmark)
              </button>
              <button
                className={`cr-tab ${activeTab === 'manual' ? 'active' : ''}`}
                onClick={() => handleTabChange('manual')}
                title="Soil Test mode — benchmark model using manual soil test values"
              >
                🧪 Soil Test (Benchmark)
              </button>
              <button
                className={`cr-tab regional ${activeTab === 'regional' ? 'active' : ''}`}
                onClick={() => handleTabChange('regional')}
                title="Regional AI — district-level agronomic hybrid model"
              >
                🗺️ Regional AI (District Model)
                <span className="cr-tab-badge">HYBRID</span>
              </button>
            </div>

            {/* ── REGIONAL TAB INPUTS ── */}
            {activeTab === 'regional' && (
              <>
                {/* Info badge */}
                <div className="cr-regional-badge">
                  <span>🛰️</span>
                  <span>
                    Uses <strong style={{ color: '#a78bfa' }}>10-year historical climate normals</strong> (not today's weather) &middot; NBSS soil profiles &middot; DES irrigation stats &middot; Trained on Kaggle Crop Production (DES) &middot; <em>Not a yield forecast</em>
                  </span>
                </div>

                {/* District input */}
                <div className="cr-input-group">
                  <label className="cr-label">📍 DISTRICT / CITY</label>
                  <input
                    className="cr-input"
                    placeholder="e.g. Nashik, Ludhiana, Warangal..."
                    value={regLocation}
                    onChange={e => setRegLocation(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleRegional()}
                  />
                  {regLocError && <div className="cr-field-error">{regLocError}</div>}
                </div>

                {/* Season */}
                <div className="cr-input-group">
                  <label className="cr-label">🗓️ GROWING SEASON</label>
                  <div className="cr-season-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                    {REG_SEASONS.map(s => (
                      <div
                        key={s.value}
                        className={`cr-season-btn ${regSeason === s.value ? 'active' : ''}`}
                        onClick={() => setRegSeason(s.value)}
                      >
                        <span className="cr-season-icon">{s.icon}</span>
                        <div className="cr-season-name">{s.label}</div>
                        <div className="cr-season-period">{s.period}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Optional pH toggle */}
                <div className="cr-input-group">
                  <label className="cr-label">⚗️ SOIL pH OVERRIDE (OPTIONAL)</label>
                  <div
                    className="cr-ph-toggle"
                    onClick={() => setPhEnabled(p => !p)}
                    title="Toggle to use your lab-measured pH instead of regional data"
                  >
                    <div className={`cr-toggle-sw ${phEnabled ? 'on' : 'off'}`}>
                      <div className="cr-toggle-knob" style={{ left: phEnabled ? 20 : 2 }} />
                    </div>
                    <span style={{ fontSize: 12, color: phEnabled ? '#a78bfa' : '#475569' }}>
                      {phEnabled ? 'Enabled — override regional baseline pH' : 'Disabled — using regional baseline pH'}
                    </span>
                  </div>
                  {phEnabled && (
                    <>
                      <input
                        className="cr-input"
                        type="number"
                        min="3.0"
                        max="9.5"
                        step="0.1"
                        placeholder="Enter soil pH (3.0 – 9.5)"
                        value={measuredPH}
                        onChange={e => setMeasuredPH(e.target.value)}
                      />
                      {regPhError && <div className="cr-field-error">{regPhError}</div>}
                      <div style={{ fontSize: 11, color: '#475569', marginTop: 5 }}>
                        📊 Replaces regional soil pH with your lab-test value
                      </div>
                    </>
                  )}
                </div>

                {/* Analyze button */}
                <button
                  className="cr-analyze-btn cr-analyze-regional"
                  onClick={handleRegional}
                  disabled={regLoading || !!regLocError}
                >
                  {regLoading ? '⏳ Analyzing...' : '🗺️ Get Regional Recommendation'}
                </button>

                <div className="cr-data-note" style={{ marginTop: 14 }}>
                  <span>📡</span>
                  <span>Kaggle Crop Production (DES) · Open-Meteo climate · NBSS soil · DES irrigation</span>
                </div>
              </>
            )}

            {/* ── AUTO / MANUAL TAB INPUTS ── */}
            {activeTab !== 'regional' && (
              <>
                {/* Location */}
                <div className="cr-input-group">
                  <label className="cr-label">📍 Location</label>
                  <input
                    className="cr-input"
                    placeholder="e.g. Hyderabad, Punjab, Tamil Nadu..."
                    value={location}
                    onChange={e => setLocation(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleAnalyze()}
                  />
                </div>

                {/* Season */}
                <div className="cr-input-group">
                  <label className="cr-label">🗓️ Growing Season</label>
                  <div className="cr-season-grid">
                    {SEASONS.map(s => (
                      <div
                        key={s.value}
                        className={`cr-season-btn ${season === s.value ? 'active' : ''}`}
                        onClick={() => setSeason(s.value)}
                      >
                        <span className="cr-season-icon">{s.icon}</span>
                        <div className="cr-season-name">{s.label}</div>
                        <div className="cr-season-period">{s.period}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Manual soil inputs */}
                {mode === 'manual' && (
                  <div className="cr-input-group">
                    <label className="cr-label">🧪 Soil Test Values</label>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                      {['nitrogen', 'phosphorous', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall'].map(field => {
                        const r = FIELD_RANGES[field];
                        return (
                          <div key={field}>
                            <label className="cr-label" style={{ marginBottom: 3 }}>
                              {r.icon} {r.label} ({r.unit})
                            </label>
                            <input
                              type="number"
                              className="cr-input"
                              min={r.min} max={r.max} step="0.1"
                              value={manual[field]}
                              onChange={e => handleManualChange(field, e.target.value)}
                            />
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Auto mode weather preview */}
                {mode === 'auto' && location && (
                  <div style={{ marginBottom: 14 }}>
                    <label className="cr-label">📊 Weather Inputs (Auto-fetched)</label>
                    <div className="cr-weather-bar">
                      <div className="cr-weather-chip">
                        <div className="cr-weather-chip-icon">🌡️</div>
                        <div className="cr-weather-chip-val">~28°C</div>
                        <div className="cr-weather-chip-lbl">Temp</div>
                      </div>
                      <div className="cr-weather-chip">
                        <div className="cr-weather-chip-icon">💧</div>
                        <div className="cr-weather-chip-val">~65%</div>
                        <div className="cr-weather-chip-lbl">Humidity</div>
                      </div>
                      <div className="cr-weather-chip">
                        <div className="cr-weather-chip-icon">🌧️</div>
                        <div className="cr-weather-chip-val">{displayRainfall}mm</div>
                        <div className="cr-weather-chip-lbl">Rainfall</div>
                      </div>
                    </div>
                  </div>
                )}

                <button
                  className="cr-analyze-btn"
                  onClick={handleAnalyze}
                  disabled={loading}
                >
                  {loading ? '⏳ Analyzing...' : '🌱 Get Recommendation'}
                </button>
              </>
            )}
          </div>

          {/* Soil summary (after Auto/Manual results) */}
          {data && activeTab !== 'regional' && (
            <div className="cr-panel">
              <div className="cr-panel-head">
                <div className="cr-panel-icon">🪱</div>
                <h3>Soil Profile Used</h3>
              </div>
              <div style={{ marginBottom: 6, fontSize: 12, color: '#64748b' }}>
                Type: <span style={{ color: '#94a3b8', fontWeight: 600 }}>{data.soil_type}</span>
                {' · '} Source: <span style={{ color: '#94a3b8', fontWeight: 600 }}>
                  {data.input_mode === 'manual' ? 'Manual Input' : 'Regional Database'}
                </span>
              </div>
              <div className="cr-soil-grid">
                <SoilBar value={data.soil_summary.nitrogen}    field="nitrogen" />
                <SoilBar value={data.soil_summary.phosphorous} field="phosphorous" />
                <SoilBar value={data.soil_summary.potassium}   field="potassium" />
                <SoilBar value={data.soil_summary.ph}          field="ph" />
                <SoilBar value={data.weather_inputs.temperature} field="temperature" />
                <SoilBar value={data.weather_inputs.humidity}    field="humidity" />
                <SoilBar value={data.weather_inputs.rainfall}    field="rainfall" />
              </div>
              {data.weather_source && data.weather_source !== 'manual' && (
                <p style={{ fontSize: 11, color: '#475569', margin: '10px 0 0' }}>
                  🌐 Weather: {data.weather_source === 'openweathermap' ? 'Live from OpenWeatherMap' : 'Fallback estimate'}
                </p>
              )}
            </div>
          )}
        </div>

        {/* ── Right: Results Panel ──────────────────────────── */}
        <div>
          {/* ── AUTO / MANUAL RESULTS ── */}
          {activeTab !== 'regional' && (
            <>
              {loading && (
                <div className="cr-loading">
                  <div className="cr-spinner" />
                  <span className="cr-loading-text">Running AI Crop Intelligence Analysis...</span>
                </div>
              )}
              {!loading && !data && (
                <div className="cr-empty">
                  <div className="cr-empty-icon">🌱</div>
                  <h3>No Analysis Yet</h3>
                  <p>Enter your location and season (or soil test values) to get your AI-powered crop recommendation.</p>
                </div>
              )}
              {!loading && data && data.success && (
                <>
                  <HeroCard rec={data.recommendations[0]} season={data.season} />
                  <ShapPanel shap={data.shap_explanation} />
                  {data.recommendations.length > 1 && (
                    <div className="cr-panel">
                      <div className="cr-panel-head">
                        <div className="cr-panel-icon" style={{ background: 'rgba(59,130,246,0.1)' }}>🥈</div>
                        <h3>Alternative Recommendations</h3>
                      </div>
                      <div className="cr-alt-grid">
                        {data.recommendations.slice(1).map((rec, i) => (
                          <AltCard key={i} rec={rec} rank={i + 2} />
                        ))}
                      </div>
                    </div>
                  )}
                  <AgronomyDetail rec={data.recommendations[0]} />
                </>
              )}
              {!loading && data && !data.success && (
                <div className="cr-panel" style={{ textAlign: 'center', padding: 40 }}>
                  <div style={{ fontSize: 40, marginBottom: 12 }}>⚠️</div>
                  <h3 style={{ color: '#ef4444', margin: '0 0 8px' }}>Analysis Failed</h3>
                  <p style={{ color: '#64748b', fontSize: 14 }}>{data.error}</p>
                </div>
              )}
            </>
          )}

          {/* ── REGIONAL AI RESULTS ── */}
          {activeTab === 'regional' && (
            <div className="cr-regional-result">
              {regLoading && (
                <div className="cr-panel cr-reg-loading">
                  <div className="cr-reg-loading-icon">🗺️</div>
                  <div className="cr-reg-loading-title">Analyzing regional data for {regLocation}...</div>
                  <div className="cr-reg-loading-sub">
                    Fetching district features · Scoring 15 crops · Applying agronomic guardrails
                  </div>
                  <div className="cr-reg-progress-wrap">
                    <div className="cr-reg-progress-bar" />
                  </div>
                </div>
              )}

              {!regLoading && !regResult && <RegionalEmptyState />}

              {!regLoading && regResult && regResult.success && (
                <>
                  <RegionalResultHeader result={regResult} />
                  <ContextStrip ctx={regResult.context} />
                  
                  {phEnabled && measuredPH && regResult?.context?.soil_ph && (
                    <div className="cr-ph-compare">
                      <div className="cr-ph-compare-item">
                        <div className="cr-ph-compare-label">Regional Baseline</div>
                        <div className="cr-ph-compare-val cr-ph-inactive">pH {parseFloat(regResult.context.soil_ph).toFixed(2)}</div>
                      </div>
                      <div className="cr-ph-compare-arrow">&#8594;</div>
                      <div className="cr-ph-compare-item">
                        <div className="cr-ph-compare-label">Your Lab Test (Active)</div>
                        <div className="cr-ph-compare-val cr-ph-active">pH {parseFloat(measuredPH).toFixed(2)}</div>
                      </div>
                    </div>
                  )}

                  {regResult.abstain ? (
                    <AbstainBanner reason={regResult.abstain_reason} recs={regResult.recommendations} />
                  ) : (
                    <>
                      {regResult.recommendations.length > 0 && (
                        <>
                          <RegionalHeroCard rec={regResult.recommendations[0]} />
                          <ConfBandLegend />
                        </>
                      )}
                      {regResult.recommendations.length > 1 && (
                        <div className="cr-alt-grid">
                          {regResult.recommendations.slice(1).map((r, i) => (
                            <RegionalAltCard key={r.crop} rec={r} rank={i + 2} />
                          ))}
                        </div>
                      )}
                      <AllCropsAccordion result={regResult} />
                    </>
                  )}

                  <RegionalDataNote result={regResult} />
                </>
              )}

              {!regLoading && regResult && !regResult.success && (
                <div className="cr-panel" style={{ textAlign: 'center', padding: 40 }}>
                  <div style={{ fontSize: 40, marginBottom: 12 }}>⚠️</div>
                  <h3 style={{ color: '#ef4444', margin: '0 0 8px' }}>Regional Analysis Failed</h3>
                  <p style={{ color: '#64748b', fontSize: 14 }}>{regResult.error}</p>
                  <p style={{ fontSize: 12, color: '#475569', marginTop: 8 }}>
                    Try: Nashik, Ludhiana, Warangal, Indore, Patna, or any of 644 supported districts.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}