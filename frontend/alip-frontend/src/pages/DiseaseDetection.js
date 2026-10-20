import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { predictDisease } from '../services/api';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Shield, Beaker, HelpCircle, Layers, MessageCircle, ChevronDown, X, Clock, Leaf, AlertTriangle, TrendingUp, MapPin } from 'lucide-react';
import AnimatedBackground from '../components/AnimatedBackground';
import './DiseaseDetection.css';

/* ══════════════════════════════════════════════════
   HELPERS – Parse AI report into sections
   ══════════════════════════════════════════════════ */
function splitScreens(report) {
  if (!report) return { first: '', second: '', third: '' };
  const f = report.indexOf('---FIRST_SCREEN---');
  const s = report.indexOf('---SECOND_SCREEN---');
  const t = report.indexOf('---THIRD_SCREEN---');
  if (f === -1 || s === -1 || t === -1) {
    // If delimiters are missing or incomplete, search the whole report for all sections
    return { first: report, second: report, third: report };
  }
  return {
    first:  report.substring(f + 18, s).trim(),
    second: report.substring(s + 19, t).trim(),
    third:  report.substring(t + 18).trim(),
  };
}

function extractSection(md, keyword) {
  if (!md) return '';
  const lines = md.split('\n');
  let capturing = false, result = [];
  const kw = keyword.toLowerCase();
  for (const line of lines) {
    const trimmed = line.trim();
    if (/^#{1,4}\s+/.test(trimmed)) {
      if (capturing) break;
      if (trimmed.toLowerCase().includes(kw)) capturing = true;
      continue;
    }
    if (capturing) result.push(line);
  }
  return result.join('\n').trim();
}

function parseMedicines(md) {
  const raw = extractSection(md, 'Recommended Medicines') || extractSection(md, 'Recommended Products') || extractSection(md, 'Medicine');
  if (!raw) return [];
  const meds = [];
  const lines = raw.split('\n');
  let current = null;
  for (const line of lines) {
    const trimmed = line.trim();
    const rankMatch = trimmed.match(/^(?:[-*•]\s*)?(?:(🥇|🥈|🥉)|(\d+)[\.)])\s*\*\*(.+?)\*\*/);
    if (rankMatch) {
      if (current) meds.push(current);
      const medal = rankMatch[1];
      const num = rankMatch[2];
      const rank = medal === '🥇' || num === '1' ? '1st' : (medal === '🥈' || num === '2' ? '2nd' : '3rd');
      current = { rank, name: rankMatch[3], details: [] };
    } else if (current && (trimmed.startsWith('-') || trimmed.startsWith('*') || trimmed.startsWith('•'))) {
      current.details.push(trimmed.replace(/^[-*•]\s*/, ''));
    }
  }
  if (current) meds.push(current);
  return meds;
}

function parseActionPlan(md) {
  const raw = extractSection(md, 'Action Plan') || extractSection(md, "Today's Action");
  if (!raw) return [];
  return raw.split('\n')
    .map(l => l.trim())
    .filter(Boolean)
    .map(l => l.replace(/^[0-9️⃣\d]+[\.\)]\s*/, '').replace(/^[-*•]\s*/, '').trim())
    .filter(Boolean)
    .slice(0, 6);
}

function parseBullets(md, keyword) {
  const raw = extractSection(md, keyword);
  if (!raw) return [];
  return raw.split('\n')
    .map(l => l.trim())
    .map(l => l.replace(/^[-•*✔☁🌧💧🌱🌾✅\d\.)]+\s*/, '').trim())
    .filter(Boolean)
    .slice(0, 8);
}

function parseFaq(md) {
  const raw = extractSection(md, 'Frequently Asked') || extractSection(md, 'FAQ');
  if (!raw) return [];
  const items = [];
  const lines = raw.split('\n');
  let q = null, a = [];
  for (const line of lines) {
    const trimmed = line.trim();
    const qMatch = trimmed.match(/^\*?\*?(?:❓|\d+[\.\)]|Q:?)\s*(.+?)\*?\*?$/i) || trimmed.match(/\*\*(.+?\?)\*\*/);
    if (qMatch && (trimmed.includes('?') || trimmed.toLowerCase().startsWith('q:') || trimmed.includes('❓'))) {
      if (q) items.push({ q, a: a.join('\n').trim() });
      q = qMatch[1].replace(/^[❓\d\.\)\s]+/, '').replace(/\*+/g, '').trim();
      if (!q.endsWith('?')) q += '?';
      a = [];
    } else if (trimmed && q) {
      a.push(trimmed);
    }
  }
  if (q) items.push({ q, a: a.join('\n').trim() });
  return items;
}

function parseRecovery(md) {
  const raw = extractSection(md, 'Recovery Timeline') || extractSection(md, 'Timeline');
  if (!raw) return [];
  const steps = [];
  const lines = raw.split('\n');
  let current = null;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.toLowerCase().includes('treatment step:')) {
      if (current) steps.push(current);
      current = { step: trimmed.split(/:\s*/)[1]?.trim() || trimmed, time: '', outcome: '' };
    } else if (trimmed.toLowerCase().includes('estimated time:')) {
      if (current) current.time = trimmed.split(/:\s*/)[1]?.trim() || '';
    } else if (trimmed.toLowerCase().includes('expected outcome:')) {
      if (current) current.outcome = trimmed.split(/:\s*/)[1]?.trim() || '';
    }
  }
  if (current) steps.push(current);
  return steps;
}

function sevConfig(severity) {
  switch (severity) {
    case 'High':   return { color: '#ef4444', bg: 'rgba(239,68,68,0.1)', label: 'High Risk', pct: 80 };
    case 'Medium': return { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)', label: 'Moderate Risk', pct: 50 };
    case 'None':   return { color: '#22c55e', bg: 'rgba(34,197,94,0.1)',  label: 'Healthy', pct: 5 };
    default:       return { color: '#22c55e', bg: 'rgba(34,197,94,0.1)',  label: 'Low Risk', pct: 25 };
  }
}

const TAB_ICONS = { overview: Activity, treatment: Beaker, prevention: Shield, faq: HelpCircle, advanced: Layers };
const TABS = ['overview', 'treatment', 'prevention', 'faq', 'advanced'];
const ADV_TABS = ['report', 'scientific', 'recovery', 'analysis'];
const fadeVariant = { initial: { opacity: 0, y: 12 }, animate: { opacity: 1, y: 0 }, exit: { opacity: 0, y: -8 }, transition: { duration: 0.25 } };

/* ══════════════════════════════════════════════════
   MAIN COMPONENT
   ══════════════════════════════════════════════════ */
export default function DiseaseDetection() {
  const nav = useNavigate();
  const [image, setImage]     = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab]         = useState('overview');
  const [advTab, setAdvTab]   = useState('report');
  const [faqOpen, setFaqOpen] = useState(null);
  const [drawer, setDrawer]   = useState(false);
  const { user } = useAuth();
  const userLocation = user?.location || '';

  const handleImage = (e) => { const f = e.target.files[0]; if (f) { setImage(f); setPreview(URL.createObjectURL(f)); setResult(null); } };

  const handleAnalyze = async () => {
    if (!image) { toast.error('Upload an image first'); return; }
    setLoading(true);
    try {
      const fd = new FormData(); fd.append('file', image);
      if (userLocation.trim()) fd.append('location', userLocation.trim());
      const res = await predictDisease(fd);
      setResult(res.data); setTab('overview');
      if (res.data?.status === 'diseased') {
        try {
          const scanItem = {
            scan_id: res.data.scan_id || ('local_' + Date.now()),
            crop: (res.data.crop || '').toLowerCase(),
            disease: res.data.disease || '',
            severity: res.data.severity || 'Medium',
            treatment: res.data.treatment || '',
            location: userLocation || '',
            timestamp: new Date().toISOString(),
            active: true,
          };
          const existing = JSON.parse(localStorage.getItem('alip_active_disease_scans') || '[]');
          const updated = [scanItem, ...existing.filter(s => s.crop !== scanItem.crop)];
          localStorage.setItem('alip_active_disease_scans', JSON.stringify(updated));
        } catch (e) {
          console.warn('LocalStorage scan save error:', e);
        }
      }
      toast.success('Analysis complete — Synced with Weather Advisory');
    } catch (err) {
      const detail = err?.response?.data?.detail;
      const msg = typeof detail === 'string' ? detail.replace('Prediction failed: ', '') : 'Detection failed';
      toast.error(msg);
    }
    setLoading(false);
  };

  /* Parsed data */
  const screens = useMemo(() => splitScreens(result?.detailed_report), [result]);
  const meds = useMemo(() => parseMedicines(screens.first || screens.second), [screens]);
  const actions = useMemo(() => parseActionPlan(screens.first), [screens]);
  const causes = useMemo(() => parseBullets(screens.second, 'Why This Happened'), [screens]);
  const prevention = useMemo(() => parseBullets(screens.second, 'Prevent'), [screens]);
  const faqItems = useMemo(() => parseFaq(screens.second), [screens]);
  const recovery = useMemo(() => parseRecovery(screens.third), [screens]);
  const sev = result ? sevConfig(result.severity) : null;

  const cropName = result?.crop || (result?.raw_class?.includes('___') ? result.raw_class.split('___')[0].replace(/_/g, ' ') : result?.raw_class?.split('_')[0]?.toUpperCase()) || '';
  const diseaseName = result?.disease
    ? (result.disease.includes(' - ') ? result.disease.split(' - ').slice(1).join(' - ') : result.disease)
    : (result?.raw_class?.split('___')[1]?.replace(/_/g, ' ') || (result?.status === 'healthy' ? 'Healthy' : 'Unknown'));

  /* ── Render ─────────────────────────────────── */
  return (
    <div className="dd-page">
      <AnimatedBackground diseaseDetected={result?.status === 'diseased'} />

      {/* Header */}
      <header className="dd-header">
        <button className="dd-header-back" onClick={() => nav('/farmer')}>← Back</button>
        <h1 className="dd-header-title">Disease Intelligence</h1>
        <span className="dd-header-badge">AI Powered</span>
      </header>

      {/* ── Upload View ──────────────────────────── */}
      {!result && (
        <div className="dd-upload-wrap">
          <div className="dd-upload-card">
            <h2 className="dd-upload-title">Analyze Crop Disease</h2>
            <p className="dd-upload-sub">Upload a clear leaf image for instant AI diagnosis</p>
            <label className="dd-upload-box">
              {preview ? <img src={preview} alt="leaf" className="dd-upload-preview" />
                : <><div className="dd-upload-icon"><Leaf size={36} strokeWidth={1.5} /></div>
                    <p style={{ margin: 0, color: '#94A3B8' }}>Click to upload leaf image</p>
                    <p className="dd-upload-hint">JPG, PNG supported</p></>}
              <input type="file" accept="image/*" onChange={handleImage} style={{ display: 'none' }} />
            </label>
            {image && <p className="dd-upload-filename">{image.name}</p>}
            {userLocation && (
              <div className="dd-location-row">
                <MapPin size={18} className="dd-location-icon" />
                <span style={{ color: 'var(--dd-text)', fontSize: 14 }}>Weather-aware analysis for <strong style={{ color: 'var(--dd-accent)' }}>{userLocation}</strong></span>
              </div>
            )}
            <button className="dd-analyze-btn" onClick={handleAnalyze} disabled={loading}>
              {loading ? 'Analyzing…' : 'Analyze Disease'}
            </button>
          </div>
        </div>
      )}

      {/* ── Dashboard View ───────────────────────── */}
      {result && (
        <div className="dd-dashboard">

          {/* ── Sidebar ──────────────────────────── */}
          <aside className="dd-sidebar">
            {preview && <img src={preview} alt="leaf" className="dd-sidebar-img" />}

            <div>
              <p className="dd-sidebar-label">Crop</p>
              <p className="dd-sidebar-value green">{cropName}</p>
              <p className="dd-sidebar-label">Disease</p>
              <p className="dd-sidebar-value">{diseaseName}</p>
            </div>

            {result.weather && (
              <>
                <div className="dd-sidebar-divider" />
                <div>
                  <p className="dd-sidebar-label">📍 Weather at {result.weather.location}</p>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
                    <span className="dd-weather-chip">🌡️ {result.weather.temperature}°C</span>
                    <span className="dd-weather-chip">💧 {result.weather.humidity}%</span>
                    <span className="dd-weather-chip">💨 {result.weather.wind_speed} km/h</span>
                    <span className="dd-weather-chip">🌧️ {result.weather.rainfall}mm</span>
                  </div>
                </div>
              </>
            )}





            <div className="dd-sidebar-divider" />

            <div>
              <p className="dd-sidebar-label">Scanned</p>
              <p className="dd-sidebar-value" style={{ fontSize: 13 }}>{new Date().toLocaleString()}</p>
            </div>

            <button className="dd-analyze-btn" style={{ marginTop: 'auto', fontSize: 13 }}
              onClick={() => { setResult(null); setImage(null); setPreview(null); }}>
              New Scan
            </button>
          </aside>

          {/* ── Content ──────────────────────────── */}
          <main className="dd-content">
            {/* Tabs */}
            <div className="dd-tabs">
              {TABS.map(t => {
                const Icon = TAB_ICONS[t];
                return (
                  <button key={t} className={`dd-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
                    <Icon size={14} style={{ marginRight: 6, verticalAlign: -2 }} />
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </button>
                );
              })}
            </div>

            {/* Tab Content */}
            <AnimatePresence mode="wait">
              <motion.div key={tab} {...fadeVariant}>

                {/* ═══ OVERVIEW TAB ═══════════════════ */}
                {tab === 'overview' && (
                  <div className="dd-stagger" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    {/* Summary + Urgency */}
                    <div className="dd-card-grid">
                      <div className="dd-card dd-fade-up">
                        <div className="dd-card-title"><Activity size={14} /> Disease Summary</div>
                        <div className="dd-metric"><span className="dd-metric-label">Crop</span><span className="dd-metric-value" style={{ color: 'var(--dd-accent)' }}>{cropName}</span></div>
                        <div className="dd-metric"><span className="dd-metric-label">Disease</span><span className="dd-metric-value">{diseaseName}</span></div>
                        <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
                          <div className="dd-metric"><span className="dd-metric-label">Confidence</span><span className="dd-metric-value">{result.confidence}%</span></div>
                          <div className="dd-metric"><span className="dd-metric-label">Severity</span><span className="dd-metric-value" style={{ color: sev.color }}>{sev.label}</span></div>
                        </div>
                      </div>
                      <div className="dd-card dd-fade-up">
                        <div className="dd-card-title"><AlertTriangle size={14} /> Urgency Status</div>
                        <div className="dd-md">
                          {(extractSection(screens.first, 'Action Required') || extractSection(screens.first, 'Action')) ? (
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {extractSection(screens.first, 'Action Required') || extractSection(screens.first, 'Action')}
                            </ReactMarkdown>
                          ) : (
                            <p style={{ margin: 0, color: '#e2e8f0', lineHeight: 1.6 }}>
                              <strong>{result.severity || 'Moderate'} Urgency:</strong> Immediate treatment and sanitation recommended within {result.severity === 'High' ? '24 hours' : '2-3 days'} to halt spore dispersal and prevent canopy loss.
                            </p>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* ── Weather Context Banner (Synced with Weather Advisory) ── */}
                    {result.status === 'diseased' && (
                      <div className="dd-weather-banner" style={{ border: '1px solid rgba(16,185,129,0.35)', background: 'rgba(16,185,129,0.08)' }}>
                        <div className="dd-weather-banner-icon" style={{ background: 'rgba(16,185,129,0.2)', color: '#34d399' }}>⚡</div>
                        <div style={{ flex: 1 }}>
                          <strong style={{ color: '#34d399', fontSize: 14 }}>Diagnosis Linked to Farm Weather Intelligence</strong>
                          <p style={{ margin: '4px 0 0', opacity: 0.85, fontSize: 13, color: '#e2e8f0' }}>
                            Your diagnosed <b>{cropName} ({diseaseName})</b> has been synced with your local weather advisory. Check your Weather Portal for rain-washout warnings, humidity infection thresholds, and optimal spray window timings.
                          </p>
                        </div>
                        <button
                          className="dd-weather-banner-btn"
                          onClick={() => nav('/weather')}
                          type="button"
                          style={{ background: '#10b981', color: '#fff', fontWeight: 700, padding: '8px 16px', borderRadius: 10, cursor: 'pointer', border: 'none' }}
                        >
                          View Weather Action Plan →
                        </button>
                      </div>
                    )}

                    {/* Medicines */}
                    {meds.length > 0 ? (
                      <div className="dd-card dd-fade-up">
                        <div className="dd-card-title"><Beaker size={14} /> Recommended Medicines</div>
                        <div className="dd-card-grid-3">
                          {meds.slice(0, 3).map((m, i) => (
                            <div className="dd-med-card" key={i}>
                              <div className="dd-med-rank">{m.rank} Choice</div>
                              <div className="dd-med-name">{m.name}</div>
                              {m.details.map((d, j) => <div className="dd-med-detail" key={j}>{d}</div>)}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : result.treatment ? (
                      <div className="dd-card dd-fade-up">
                        <div className="dd-card-title"><Beaker size={14} /> Recommended Medicines</div>
                        <div className="dd-med-card" style={{ maxWidth: '100%' }}>
                          <div className="dd-med-rank">1st Choice</div>
                          <div className="dd-med-name">{result.treatment.split('.')[0]}</div>
                          <div className="dd-med-detail">{result.treatment}</div>
                        </div>
                      </div>
                    ) : null}

                    {/* Action Plan */}
                    <div className="dd-card dd-fade-up">
                      <div className="dd-card-title"><Clock size={14} /> Today's Action Plan</div>
                      <ul className="dd-checklist">
                        {(actions.length > 0 ? actions : [
                          'Prune and destroy heavily infected leaves immediately to stop spore spread.',
                          `Apply targeted treatment: ${result.treatment || 'prescribed fungicide / bactericide'}.`,
                          'Water only at the base of plants; avoid wetting foliage.',
                          'Check neighboring rows for early signs of infection.',
                        ]).map((a, i) => (
                          <li className="dd-check-item" key={i}>
                            <div className="dd-check-icon" style={{ background: 'rgba(82,183,136,0.15)', color: 'var(--dd-accent)' }}>{i + 1}</div>
                            {a}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Economic Impact */}
                    {(extractSection(screens.second, 'Economic') || extractSection(screens.first, 'Economic')) && (
                      <div className="dd-card dd-fade-up">
                        <div className="dd-card-title"><TrendingUp size={14} /> Economic Impact</div>
                        <div className="dd-md">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{extractSection(screens.second, 'Economic') || extractSection(screens.first, 'Economic')}</ReactMarkdown>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* ═══ TREATMENT TAB ══════════════════ */}
                {tab === 'treatment' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    {/* Products Grid */}
                    {meds.length > 0 ? (
                      <div className="dd-card dd-fade-up">
                        <div className="dd-card-title"><Beaker size={14} /> Medicine Details</div>
                        <div className="dd-card-grid-3">
                          {meds.map((m, i) => (
                            <div className="dd-med-card" key={i}>
                              <div className="dd-med-rank">{m.rank} Choice</div>
                              <div className="dd-med-name">{m.name}</div>
                              {m.details.map((d, j) => <div className="dd-med-detail" key={j}>{d}</div>)}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : result.treatment ? (
                      <div className="dd-card dd-fade-up">
                        <div className="dd-card-title"><Beaker size={14} /> Medicine Details</div>
                        <div className="dd-med-card" style={{ maxWidth: '100%' }}>
                          <div className="dd-med-rank">1st Choice</div>
                          <div className="dd-med-name">{result.treatment.split('.')[0]}</div>
                          <div className="dd-med-detail">{result.treatment}</div>
                        </div>
                      </div>
                    ) : null}

                    {/* Treatment Timeline */}
                    <div className="dd-card dd-fade-up">
                      <div className="dd-card-title"><Clock size={14} /> Treatment Timeline</div>
                      <div className="dd-timeline">
                        {(actions.length > 0 ? actions : [
                          'Day 1: Rogue out heavily diseased leaves and calibrate spray equipment.',
                          'Day 2: Spray recommended fungicide during dry morning window.',
                          'Day 7: Re-evaluate canopy for new active lesions or pustules.',
                        ]).map((a, i) => (
                          <div className="dd-timeline-item" key={i}>
                            <div className="dd-timeline-dot" />
                            <div className="dd-timeline-step">Step {i + 1}</div>
                            <div className="dd-timeline-text">{a}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Application Instructions */}
                    <div className="dd-card dd-fade-up">
                      <div className="dd-card-title"><Shield size={14} /> Application Instructions</div>
                      <div className="dd-md">
                        {extractSection(screens.third, 'Treatment Procedure') || extractSection(screens.third, 'Procedure') ? (
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {extractSection(screens.third, 'Treatment Procedure') || extractSection(screens.third, 'Procedure')}
                          </ReactMarkdown>
                        ) : (
                          <ul style={{ margin: 0, paddingLeft: 20, color: '#cbd5e1', lineHeight: 1.7 }}>
                            <li>Wear protective gloves and mask when preparing chemical mixtures.</li>
                            <li>Ensure complete spray coverage of both upper and lower leaf surfaces until light runoff.</li>
                            <li>Do not apply immediately before forecast rain to avoid chemical wash-off.</li>
                          </ul>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* ═══ PREVENTION TAB ═════════════════ */}
                {tab === 'prevention' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    <div className="dd-card-grid">
                      {/* Causes */}
                      <div className="dd-card dd-fade-up">
                        <div className="dd-card-title"><AlertTriangle size={14} /> Why It Happened</div>
                        <ul className="dd-checklist">
                          {(causes.length > 0 ? causes : [
                            'Elevated canopy humidity and leaf wetness favor spore germination.',
                            'Water splashing from soil or overhead irrigation spreads pathogens.',
                            'Dense planting reduces internal airflow, trapping moisture on foliage.',
                          ]).map((c, i) => (
                            <li className="dd-check-item" key={i}>
                              <div className="dd-check-icon" style={{ background: 'rgba(239,68,68,0.12)', color: 'var(--dd-danger)', fontSize: 10 }}>!</div>
                              {c}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Prevention Checklist */}
                      <div className="dd-card dd-fade-up">
                        <div className="dd-card-title"><Shield size={14} /> Prevention Checklist</div>
                        <ul className="dd-checklist">
                          {(prevention.length > 0 ? prevention : (result.prevention ? result.prevention.split('.').filter(s => s.trim()).map(s => s.trim()) : [
                            'Space crops properly to encourage maximum air circulation.',
                            'Irrigate at plant base using drip systems; never wet foliage overhead.',
                            'Rotate crops with non-susceptible families every season.',
                          ])).map((p, i) => (
                            <li className="dd-check-item" key={i}>
                              <div className="dd-check-icon" style={{ background: 'rgba(34,197,94,0.12)', color: 'var(--dd-success)' }}>✓</div>
                              {p}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    {/* Expected Recovery Timeline Card */}
                    <div className="dd-card dd-fade-up">
                      <div className="dd-card-title"><Clock size={14} /> Expected Recovery Timeline</div>
                      <div className="dd-timeline">
                        {(recovery.length > 0 ? recovery : [
                          { step: 'Sanitation & Pruning', time: 'Day 1', outcome: 'Remove primary inoculum and improve airflow.' },
                          { step: 'Protective Treatment', time: 'Day 2 – 3', outcome: 'Halt active spore growth on remaining foliage.' },
                          { step: 'Canopy Recovery', time: 'Day 7 – 14', outcome: 'Clean new growth emerges without disease symptoms.' }
                        ]).map((r, i) => (
                          <div className="dd-timeline-item" key={i}>
                            <div className="dd-timeline-dot" style={{ borderColor: 'var(--dd-accent)' }} />
                            <div className="dd-timeline-step">{r.step} <span style={{ color: 'var(--dd-text-muted)', fontWeight: 400 }}>({r.time})</span></div>
                            <div className="dd-timeline-text">{r.outcome}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* ═══ FAQ TAB ════════════════════════ */}
                {tab === 'faq' && (
                  <div className="dd-card dd-fade-up">
                    <div className="dd-card-title"><HelpCircle size={14} /> Frequently Asked Questions</div>
                    {faqItems.length > 0 ? (
                      <div className="dd-accordion">
                        {faqItems.map((item, i) => (
                          <div key={i}>
                            <div className={`dd-acc-header ${faqOpen === i ? 'open' : ''}`} onClick={() => setFaqOpen(faqOpen === i ? null : i)}>
                              {item.q}
                              <ChevronDown size={14} className={`dd-acc-chevron ${faqOpen === i ? 'open' : ''}`} />
                            </div>
                            {faqOpen === i && (
                              <div className="dd-acc-body dd-md">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.a}</ReactMarkdown>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="dd-md">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{extractSection(screens.second, 'Frequently Asked')}</ReactMarkdown>
                      </div>
                    )}
                  </div>
                )}

                {/* ═══ ADVANCED TAB ═══════════════════ */}
                {tab === 'advanced' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    <div className="dd-subtabs">
                      {ADV_TABS.map(t => (
                        <button key={t} className={`dd-subtab ${advTab === t ? 'active' : ''}`} onClick={() => setAdvTab(t)}>
                          {t.charAt(0).toUpperCase() + t.slice(1)}
                        </button>
                      ))}
                    </div>

                    <AnimatePresence mode="wait">
                      <motion.div key={advTab} {...fadeVariant}>

                        {/* AI Report */}
                        {advTab === 'report' && (
                          <div className="dd-card dd-fade-up">
                            <div className="dd-card-title"><Layers size={14} /> AI Diagnostic Report</div>
                            <div className="dd-md">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{extractSection(screens.third, 'Detailed AI Report') || extractSection(screens.third, 'AI Report')}</ReactMarkdown>
                            </div>
                          </div>
                        )}

                        {/* Scientific */}
                        {advTab === 'scientific' && (
                          <div className="dd-card dd-fade-up">
                            <div className="dd-card-title"><Beaker size={14} /> Scientific Information</div>
                            <div className="dd-md">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{extractSection(screens.third, 'Scientific')}</ReactMarkdown>
                            </div>
                          </div>
                        )}

                        {/* Recovery Timeline */}
                        {advTab === 'recovery' && (
                          <div className="dd-card dd-fade-up">
                            <div className="dd-card-title"><Clock size={14} /> Expected Recovery Timeline</div>
                            {recovery.length > 0 ? (
                              <div className="dd-timeline" style={{ marginTop: 12 }}>
                                {recovery.map((r, i) => (
                                  <div className="dd-timeline-item" key={i}>
                                    <div className="dd-timeline-dot" style={{ borderColor: 'var(--dd-accent)' }} />
                                    <div className="dd-timeline-step">{r.step} <span style={{ color: 'var(--dd-text-muted)', fontWeight: 400 }}>({r.time})</span></div>
                                    <div className="dd-timeline-text">{r.outcome}</div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="dd-md">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{extractSection(screens.third, 'Expected Recovery Timeline')}</ReactMarkdown>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Analysis */}
                        {advTab === 'analysis' && (
                          <div className="dd-card dd-fade-up">
                            <div className="dd-card-title"><Activity size={14} /> Advanced Analysis</div>
                            <div className="dd-md">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{extractSection(screens.third, 'Advanced Analysis') || extractSection(screens.third, 'Analysis')}</ReactMarkdown>
                            </div>
                          </div>
                        )}

                      </motion.div>
                    </AnimatePresence>
                  </div>
                )}

              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      )}

      {/* ── Floating AI Button ──────────────────── */}
      {result && (
        <button className="dd-fab" onClick={() => setDrawer(true)} title="Ask AI">
          <MessageCircle size={22} />
        </button>
      )}

      {/* ── AI Assistant Drawer ─────────────────── */}
      {drawer && (
        <>
          <div className="dd-drawer-overlay dd-fade-in" onClick={() => setDrawer(false)} />
          <div className="dd-drawer dd-slide-in">
            <div className="dd-drawer-title">
              AI Assistant
              <button className="dd-drawer-close" onClick={() => setDrawer(false)}><X size={18} /></button>
            </div>
            <p style={{ color: 'var(--dd-text-dim)', fontSize: 13, marginBottom: 20 }}>
              Ask any question about <strong>{diseaseName}</strong> on <strong>{cropName}</strong>.
            </p>
            <div className="dd-drawer-suggestions">
              {['Why did this disease happen?', 'Which medicine should I use?', 'Can it spread to other plants?', 'Will my yield be affected?', 'What should I do today?'].map((q, i) => (
                <button className="dd-drawer-sug" key={i} onClick={() => { toast(`💬 ${q}\n\nAI Chat feature coming soon!`, { duration: 3000 }); }}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}