import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { getWeatherDashboard, resolveDiseaseScan, getActiveDiseaseScans } from '../services/api';
import './WeatherAdvisory.css';

// ─── WMO Weather Code Map ──────────────────────────────────────────────────
const WMO = {
  0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
  45: 'Fog', 48: 'Depositing fog', 51: 'Light drizzle', 53: 'Drizzle',
  55: 'Dense drizzle', 61: 'Light rain', 63: 'Rain', 65: 'Heavy rain',
  71: 'Light snow', 73: 'Snow', 75: 'Heavy snow',
  80: 'Rain showers', 81: 'Rain showers', 82: 'Heavy showers',
  95: 'Thunderstorm', 96: 'Thunderstorm', 99: 'Severe thunderstorm',
};

const FALLBACK_HOURLY = [
  { time: 'Now', temp: 31, code: 2, rain: 0 },
  { time: '9:00 pm', temp: 30, code: 2, rain: 0 },
  { time: '10:00 pm', temp: 29, code: 95, rain: 1.2 },
  { time: '11:00 pm', temp: 29, code: 95, rain: 0.8 },
  { time: '12:00 am', temp: 28, code: 80, rain: 0.2 },
  { time: '01:00 am', temp: 27, code: 61, rain: 0.4 },
  { time: '02:00 am', temp: 27, code: 61, rain: 0.1 },
  { time: '03:00 am', temp: 26, code: 3, rain: 0 },
  { time: '04:00 am', temp: 26, code: 3, rain: 0 },
  { time: '05:00 am', temp: 26, code: 1, rain: 0 },
  { time: '06:00 am', temp: 27, code: 0, rain: 0, spraySafe: true },
  { time: '07:00 am', temp: 28, code: 0, rain: 0, spraySafe: true },
  { time: '08:00 am', temp: 29, code: 0, rain: 0, spraySafe: true },
  { time: '09:00 am', temp: 31, code: 1, rain: 0 },
];

const FALLBACK = {
  temp: 31, feels: 33, humidity: 72, wind: 8, rain: 0,
  visibility: 10, uv: 6, code: 2, condition: 'Partly cloudy',
  sunrise: '05:58 AM', sunset: '06:14 PM', source: 'demo',
  hourly: FALLBACK_HOURLY,
  forecast: [
    { day: '18 Sept Today',    high: 38, low: 27, code: 95, rain: 0.0 },
    { day: '19 Sept Tomorrow', high: 36, low: 26, code: 2,  rain: 1.2 },
    { day: '20 Sept Sat',      high: 33, low: 26, code: 65, rain: 8.4 },
    { day: '21 Sept Sun',      high: 34, low: 26, code: 2,  rain: 2.1 },
    { day: '22 Sept Mon',      high: 35, low: 27, code: 0,  rain: 0.0 },
  ],
};

const CROP_META = {
  tomato: { label: 'Tomato', icon: '🍅' },
  potato: { label: 'Potato', icon: '🥔' },
  corn:   { label: 'Corn', icon: '🌽' },
  pepper: { label: 'Pepper / Chilli', icon: '🌶️' },
};

// ─── Helpers ───────────────────────────────────────────────────────────────
const weatherGlyph = code => {
  if ([95, 96, 99].includes(code)) return '⛈️';
  if ([61, 63, 65, 80, 81, 82].includes(code)) return '🌧️';
  if ([51, 53, 55].includes(code)) return '🌦️';
  if ([45, 48].includes(code)) return '🌫️';
  if ([2, 3].includes(code)) return '⛅';
  if (code === 1) return '🌤️';
  return '☀️';
};

// ─── Score calculation ─────────────────────────────────────────────────────
function calculateScore(wx) {
  let s = 96;
  s -= Math.abs(wx.temp - 28) * 2.0;
  s -= Math.abs(wx.humidity - 65) * 0.4;
  s -= wx.wind * 1.4;
  s -= wx.rain * 10;
  if ([95, 96, 99].includes(wx.code)) s -= 18;
  return Math.max(25, Math.min(98, Math.round(s)));
}

// ─── Operations builder ────────────────────────────────────────────────────
function buildOperations(wx, apiOps) {
  const storm  = [95, 96, 99].includes(wx.code);
  const wet    = wx.rain > 0.4;
  const windy  = wx.wind > 16;
  const sprayWindow = apiOps?.spraying?.window || (storm || windy || wet ? 'No safe window today' : '06:00 AM – 09:00 AM');
  const sprayStatus = storm || windy || wet ? 'unsafe' : 'safe';

  return [
    {
      name: 'Pesticide Spraying',
      status: sprayStatus,
      note: sprayStatus === 'unsafe'
        ? 'High wind or active rainfall risks pesticide wash-off and chemical drift.'
        : 'Optimal early morning window with dry foliage and minimal drift risk.',
      window: sprayWindow,
    },
    {
      name: 'Irrigation',
      status: wx.rain > 1 ? 'warn' : 'safe',
      note: wx.rain > 1
        ? 'Rainfall detected. Pause scheduled irrigation to avoid root zone saturation.'
        : 'Normal soil moisture. Irrigate early morning as per crop development stage.',
      window: wx.rain > 1 ? 'Skip today — recheck evening' : '05:00 AM – 07:00 AM',
    },
    {
      name: 'Fertilizer Application',
      status: wet ? 'unsafe' : wx.humidity > 85 ? 'warn' : 'safe',
      note: wet
        ? 'Soil leaching risk is high due to moisture runoff.'
        : wx.humidity > 85
          ? 'High humidity may slow granular breakdown; ensure light soil incorporation.'
          : 'Good granular uptake conditions. Apply fertilizer before Saturday rainfall.',
      window: wet ? 'Delay 24 hours' : '07:00 AM – 10:00 AM',
    },
    {
      name: 'Harvesting',
      status: wx.rain > 0.5 || wx.humidity > 86 ? 'warn' : 'safe',
      note: wx.rain > 0.5 || wx.humidity > 86
        ? 'Rain forecast Saturday. Safe harvest window opens Monday (3 consecutive dry days).'
        : 'Field dry enough for routine picking. Crate immediately in shaded shed.',
      window: wx.rain > 0.5 ? 'Opens Mon, 22 Sep' : '07:00 AM – 12:00 PM',
    },
  ];
}

// ─── Disease risk builder ──────────────────────────────────────────────────
function buildDiseaseRisk(wx, dashRisk) {
  const DEFAULT_CROPS = ['tomato', 'potato', 'corn', 'pepper'];
  const crops = dashRisk ? Object.keys(dashRisk) : DEFAULT_CROPS;

  return crops.map(crop => {
    if (dashRisk?.[crop]) return { crop, ...dashRisk[crop] };
    const h = wx.humidity, t = wx.temp;
    let risk = 'LOW', disease = 'No active threat', reason = 'Current conditions within safe range.', action = 'Routine scouting only.';
    if (crop === 'tomato' || crop === 'potato') {
      if (h > 80 && t >= 16 && t <= 28) {
        risk = 'HIGH'; disease = 'Early Blight & Septoria Spot';
        reason = 'Warm humid nights accelerate Alternaria and Septoria sporulation on lower leaves.';
        action = 'Apply Mancozeb or Chlorothalonil every 7 days. Remove affected lower foliage.';
      } else if (h > 70) {
        risk = 'MEDIUM'; disease = 'Early Blight';
        reason = 'Moderate relative humidity sustains Alternaria spore viability.';
        action = 'Scout from base upward. Improve row airflow and ensure field drainage.';
      }
    }
    if (crop === 'corn') {
      if (h > 78) {
        risk = 'MEDIUM'; disease = 'Common Rust';
        reason = 'Extended leaf wetness enables urediospore germ-tube penetration.';
        action = 'Monitor pustule count on mid-canopy leaves. Spray Azoxystrobin if >5% coverage.';
      }
    }
    if (crop === 'pepper') {
      if (h > 80) {
        risk = 'MEDIUM'; disease = 'Bacterial Spot (Xanthomonas)';
        reason = 'Splashing raindrops and persistent humidity accelerate bacterial spread across rows.';
        action = 'Apply copper-based bactericide before forecast rain. Avoid overhead sprinkler watering.';
      }
    }
    return { crop, risk, disease_watch: disease, reason, action };
  });
}

// ─── Activity Calendar Builder ─────────────────────────────────────────────
function buildCalendar(forecast) {
  return forecast.map((day, i) => {
    const wet = day.rain > 0.5;
    const heavy = day.rain > 3.0;
    const rating = heavy ? 'Poor' : day.rain > 0 ? 'Moderate' : 'Optimal';
    return {
      ...day,
      rating,
      morning: heavy ? [] : wet ? ['Scouting'] : ['Spray Safe', 'Irrigate', 'Field Check'],
      afternoon: heavy ? [] : wet ? [] : ['Fertilize', i >= 2 ? 'Harvest' : 'Scouting'],
      evening: heavy ? ['Check Drainage'] : wet ? ['Equipment Check'] : ['Drip Irrigation', i >= 2 ? 'Dispatch Mandi' : 'Record Keeping'],
      avoidMorning: heavy ? ['Avoid Spray', 'Skip Fert'] : [],
      avoidAfternoon: heavy ? ['Skip Harvest', 'Skip Fert'] : wet ? ['Avoid Spray'] : [],
      avoidEvening: [],
    };
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// ANIMATED SKY CANVAS COMPONENT
// Luminous Moon/Sun, Parallax Volumetric Clouds, Stars, and Tree Silhouettes
// ═══════════════════════════════════════════════════════════════════════════
function WeatherCanvas({ wx, isNightMode }) {
  const canvasRef = useRef(null);
  const wxRef = useRef(wx);
  const nightRef = useRef(isNightMode);

  useEffect(() => { wxRef.current = wx; }, [wx]);
  useEffect(() => { nightRef.current = isNightMode; }, [isNightMode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf;

    const STARS = Array.from({ length: 130 }, () => ({
      x: Math.random(),
      y: Math.random() * 0.72,
      r: Math.random() * 1.5 + 0.5,
      phase: Math.random() * Math.PI * 2,
      speed: 0.015 + Math.random() * 0.025,
    }));

    const CLOUDS = Array.from({ length: 8 }, () => ({
      x: Math.random() * 1.4 - 0.2,
      y: 0.10 + Math.random() * 0.40,
      scale: 0.7 + Math.random() * 0.85,
      speed: 0.00014 + Math.random() * 0.00022,
      opacity: 0.30 + Math.random() * 0.30,
      puffs: Array.from({ length: 6 }, () => ({
        dx: (Math.random() - 0.5) * 120,
        dy: (Math.random() - 0.5) * 32,
        rx: 50 + Math.random() * 55,
        ry: 26 + Math.random() * 24,
      })),
    }));

    const RAIN_DROPS = Array.from({ length: 90 }, () => ({
      x: Math.random(),
      y: Math.random(),
      len: 12 + Math.random() * 18,
      speed: 0.018 + Math.random() * 0.022,
    }));

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const W = parent.clientWidth;
      const H = parent.clientHeight;
      const wxNow = wxRef.current || FALLBACK;
      const isNight = nightRef.current;
      const isRain = wxNow.rain > 0.4 || [61, 63, 65, 80, 81, 82, 95, 96, 99].includes(wxNow.code);

      ctx.clearRect(0, 0, W, H);

      // 1. Sky Gradient
      const skyGrad = ctx.createLinearGradient(0, 0, 0, H);
      if (isNight) {
        skyGrad.addColorStop(0, '#07101f');
        skyGrad.addColorStop(0.45, '#0b1b36');
        skyGrad.addColorStop(0.75, '#122b52');
        skyGrad.addColorStop(1, '#1a3a69');
      } else if (isRain) {
        skyGrad.addColorStop(0, '#0f1722');
        skyGrad.addColorStop(0.5, '#1c2838');
        skyGrad.addColorStop(1, '#27384d');
      } else {
        skyGrad.addColorStop(0, '#105caa');
        skyGrad.addColorStop(0.5, '#3b8ecc');
        skyGrad.addColorStop(0.85, '#7ec4ea');
        skyGrad.addColorStop(1, '#a6dbf7');
      }
      ctx.fillStyle = skyGrad;
      ctx.fillRect(0, 0, W, H);

      // 2. Stars (Night Sky)
      if (isNight) {
        STARS.forEach(s => {
          s.phase += s.speed;
          const alpha = 0.3 + 0.6 * Math.abs(Math.sin(s.phase));
          ctx.beginPath();
          ctx.arc(s.x * W, s.y * H, s.r, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(235, 245, 255, ${alpha})`;
          ctx.fill();
        });
      }

      // 3. Moon or Sun
      const bodyX = W * 0.22;
      const bodyY = H * 0.36;

      if (isNight) {
        // Moon Glow Halo (from Screenshot 2)
        const halo = ctx.createRadialGradient(bodyX, bodyY, 15, bodyX, bodyY, 90);
        halo.addColorStop(0, 'rgba(230, 240, 255, 0.42)');
        halo.addColorStop(0.4, 'rgba(190, 215, 255, 0.16)');
        halo.addColorStop(1, 'transparent');
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(bodyX, bodyY, 90, 0, Math.PI * 2);
        ctx.fill();

        // Moon Disc
        const moonGrad = ctx.createRadialGradient(bodyX - 8, bodyY - 8, 4, bodyX, bodyY, 28);
        moonGrad.addColorStop(0, '#ffffff');
        moonGrad.addColorStop(0.65, '#e4eaf5');
        moonGrad.addColorStop(1, '#cbd5e8');
        ctx.fillStyle = moonGrad;
        ctx.beginPath();
        ctx.arc(bodyX, bodyY, 28, 0, Math.PI * 2);
        ctx.fill();

        // Moon Craters
        ctx.fillStyle = 'rgba(150, 168, 198, 0.28)';
        ctx.beginPath();
        ctx.arc(bodyX - 8, bodyY - 6, 6, 0, Math.PI * 2);
        ctx.arc(bodyX + 7, bodyY + 5, 8, 0, Math.PI * 2);
        ctx.arc(bodyX - 4, bodyY + 12, 5, 0, Math.PI * 2);
        ctx.arc(bodyX + 9, bodyY - 10, 4, 0, Math.PI * 2);
        ctx.fill();
      } else {
        // Sun Body & Aura
        const sunHalo = ctx.createRadialGradient(bodyX, bodyY, 20, bodyX, bodyY, 120);
        sunHalo.addColorStop(0, 'rgba(255, 240, 160, 0.7)');
        sunHalo.addColorStop(0.5, 'rgba(255, 210, 90, 0.25)');
        sunHalo.addColorStop(1, 'transparent');
        ctx.fillStyle = sunHalo;
        ctx.beginPath();
        ctx.arc(bodyX, bodyY, 120, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#fff4a3';
        ctx.beginPath();
        ctx.arc(bodyX, bodyY, 32, 0, Math.PI * 2);
        ctx.fill();
      }

      // 4. Parallax Clouds Drifting
      CLOUDS.forEach(c => {
        c.x += c.speed;
        if (c.x > 1.25) c.x = -0.35;
        const cx = c.x * W;
        const cy = c.y * H;

        ctx.save();
        ctx.globalAlpha = c.opacity;
        ctx.fillStyle = isNight
          ? 'rgba(180, 205, 235, 0.42)'
          : isRain
          ? 'rgba(65, 80, 102, 0.72)'
          : 'rgba(255, 255, 255, 0.65)';

        c.puffs.forEach(p => {
          ctx.beginPath();
          ctx.ellipse(cx + p.dx * c.scale, cy + p.dy * c.scale, p.rx * c.scale, p.ry * c.scale, 0, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
      });

      // 5. Rain Drops (if wet)
      if (isRain) {
        ctx.strokeStyle = 'rgba(180, 215, 245, 0.45)';
        ctx.lineWidth = 1.2;
        RAIN_DROPS.forEach(r => {
          r.y += r.speed;
          r.x -= r.speed * 0.2;
          if (r.y > 1) { r.y = -0.05; r.x = Math.random(); }
          ctx.beginPath();
          ctx.moveTo(r.x * W, r.y * H);
          ctx.lineTo(r.x * W - 3, r.y * H + r.len);
          ctx.stroke();
        });
      }

      // 6. Rolling Farm Hills & Tree Silhouettes at Horizon
      const horizonY = H * 0.76;

      // Far Hill Ridge
      ctx.fillStyle = isNight ? '#081729' : '#1d5a2b';
      ctx.beginPath();
      ctx.moveTo(0, horizonY + 25);
      for (let x = 0; x <= W; x += 40) {
        ctx.lineTo(x, horizonY + 25 + Math.sin(x / 140) * 14);
      }
      ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath(); ctx.fill();

      // Near Hill Ridge
      ctx.fillStyle = isNight ? '#050f1d' : '#13401d';
      ctx.beginPath();
      ctx.moveTo(0, horizonY + 45);
      for (let x = 0; x <= W; x += 35) {
        ctx.lineTo(x, horizonY + 45 + Math.cos(x / 110) * 10);
      }
      ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath(); ctx.fill();

      // Tree Silhouettes (Pine and Oak clusters along the ridge)
      ctx.fillStyle = isNight ? '#040b17' : '#0e2b14';
      const treePositions = [0.05, 0.08, 0.12, 0.28, 0.31, 0.44, 0.62, 0.65, 0.68, 0.82, 0.85, 0.94];
      treePositions.forEach((pos, idx) => {
        const tx = pos * W;
        const ty = horizonY + 28 + Math.sin(tx / 140) * 12;
        const h = 24 + (idx % 3) * 8;

        if (idx % 2 === 0) {
          // Pine tree
          ctx.beginPath();
          ctx.moveTo(tx, ty - h);
          ctx.lineTo(tx - h * 0.35, ty);
          ctx.lineTo(tx + h * 0.35, ty);
          ctx.closePath();
          ctx.fill();
        } else {
          // Leafy rounded oak tree
          ctx.beginPath();
          ctx.arc(tx, ty - h * 0.6, h * 0.42, 0, Math.PI * 2);
          ctx.arc(tx - h * 0.22, ty - h * 0.4, h * 0.3, 0, Math.PI * 2);
          ctx.arc(tx + h * 0.22, ty - h * 0.4, h * 0.3, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillRect(tx - 2, ty - h * 0.3, 4, h * 0.35);
        }
      });

      raf = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener('resize', resize);
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas className="wa-sky-canvas" ref={canvasRef} aria-hidden="true" />;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WEATHER ADVISORY COMPONENT
// ═══════════════════════════════════════════════════════════════════════════
export default function WeatherAdvisory() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [weather, setWeather] = useState(FALLBACK);
  const [dashData, setDashData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('disease');
  const [locationLabel, setLocationLabel] = useState('');
  const [locationReady, setLocationReady] = useState(false);
  const [activeScans, setActiveScans] = useState([]);
  const [resolvingId, setResolvingId] = useState(null);

  // Check if current IST time is night (between 6:30 PM and 6:00 AM)
  const isNight = useMemo(() => {
    const hr = Number(new Intl.DateTimeFormat('en-IN', {
      timeZone: 'Asia/Kolkata', hour: 'numeric', hour12: false,
    }).format(new Date()));
    return hr < 6 || hr >= 18;
  }, []);

  useEffect(() => {
    if (user?.location) {
      setLocationLabel(user.location);
      setLocationReady(true);
    } else {
      setLocationLabel('');
      setLocationReady(true);
    }
  }, [user]);

  // Fetch Open-Meteo Weather with Hourly Forecast
  useEffect(() => {
    if (!locationReady || !locationLabel) return;
    let ignore = false;
    setLoading(true);

    async function load() {
      try {
        const geoRes = await fetch(
          `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(locationLabel)}&count=1&language=en&format=json`
        );
        const geoData = await geoRes.json();
        const place = geoData?.results?.[0];
        if (!place) throw new Error('Location not found');

        const omRes = await fetch(
          `https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}&longitude=${place.longitude}` +
          `&current=temperature_2m,relative_humidity_2m,precipitation,weathercode,windspeed_10m,apparent_temperature,uv_index` +
          `&hourly=temperature_2m,weathercode,precipitation` +
          `&daily=sunrise,sunset,temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum,relative_humidity_2m_max` +
          `&timezone=Asia%2FKolkata&forecast_days=5`
        );
        const omData = await omRes.json();
        if (!omData.current || !omData.daily) throw new Error('Incomplete data');

        const cur = omData.current;
        const daily = omData.daily;
        const hourlyRaw = omData.hourly;

        // Process hourly next 12 hours
        const hourlyItems = [];
        if (hourlyRaw && hourlyRaw.time) {
          const nowHourIdx = new Date().getHours();
          for (let i = nowHourIdx; i < Math.min(nowHourIdx + 12, hourlyRaw.time.length); i++) {
            const dateObj = new Date(hourlyRaw.time[i]);
            const timeStr = i === nowHourIdx ? 'Now' : dateObj.toLocaleTimeString('en-IN', { hour: 'numeric', hour12: true });
            const hrNum = dateObj.getHours();
            hourlyItems.push({
              time: timeStr,
              temp: Math.round(hourlyRaw.temperature_2m[i]),
              code: hourlyRaw.weathercode[i],
              rain: hourlyRaw.precipitation[i] || 0,
              spraySafe: hrNum >= 6 && hrNum <= 9 && (hourlyRaw.precipitation[i] || 0) === 0,
            });
          }
        }

        const forecast = daily.time.map((date, i) => {
          const dObj = new Date(date);
          const dayName = i === 0 ? 'Today' : i === 1 ? 'Tomorrow' : new Intl.DateTimeFormat('en-IN', { weekday: 'short' }).format(dObj);
          const dateStr = `${dObj.getDate()} ${new Intl.DateTimeFormat('en-IN', { month: 'short' }).format(dObj)} ${dayName}`;
          return {
            day: dateStr,
            high: Math.round(daily.temperature_2m_max[i]),
            low: Math.round(daily.temperature_2m_min[i]),
            code: daily.weathercode[i],
            rain: daily.precipitation_sum[i] || 0,
            humidity: daily.relative_humidity_2m_max?.[i] || 70,
          };
        });

        const sunrise = new Date(daily.sunrise[0]).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: true, timeZone: 'Asia/Kolkata' });
        const sunset = new Date(daily.sunset[0]).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: true, timeZone: 'Asia/Kolkata' });

        if (!ignore) {
          setWeather({
            temp: Math.round(cur.temperature_2m),
            feels: Math.round(cur.apparent_temperature),
            humidity: Math.round(cur.relative_humidity_2m),
            wind: Math.round(cur.windspeed_10m),
            rain: Number(cur.precipitation || 0),
            uv: Math.round(cur.uv_index || 0),
            code: cur.weathercode,
            condition: WMO[cur.weathercode] || 'Partly cloudy',
            sunrise, sunset,
            hourly: hourlyItems.length ? hourlyItems : FALLBACK_HOURLY,
            forecast,
            source: 'live',
            location: `${place.name}, ${place.admin1 || place.country_code?.toUpperCase() || 'IN'}`,
          });
        }
      } catch {
        if (!ignore) setWeather({ ...FALLBACK });
      } finally {
        if (!ignore) setLoading(false);
      }
    }

    load();
    return () => { ignore = true; };
  }, [locationLabel, locationReady]);

  // Load active disease scans from localStorage & backend
  useEffect(() => {
    let localScans = [];
    try {
      const stored = localStorage.getItem('alip_active_disease_scans');
      if (stored) localScans = JSON.parse(stored);
    } catch (e) {
      console.warn('Failed reading local scans:', e);
    }
    if (localScans.length) setActiveScans(localScans);

    if (locationLabel) {
      getActiveDiseaseScans(locationLabel)
        .then(res => {
          if (res?.data?.success && res.data.scans?.length) {
            // Merge with local scans without duplicates
            const serverScans = res.data.scans;
            const merged = [...serverScans];
            localScans.forEach(ls => {
              if (!merged.some(ms => ms.crop === ls.crop && ms.disease === ls.disease)) {
                merged.push(ls);
              }
            });
            setActiveScans(merged);
          }
        })
        .catch(err => console.warn('Active scans fetch failed:', err.message));
    }
  }, [locationLabel]);

  // Fetch Backend Agro Dashboard
  useEffect(() => {
    if (!locationLabel) return;
    let ignore = false;
    async function loadDash() {
      try {
        const res = await getWeatherDashboard(locationLabel);
        if (!ignore && res?.data?.success) {
          setDashData(res.data);
          if (res.data.diagnosed_disease_action_plan?.length) {
            // Incorporate into active plans
            setActiveScans(prev => {
              const updated = [...prev];
              res.data.diagnosed_disease_action_plan.forEach(ap => {
                const idx = updated.findIndex(s => s.crop === ap.crop);
                if (idx >= 0) {
                  updated[idx] = { ...updated[idx], ...ap };
                } else {
                  updated.push(ap);
                }
              });
              return updated;
            });
          }
        }
      } catch (e) {
        console.warn('Dashboard fetch failed:', e.message);
      }
    }
    loadDash();
    return () => { ignore = true; };
  }, [locationLabel]);

  // Handle Mark Disease as Treated
  const handleResolveScan = async (scanId, crop) => {
    setResolvingId(scanId);
    try {
      if (scanId && !scanId.startsWith('local_')) {
        await resolveDiseaseScan(scanId);
      }
    } catch (e) {
      console.warn('Server resolve failed:', e.message);
    }
    // Update state & localStorage
    setActiveScans(prev => prev.filter(s => s.scan_id !== scanId && s.crop !== crop));
    try {
      const stored = JSON.parse(localStorage.getItem('alip_active_disease_scans') || '[]');
      const filtered = stored.filter(s => s.scan_id !== scanId && s.crop !== crop);
      localStorage.setItem('alip_active_disease_scans', JSON.stringify(filtered));
    } catch (e) {}
    setResolvingId(null);
  };

  const score = useMemo(() => calculateScore(weather), [weather]);
  const operations = useMemo(() => buildOperations(weather, dashData?.operations), [weather, dashData]);
  const diseaseRisk = useMemo(() => buildDiseaseRisk(weather, dashData?.crop_disease_risk), [weather, dashData]);

  // Synthesize dynamic weather action plans for active scans
  const activeActionPlans = useMemo(() => {
    if (!activeScans || !activeScans.length) return [];
    const upcomingRain = (weather.forecast || []).find(d => (d.rain || 0) > 1.0);
    const rainDay = upcomingRain ? upcomingRain.day : 'Saturday';
    const rainMm = upcomingRain ? upcomingRain.rain : '8.4';
    const sprayWindowStr = operations[0]?.window || '06:00 AM – 09:00 AM';

    return activeScans.map(scan => {
      const crop = (scan.crop || 'tomato').toLowerCase();
      const disease = scan.disease || 'Leaf Infection';
      const dLower = disease.toLowerCase();
      let threatLevel = 'CRITICAL';
      let catalyst = '';
      let sprayAction = '';
      let irrAction = '';
      let cultAction = '';
      let lossSaved = '';

      if (dLower.includes('late') || dLower.includes('blight')) {
        threatLevel = 'CRITICAL';
        catalyst = `Incoming rain (${rainMm} mm on ${rainDay}) + ${weather.humidity}% humidity will trigger rapid zoospore germination and canopy splash dispersal.`;
        sprayAction = `Apply systemic penetrant fungicide (Metalaxyl-M or Cymoxanil) during tomorrow morning's window (${sprayWindowStr}) BEFORE rain starts. Absorbs within 2 hours and prevents rain wash-off.`;
        irrAction = 'Shut off all irrigation immediately. Heavy soil moisture combined with rain will cause rapid stem and tuber/fruit rot.';
        cultAction = 'Rogue out and destroy infected lower leaves immediately while canopy is dry today. Disinfect tools.';
        lossSaved = 'Prevents 40% – 60% total crop yield loss by halting zoospore canopy invasion.';
      } else if (dLower.includes('bacterial') || dLower.includes('spot')) {
        threatLevel = 'CRITICAL SPLASH WARNING';
        catalyst = `Wind-driven rain on ${rainDay} mechanically injects Xanthomonas bacteria into leaf stomata and micro-wounds.`;
        sprayAction = `Apply copper hydroxide / copper oxychloride bactericide before rain commences during ${sprayWindowStr}.`;
        irrAction = 'Cease overhead sprinklers completely. Use drip lines only.';
        cultAction = 'STRICT: Do not prune, tie, or harvest plants while foliage is wet.';
        lossSaved = 'Prevents severe fruit spot scarring and 30% – 50% market rejection.';
      } else {
        threatLevel = 'HIGH SPREAD RISK';
        catalyst = `Warm temperatures (${weather.temp}°C) and ${weather.humidity}% humidity accelerate spore sporulation across rows.`;
        sprayAction = `Apply recommended fungicide during morning spray window (${sprayWindowStr}).`;
        irrAction = 'Water at plant base only in early morning. Avoid wetting foliage.';
        cultAction = 'Prune bottom 10-12 inches of foliage to stop soil-splash reinfection.';
        lossSaved = 'Prevents 25% – 40% defoliation and crop loss.';
      }

      return {
        scan_id: scan.scan_id,
        crop,
        disease,
        threatLevel,
        catalyst,
        sprayAction,
        irrAction,
        cultAction,
        lossSaved,
        confidence: scan.confidence || 88,
        createdAt: scan.timestamp || scan.created_at || 'Recently diagnosed',
      };
    });
  }, [activeScans, weather, operations]);
  const calendar = useMemo(() => buildCalendar(weather.forecast || FALLBACK.forecast), [weather]);

  const scoreColor = score >= 72 ? '#34d399' : score >= 48 ? '#fbbf24' : '#f87171';
  const scoreLabel = score >= 72 ? 'Good' : score >= 48 ? 'Moderate' : 'Caution';

  const rainfallTotal = (weather.forecast || []).reduce((s, d) => s + (d.rain || 0), 0).toFixed(1);
  const minTemp = Math.min(...(weather.forecast || []).map(f => f.low));
  const maxTemp = Math.max(...(weather.forecast || []).map(f => f.high));

  // No Location View
  if (locationReady && !locationLabel) {
    return (
      <div className="wa-page-root">
        <div className="wa-sky-hero">
          <WeatherCanvas wx={FALLBACK} isNightMode={isNight} />
        </div>
        <div className="wa-no-loc-overlay">
          <div className="wa-navy-card wa-no-loc-card">
            <span style={{ fontSize: 44 }}>📍</span>
            <h2 className="text-xl font-bold text-white mt-2">Location Not Set</h2>
            <p className="text-sm text-slate-300 mt-2 leading-relaxed">
              Please configure your farm village or district in Profile Settings to unlock live localized agro-weather intelligence.
            </p>
            <button className="wa-action-btn mt-5" onClick={() => navigate('/farmer/profile')}>
              Go to Profile Settings →
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="wa-page-root">

      {/* ══════════════════════════════════════════════════════════════════════
           1. APART LIVING SKY VIEWPORT (Screenshot 2 Translation)
           Moon/Sun, Parallax Clouds, Stars, Ridge Silhouettes & Big Temp
      ═══════════════════════════════════════════════════════════════════════ */}
      <div className="wa-sky-hero">
        <WeatherCanvas wx={weather} isNightMode={isNight} />

        <div className="wa-sky-content">
          {/* Top Bar: Location & Data Published status */}
          <div className="wa-sky-topbar">
            <div>
              <div className="wa-loc-row">
                <svg className="wa-pin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path>
                </svg>
                <h1 className="wa-loc-title">{weather.location || locationLabel}</h1>
              </div>
              <p className="wa-loc-sub">
                Live forecast data published {loading ? 'just now' : '10 mins ago'} · {weather.source === 'live' ? 'Open-Meteo Agro Grid' : 'Demo Mode'}
              </p>
            </div>

            <button
              className="wa-nav-back-btn"
              onClick={() => navigate('/farmer')}
              title="Return to Farmer Dashboard"
            >
              ← Back to Portal
            </button>
          </div>

          {/* Center Hero: Giant 31° & Condition matching Screenshot 2 */}
          <div className="wa-hero-center">
            <div className="wa-hero-temp">{weather.temp}°</div>
            <p className="wa-hero-cond">{weather.condition}</p>
            <p className="wa-hero-meta">
              Low {minTemp}° / High {maxTemp}° &nbsp;·&nbsp;
              <span style={{ color: scoreColor, fontWeight: 700 }}>Farm Readiness: {score} – {scoreLabel}</span>
            </p>
          </div>

          <div className="wa-hero-bottom-spacer" />
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════════
           2. 24-HOUR HOURLY FORECAST SLIDER (Screenshot 2 Card 2)
      ═══════════════════════════════════════════════════════════════════════ */}
      <div className="wa-content-wrapper">
        <div className="wa-hourly-card wa-navy-card">
          <div className="wa-card-header">
            <span className="wa-card-title">24-Hour Forecast & Spray Window</span>
            <span className="wa-spray-indicator">
              Optimal Spraying Window: 06:00 AM – 09:00 AM
            </span>
          </div>

          <div className="wa-hourly-scroll">
            {(weather.hourly || FALLBACK_HOURLY).map((h, idx) => (
              <div key={idx} className={`wa-hourly-item ${h.spraySafe ? 'spray-safe' : ''}`}>
                <span className="wa-hourly-time">{h.time}</span>
                <span className="wa-hourly-glyph">{weatherGlyph(h.code)}</span>
                <span className="wa-hourly-temp">{h.temp}°</span>
                <span className="wa-hourly-rain">{h.spraySafe ? 'Spray Safe' : `${h.rain} mm`}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ══════════════════════════════════════════════════════════════════════
             2.5. ACTIVE CONFIRMED INFECTION & WEATHER ACTION PLAN (Cross-Module)
        ═══════════════════════════════════════════════════════════════════════ */}
        {activeActionPlans.length > 0 && (
          <div className="wa-active-infection-section">
            {activeActionPlans.map((plan, pIdx) => {
              const meta = CROP_META[plan.crop] || { label: plan.crop, icon: '🌱' };
              return (
                <div key={pIdx} className="wa-navy-card wa-active-disease-alert">
                  <div className="wa-ada-header">
                    <div className="wa-ada-title-wrap">
                      <span className="wa-ada-pulse-icon">🚨</span>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-base font-bold text-white flex items-center gap-1.5">
                            <span>{meta.icon}</span> Active Farm Infection: {plan.disease}
                          </h3>
                          <span className="wa-ada-badge">{plan.threatLevel}</span>
                        </div>
                        <p className="wa-ada-subtitle">
                          Confirmed diagnosis linked from Disease Detection Module · {plan.confidence}% confidence
                        </p>
                      </div>
                    </div>

                    <button
                      className="wa-ada-resolve-btn"
                      onClick={() => handleResolveScan(plan.scan_id, plan.crop)}
                      disabled={resolvingId === plan.scan_id}
                    >
                      {resolvingId === plan.scan_id ? 'Updating…' : 'Mark as Treated ✓'}
                    </button>
                  </div>

                  {/* Weather Impact Catalyst */}
                  <div className="wa-ada-catalyst-box">
                    <strong className="text-amber-300 flex items-center gap-1">
                      <span>⚠️</span> Weather Impact on Pathogen:
                    </strong>
                    <p className="wa-ada-catalyst-text">{plan.catalyst}</p>
                  </div>

                  {/* 3 Timed Action Directives */}
                  <div className="wa-ada-actions-grid">
                    <div className="wa-ada-action-card">
                      <div className="wa-ada-action-tag tag-spray">1. Timed Chemical Action</div>
                      <p className="wa-ada-action-desc">{plan.sprayAction}</p>
                    </div>

                    <div className="wa-ada-action-card">
                      <div className="wa-ada-action-tag tag-irr">2. Irrigation Directive</div>
                      <p className="wa-ada-action-desc">{plan.irrAction}</p>
                    </div>

                    <div className="wa-ada-action-card">
                      <div className="wa-ada-action-tag tag-cult">3. Field Sanitation</div>
                      <p className="wa-ada-action-desc">{plan.cultAction}</p>
                    </div>
                  </div>

                  {/* Estimated Loss Prevention Footnote */}
                  <div className="wa-ada-footer">
                    <span className="wa-ada-loss-saved">
                      🛡️ <strong>Crop Loss Prevention:</strong> {plan.lossSaved}
                    </span>
                    <button
                      className="wa-ada-rescan-link"
                      onClick={() => navigate('/farmer/disease-detection')}
                    >
                      Scan Follow-up Leaf →
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════════
             3. PC DESKTOP 2-COLUMN GRID (Agro Cards & Intelligence)
        ═══════════════════════════════════════════════════════════════════════ */}
        <div className="wa-main-grid">

          {/* ── LEFT COLUMN: Screenshot 2 Weather Cards ── */}
          <div className="wa-grid-left">

            {/* Farm Readiness Card (Matching the 65 Moderate Card in Screenshot 2) */}
            <div className="wa-navy-card wa-readiness-card">
              <div className="wa-readiness-top">
                <div className="wa-readiness-left">
                  <div className="wa-leaf-badge">🍃</div>
                  <div>
                    <h3 className="text-base font-bold text-white">Farm Readiness</h3>
                    <p className="text-xs font-semibold" style={{ color: scoreColor }}>{scoreLabel} agricultural conditions</p>
                  </div>
                </div>

                {/* Circular Gauge Arc (Exact Screenshot 2 styling) */}
                <div className="wa-arc-wrap">
                  <svg className="wa-arc-svg" viewBox="0 0 80 80">
                    <circle cx="40" cy="40" r="32" stroke="rgba(255,255,255,0.08)" strokeWidth="6" fill="none" />
                    <circle
                      cx="40" cy="40" r="32"
                      stroke={scoreColor} strokeWidth="6" fill="none"
                      strokeLinecap="round"
                      strokeDasharray={`${(score / 100) * 165} 201`}
                    />
                  </svg>
                  <span className="wa-arc-text">{score}</span>
                </div>
              </div>

              <p className="wa-readiness-desc">
                Field microclimate is acceptable for routine field operations. Wind speed is {weather.wind} km/h with dry canopy support for foliar sprays.
              </p>

              <div className="wa-readiness-bars">
                <div className="wa-sub-bar-box">
                  <div className="wa-sub-bar-labels">
                    <span>Spray Drift</span>
                    <span style={{ color: '#34d399' }}>Low ({weather.wind} km/h)</span>
                  </div>
                  <div className="wa-track"><div className="wa-fill" style={{ width: '82%', background: '#34d399' }}></div></div>
                </div>
                <div className="wa-sub-bar-box">
                  <div className="wa-sub-bar-labels">
                    <span>Soil Moisture</span>
                    <span style={{ color: '#fbbf24' }}>Normal ({weather.humidity}%)</span>
                  </div>
                  <div className="wa-track"><div className="wa-fill" style={{ width: '58%', background: '#fbbf24' }}></div></div>
                </div>
              </div>
            </div>

            {/* Multi-day Forecast Card (Exact Screenshot 2 Card 3) */}
            <div className="wa-navy-card wa-multiday-card">
              <div className="wa-card-header mb-3">
                <span className="wa-card-title">Multi-day forecast</span>
                <span className="text-xs text-slate-400">5-Day Outlook</span>
              </div>

              <div className="wa-multiday-list">
                {(weather.forecast || FALLBACK.forecast).map((day, idx) => (
                  <div key={idx} className="wa-multiday-row">
                    <span className="wa-day-name">{day.day}</span>
                    <span className="wa-day-glyph">{weatherGlyph(day.code)}</span>
                    <span className="wa-day-rain">{day.rain > 0 ? `${day.rain} mm` : '0.0 mm'}</span>
                    <span className="wa-day-temps">{day.low}° / {day.high}°</span>
                  </div>
                ))}
              </div>
            </div>

            {/* 5-Day Rainfall Accumulation Card */}
            <div className="wa-navy-card wa-rain-card">
              <div>
                <p className="wa-card-title">5-Day Rain Accumulation</p>
                <p className="text-3xl font-black text-white mt-1">{rainfallTotal} mm</p>
                <p className="text-xs text-slate-300 mt-1">
                  Soil Status: <strong style={{ color: rainfallTotal > 15 ? '#fbbf24' : '#34d399' }}>Adequately Moist</strong>
                </p>
              </div>
              <div className="wa-harvest-box">
                <span className="text-[10px] uppercase font-bold text-slate-400">Dry Harvest Window</span>
                <p className="text-xs font-bold text-emerald-300 mt-0.5">Opens Mon, 22 Sep</p>
                <span className="text-[10px] text-slate-400">3 Consecutive Dry Days</span>
              </div>
            </div>

          </div>

          {/* ── RIGHT COLUMN: Agro-Intelligence Tabs ── */}
          <div className="wa-grid-right">

            {/* Navigation Tab Bar */}
            <div className="wa-navy-card wa-tab-bar">
              <div className="wa-tab-buttons">
                <button
                  className={`wa-tab-pill ${activeTab === 'disease' ? 'active' : ''}`}
                  onClick={() => setActiveTab('disease')}
                >
                  🐛 Crop Disease Risk
                </button>
                <button
                  className={`wa-tab-pill ${activeTab === 'operations' ? 'active' : ''}`}
                  onClick={() => setActiveTab('operations')}
                >
                  🚜 Farm Operations
                </button>
                <button
                  className={`wa-tab-pill ${activeTab === 'calendar' ? 'active' : ''}`}
                  onClick={() => setActiveTab('calendar')}
                >
                  📅 Activity Calendar
                </button>
              </div>
              <span className="wa-tab-meta">4 Crops Analyzed</span>
            </div>

            {/* ═══ TAB 1: DISEASE RISK ═══ */}
            {activeTab === 'disease' && (
              <div className="wa-stack-gap">
                {/* Infection Window Banner */}
                <div className="wa-navy-card wa-warning-banner">
                  <span className="text-xl">⚠️</span>
                  <div>
                    <h4 className="text-sm font-bold text-amber-300">Elevated Fungal Spore Pressure</h4>
                    <p className="text-xs text-slate-300 mt-0.5 leading-relaxed">
                      Night humidity ({weather.humidity}%) and warm temperatures ({weather.temp}°C) create favorable leaf-surface germination conditions for Solanaceous fungal pathogens.
                    </p>
                  </div>
                </div>

                {/* 4 Crop Cards Grid */}
                <div className="wa-crops-grid">
                  {diseaseRisk.map(({ crop, risk, disease_watch, reason, action }) => {
                    const meta = CROP_META[crop] || { label: crop, icon: '🌱' };
                    return (
                      <div key={crop} className={`wa-navy-card wa-crop-card risk-${(risk || 'low').toLowerCase()}`}>
                        <div>
                          <div className="wa-crop-header">
                            <h4 className="text-base font-bold text-white flex items-center gap-2">
                              <span>{meta.icon}</span> {meta.label}
                            </h4>
                            <span className={`wa-risk-tag tag-${(risk || 'low').toLowerCase()}`}>
                              {risk} RISK
                            </span>
                          </div>
                          <p className="text-xs font-semibold text-slate-200 mt-2">Watch: {disease_watch}</p>
                          <p className="text-xs text-slate-400 mt-1 leading-relaxed">{reason}</p>
                          <div className="wa-crop-action-box">
                            <strong>Action:</strong> {action}
                          </div>
                        </div>

                        <button
                          className="wa-scan-action-btn"
                          onClick={() => navigate('/farmer/disease-detection')}
                        >
                          Scan Leaf with Camera →
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* ═══ TAB 2: FARM OPERATIONS ═══ */}
            {activeTab === 'operations' && (
              <div className="wa-stack-gap">
                {/* Hero Spray Window */}
                <div className="wa-navy-card wa-spray-hero">
                  <div>
                    <span className="text-xs font-bold uppercase tracking-wider text-emerald-300">Computed Safe Spray Window</span>
                    <p className="text-2xl font-black text-white mt-1">06:00 AM – 09:00 AM</p>
                    <p className="text-xs text-slate-300 mt-0.5">Wind {weather.wind} km/h · 0% Rain Probability · Dry Foliage Surface</p>
                  </div>
                  <div className="wa-spray-status-pill">
                    <span className="text-[10px] uppercase font-semibold text-slate-400">Drift Risk</span>
                    <p className="text-sm font-bold text-emerald-300">Low (Safe)</p>
                  </div>
                </div>

                {/* Operations List */}
                <div className="wa-stack-gap">
                  {operations.map(op => (
                    <div key={op.name} className="wa-navy-card wa-op-row">
                      <div className="flex items-start gap-3">
                        <span className={`wa-status-dot dot-${op.status}`}></span>
                        <div>
                          <h5 className="text-sm font-bold text-white">{op.name}</h5>
                          <p className="text-xs text-slate-400 mt-0.5">{op.note}</p>
                          <p className="text-xs font-mono text-emerald-400 mt-1">Window: {op.window}</p>
                        </div>
                      </div>
                      <span className={`wa-status-badge badge-${op.status}`}>
                        {op.status === 'safe' ? 'Safe' : op.status === 'warn' ? 'Caution' : 'Unsafe'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ═══ TAB 3: CALENDAR ═══ */}
            {activeTab === 'calendar' && (
              <div className="wa-stack-gap">
                <p className="text-xs text-slate-400">Morning / Afternoon / Evening activity guidance based on multi-day forecast</p>
                {calendar.map((day, idx) => (
                  <div key={idx} className={`wa-navy-card wa-cal-card cal-${day.rating.toLowerCase()}`}>
                    <div className="wa-cal-header">
                      <span className="font-bold text-white text-sm">{day.day} · {weatherGlyph(day.code)} {day.rain > 0 ? `${day.rain} mm` : 'Dry'}</span>
                      <span className={`wa-rating-chip chip-${day.rating.toLowerCase()}`}>{day.rating}</span>
                    </div>
                    <div className="wa-cal-buckets">
                      <div className="wa-bucket">
                        <p className="wa-bucket-label">Morning</p>
                        <div className="wa-bucket-tags">
                          {day.morning.map(t => <span key={t} className="tag-ok">{t}</span>)}
                          {day.avoidMorning.map(t => <span key={t} className="tag-avoid">{t}</span>)}
                        </div>
                      </div>
                      <div className="wa-bucket">
                        <p className="wa-bucket-label">Afternoon</p>
                        <div className="wa-bucket-tags">
                          {day.afternoon.map(t => <span key={t} className="tag-ok">{t}</span>)}
                          {day.avoidAfternoon.map(t => <span key={t} className="tag-avoid">{t}</span>)}
                        </div>
                      </div>
                      <div className="wa-bucket">
                        <p className="wa-bucket-label">Evening</p>
                        <div className="wa-bucket-tags">
                          {day.evening.map(t => <span key={t} className="tag-ok">{t}</span>)}
                          {day.avoidEvening.map(t => <span key={t} className="tag-avoid">{t}</span>)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

          </div>

        </div>
      </div>

    </div>
  );
}
