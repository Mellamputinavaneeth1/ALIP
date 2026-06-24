import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add token to every request automatically
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ─── Auth APIs ─────────────────────────────────────────────
export const registerUser = (data) =>
  api.post('/auth/register', data);

export const loginUser = (data) =>
  api.post('/auth/login', data);

// ─── Disease APIs ──────────────────────────────────────────
export const predictDisease = (formData) =>
  api.post('/disease/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });

// ─── Weather APIs ──────────────────────────────────────────
export const getWeatherAdvisory = (location) =>
  api.get(`/weather/advisory/${location}`);

// ─── Crop APIs ─────────────────────────────────────────────
export const getCropRecommendation = (data) =>
  api.post('/crop/recommend', data);

// ─── Market APIs ───────────────────────────────────────────
export const addCropListing = (data) =>
  api.post('/market/listing/add', data);

export const getMarketListings = () =>
  api.get('/market/listings');

export const getRegionalSupply = (region) =>
  api.get(`/market/supply/${region}`);

export const getDemandAnalysis = (region) =>
  api.get(`/market/demand/${region}`);

export const getMarketSummary = () =>
  api.get('/market/summary');

export default api;
export const getNotifications = () =>
  api.get('/notifications/alerts');