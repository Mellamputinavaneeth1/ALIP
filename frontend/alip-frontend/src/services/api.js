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

export const getActiveDiseaseScans = (location = 'all') =>
  api.get(`/disease/active-scans/${encodeURIComponent(location)}`);

export const resolveDiseaseScan = (scanId) =>
  api.post(`/disease/scan/${encodeURIComponent(scanId)}/resolve`);

// ─── Weather APIs ──────────────────────────────────────────
export const getWeatherDashboard = (location, crops = null) => {
  const params = crops ? `?crops=${encodeURIComponent(crops)}` : '';
  return api.get(`/weather/dashboard/${encodeURIComponent(location)}${params}`);
};

// ─── Crop APIs ─────────────────────────────────────────────
export const getCropRecommendation = (data) =>
  api.post('/crop/recommend', data);

export const getCropRecommendationManual = (data) =>
  api.post('/crop/recommend/manual', data);

export const getCropRecommendationRegional = (data) =>
  api.post('/crop/recommend/regional', data);
// payload: { location, latitude?, longitude?, season, measured_ph? }

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

// ─── Enhanced Demand Analysis APIs ─────────────────────────
export const getFullDemandAnalysis = (region, crop, supplyKg, params = {}) =>
  api.get(`/market/demand/analysis/${region}/${crop}/${supplyKg}`, { params });

export const getDynamicThreshold = (region, crop) =>
  api.get(`/market/demand/threshold/${region}/${crop}`);

export const getSupplyTrend = (region, crop, supplyHistory) =>
  api.post(`/market/demand/trend/${region}/${crop}`, supplyHistory);

export const getPriceImpact = (region, crop, supplyKg) =>
  api.get(`/market/demand/price-impact/${region}/${crop}/${supplyKg}`);

export const getTransportCost = (region1, region2, quantityKg, truckType = 'medium') =>
  api.get(`/market/demand/transport-cost/${region1}/${region2}/${quantityKg}`, {
    params: { truck_type: truckType }
  });

// ─── Shipment APIs ─────────────────────────────────────────
export const initiateShipment = (data) =>
  api.post('/market/demand/initiate-shipment', data);

export const updateShipmentStatus = (shipmentId, newStatus) =>
  api.patch(`/market/demand/shipment/${shipmentId}/status/${newStatus}`);

export const getBrokerShipments = (brokerId, status = null, limit = 20) =>
  api.get(`/market/demand/shipments/${brokerId}`, {
    params: { status, limit }
  });

// ─── Supply History APIs ───────────────────────────────────
export const recordSupply = (data) =>
  api.post('/market/demand/record-supply', data);

// ─── Broker Trust APIs ─────────────────────────────────────
export const getBrokerReliability = (brokerId) =>
  api.get(`/market/demand/broker-reliability/${brokerId}`);

export const rateBroker = (data) =>
  api.post('/market/demand/rate-broker', data);

// ─── Profile APIs ──────────────────────────────────────────
export const getProfile = (userId) =>
  api.get(`/auth/profile/${userId}`);

export const updateProfile = (userId, data) =>
  api.patch(`/auth/profile/${userId}`, data);

export default api;
export const getNotifications = () =>
  api.get('/notifications/alerts');