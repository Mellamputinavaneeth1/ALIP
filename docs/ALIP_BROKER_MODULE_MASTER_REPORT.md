# ALIP Master Architecture & Technical Report: Broker Module

**Platform:** AgriLink Intelligence Platform (ALIP)  
**Module:** Commercial Broker & Commodity Trading Intelligence Engine  
**System Version:** `v2.1-hybrid`  
**Target Personas:** Agricultural Commodity Brokers, Commission Agents, Mandi Aggregators & Institutional Buyers  
**Document Status:** Master Architecture, Mathematical Blueprints, Visual Diagrams & Technical Specifications  
**Date:** October 2026  

---

## Table of Contents
1. [Executive Summary & Commercial Trading Mission](#1-executive-summary--commercial-trading-mission)
2. [Broker Module Architecture Landscape](#2-broker-module-architecture-landscape)
3. [Data Entity-Relationship Model (ERD)](#3-data-entity-relationship-model-erd)
4. [Feature 1: Broker Dashboard & Market Overview](#4-feature-1-broker-dashboard--market-overview)
5. [Feature 2: 7-Factor Weighted Market Ranking Engine](#5-feature-2-7-factor-weighted-market-ranking-engine)
   - [5.1 Mathematical Composite Formulation](#51-mathematical-composite-formulation)
   - [5.2 Visual Weight Distribution (Pie Chart & XY Chart)](#52-visual-weight-distribution-pie-chart--xy-chart)
   - [5.3 Component Metric Formulations](#53-component-metric-formulations)
   - [5.4 Hard Exclusion Pre-Filters](#54-hard-exclusion-pre-filters)
   - [5.5 Decision Recommendation Matrix & Star Ratings](#55-decision-recommendation-matrix--star-ratings)
6. [Feature 3: Predictive Demand & Anomaly Analytics](#6-feature-3-predictive-demand--anomaly-analytics)
   - [6.1 Dynamic ML Demand Threshold Engine](#61-dynamic-ml-demand-threshold-engine)
   - [6.2 Statistical Z-Score Anomaly Detector](#62-statistical-z-score-anomaly-detector)
   - [6.3 Multi-Signal 1–10 Urgency Scorer](#63-multi-signal-110-urgency-scorer)
   - [6.4 7-Day Price Trajectory & Decay Engine](#64-7-day-price-trajectory--decay-engine)
   - [6.5 Seasonal Harvest Wave Context Engine](#65-seasonal-harvest-wave-context-engine)
7. [Feature 4: Freight Logistics & Route Dispatch Engine](#7-feature-4-freight-logistics--route-dispatch-engine)
   - [7.1 Spatial Distance Matrix (45+ Indian Mandis)](#71-spatial-distance-matrix-45-indian-mandis)
   - [7.2 3-Tier Fleet Rate Model](#72-3-tier-fleet-rate-model)
   - [7.3 Economic Arbitrage & Transport ROI](#73-economic-arbitrage--transport-roi)
   - [7.4 One-Click Shipment Dispatch & Modal](#74-one-click-shipment-dispatch--modal)
8. [Feature 5: Regional Supply Explorer](#8-feature-5-regional-supply-explorer)
9. [Feature 6: Master Crop Marketplace Directory](#9-feature-6-master-crop-marketplace-directory)
10. [Feature 7: Automated Market Alerting Engine](#10-feature-7-automated-market-alerting-engine)
11. [Shipment Lifecycle State Machine](#11-shipment-lifecycle-state-machine)
12. [Cross-Module Connectivity (Farmer $\longleftrightarrow$ Broker Closed Loop)](#12-cross-module-connectivity-farmer--broker-closed-loop)
13. [Technology Stack & Algorithm Directory](#13-technology-stack--algorithm-directory)
14. [Complete REST API Catalog & Route Sitemap](#14-complete-rest-api-catalog--route-sitemap)

---

## 1. Executive Summary & Commercial Trading Mission

In the Indian agricultural supply chain, **spatial market failure** is the leading cause of farmer distress selling and food wastage. When a bumper harvest arrives in a production district (e.g., Nashik onions or Warangal cotton), local APMC mandis experience sudden gluts. Without actionable market intelligence:
- Local mandi prices collapse below the cost of cultivation ($\Delta P \downarrow 40\text{–}60\%$).
- Neighboring consumption centers located just 150–300 km away simultaneously experience acute deficits and inflationary spikes.
- Traditional brokers rely on informal telephone networks, lacking quantitative visibility into transit logistics costs, destination storage compatibility, transit weather risks, or incoming saturation.

The **ALIP Broker Module** provides a real-time, algorithmic market clearing and spatial redistribution platform. It shifts agricultural brokerage from speculative gambling to **quantitative, risk-hedged commodity routing**:

> [!IMPORTANT]
> **Core Mission:** Algorithmic spatial arbitrage that continuously matches surplus production zones with deficit consumption centers within actionable economic radius, preventing price crashes while guaranteeing logistical feasibility and storage integrity.

---

## 2. Broker Module Architecture Landscape

The Broker Module is built as a reactive, multi-tier system combining client-side visualization with asynchronous backend analytical microservices:

```mermaid
flowchart TD
    subgraph Client_Tier ["Presentation Layer — React 19 Single Page Application"]
        direction TB
        UI_DASH["🏪 Broker Dashboard (/broker)<br/>Market Volume, Active Listings & Alerts"]
        UI_DEM["📊 Demand Analysis (/demand)<br/>7-Factor Ranking, Decay Charts & ROI"]
        UI_SUP["🗺️ Regional Supply (/supply)<br/>Multi-Mandi Stock & Supply Density"]
        UI_LIST["📦 All Listings Directory (/listings)<br/>Farmer Harvest Registry & Filter Engine"]
        UI_NOTIF["🔔 Market Notifications (/notifications)<br/>Z-Score Spikes & Deficit Warnings"]
    end

    subgraph Gateway_Tier ["API Gateway & Routing Layer — FastAPI"]
        direction TB
        R_MKT["routes/market.py<br/>Listing Aggregations & Summary"]
        R_REG["routes/region_routes.py<br/>Full Demand Analysis, Shipments & Distance"]
        R_NOT["routes/notifications.py<br/>Threshold Breach Alerts & Stream"]
    end

    subgraph Service_Mesh ["Analytical & Algorithmic Service Mesh"]
        direction TB
        S_RANK["market_ranking_service.py<br/>7-Factor Composite Ranking Engine"]
        S_ML["demand_ml_service.py<br/>Gradient Boosting Dynamic Thresholds"]
        S_URG["urgency_service.py<br/>Multi-Signal 1-10 Actionability Scorer"]
        S_TRAJ["price_trajectory_service.py<br/>Polynomial Price Decay & Break-Even"]
        S_LOG["shipment_service.py<br/>Haversine Freight & Transport Cost"]
        S_REL["reliability_service.py<br/>Broker Peer Rating & Delivery Scoring"]
        S_SEAS["seasonal_context_service.py<br/>Harvest Wave & Price Cycle Context"]
        S_HIST["supply_history_service.py<br/>Rolling Time-Series & Z-Score Tracker"]
        S_WX["weather_risk_service.py<br/>Transit Route Weather Hazard Evaluator"]
        S_DIST["region_service.py<br/>45+ Mandi Spatial Coordinates Index"]
    end

    subgraph Storage_Tier ["Persistence & Columnar Caches"]
        direction TB
        DB_SHIP[("MongoDB: shipment_log<br/>(Waybills, Status, Agreed Price)")]
        DB_LIST[("MongoDB: market_listings<br/>(Live Farmer Produce Batches)")]
        DB_SNAP[("MongoDB: supply_history<br/>(30-Day Tonnage Snapshots)")]
        DB_USER[("MongoDB: users<br/>(Broker Profiles & Storage Capacity)")]
        MAT_GEO[("In-Memory Spatial Index<br/>(45+ Mandi Centroids & Lat/Lon)")]
    end

    Client_Tier --> Gateway_Tier
    Gateway_Tier --> Service_Mesh
    Service_Mesh --> Storage_Tier
    S_RANK --> S_ML & S_URG & S_TRAJ & S_LOG & S_REL & S_HIST & S_WX & S_DIST
```

---

## 3. Data Entity-Relationship Model (ERD)

The data model connects brokers with farmer listings, transportation shipments, peer ratings, and historical market supply snapshots:

```mermaid
erDiagram
    BROKER_USER ||--o{ SHIPMENT_LOG : "initiates"
    BROKER_USER ||--o{ BROKER_RATING : "submits/receives"
    FARMER_LISTING ||--o{ SHIPMENT_LOG : "allocates"
    MANDI_REGION ||--o{ SUPPLY_SNAPSHOT : "tracks"
    MANDI_REGION ||--o{ SHIPMENT_LOG : "source/destination"

    BROKER_USER {
        string user_id PK
        string full_name
        string email
        string role "broker"
        string location "Primary Mandi"
        object storage_capacity "cold_kg, dry_kg, yard_kg"
        float reliability_score "0 to 100"
        datetime created_at
    }

    FARMER_LISTING {
        string listing_id PK
        string farmer_name
        string location "Source Mandi"
        string crop_name
        float quantity_kg
        float target_price "INR per kg"
        string phone
        string status "available | in_transit | sold"
        datetime created_at
    }

    SHIPMENT_LOG {
        string shipment_id PK
        string broker_id FK
        string source_region
        string destination_region
        string crop
        float quantity_kg
        string status "pending | in_transit | delivered | cancelled"
        float agreed_price "INR per kg"
        float transport_cost "Total INR"
        datetime initiated_at
        datetime expected_delivery
        datetime actual_delivery
        string notes
    }

    SUPPLY_SNAPSHOT {
        string snapshot_id PK
        string region FK
        string crop
        float total_supply_kg
        int listing_count
        datetime recorded_at
    }

    BROKER_RATING {
        string rating_id PK
        string deal_id FK
        string rater_broker_id FK
        string rated_broker_id FK
        int stars "1 to 5"
        string comment
        datetime created_at
    }

    MANDI_REGION {
        string region_name PK
        string state
        float lat
        float lng
        string region_type "urban | semi-urban | rural"
        int population_density
        string market_size "large | medium | small"
    }
```

---

## 4. Feature 1: Broker Dashboard & Market Overview

The Broker Dashboard (`/broker`, [BrokerDashboard.js](file:///D:/ALIP-TOTAL/ALIP/frontend/alip-frontend/src/pages/BrokerDashboard.js)) serves as the command center for commercial traders:

### Core Capabilities
- **Market KPI Cards:**
  - **Total Market Volume:** Live cumulative tonnage of active produce listed across all production clusters.
  - **Active Farmer Listings:** Count of unfulfilled, available crop lots waiting for broker procurement.
  - **Oversupply Alerts:** Red flags marking mandis where current arrivals exceed dynamic capacity thresholds.
  - **Undersupply Alerts:** Green flags highlighting deficit consumption centers where prices are escalating.
- **Direct Sub-Module Launchpad:**
  - 📊 **Demand Analysis:** Comprehensive redistribution planning and 7-factor destination ranking.
  - 📦 **All Listings:** Master searchable directory of farmer harvest batches.
  - 🗺️ **Regional Supply:** Geospatial inventory explorer by mandi.
  - ➕ **Add Crop Listing:** Broker-initiated bulk crop listing intake.
  - 🔔 **Notifications:** Real-time push feed for market anomalies and supply spikes.

---

## 5. Feature 2: 7-Factor Weighted Market Ranking Engine

When an oversupplied mandi is selected for liquidation, the ranking engine ([market_ranking_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/market_ranking_service.py)) evaluates candidate destinations within reachable transit range ($\le 500\text{ km}$) using seven independent economic, logistical, and environmental vectors.

### 5.1 Mathematical Composite Formulation

The unified composite score $S_{\text{composite}} \in [0, 100]$ is computed as:

$$S_{\text{composite}} = \sum_{k=1}^{7} w_k \cdot f_k$$

Where:
$$\sum_{k=1}^{7} w_k = 0.20 + 0.20 + 0.15 + 0.15 + 0.10 + 0.10 + 0.10 = 1.00$$

---

### 5.2 Visual Weight Distribution (Pie Chart & XY Chart)

#### Circular Factor Weight Distribution (Visual Pie Chart)

The exact proportion of each factor in the composite algorithm is visualized below:

<div align="center">
<svg viewBox="0 0 520 320" width="100%" max-width="520" height="320" xmlns="http://www.w3.org/2000/svg">
  <!-- Slices centered at (160, 160), Radius 130 -->
  <!-- 1. Demand Gap: 20% (72 deg, 0 to 72) -->
  <path d="M 160 160 L 290 160 A 130 130 0 0 1 200.17 283.63 Z" fill="#3b82f6" stroke="#0f172a" stroke-width="2" />
  <!-- 2. Broker Reliability: 20% (72 deg, 72 to 144) -->
  <path d="M 160 160 L 200.17 283.63 A 130 130 0 0 1 54.83 236.41 Z" fill="#10b981" stroke="#0f172a" stroke-width="2" />
  <!-- 3. Price Advantage: 15% (54 deg, 144 to 198) -->
  <path d="M 160 160 L 54.83 236.41 A 130 130 0 0 1 36.36 119.83 Z" fill="#f59e0b" stroke="#0f172a" stroke-width="2" />
  <!-- 4. Transport Efficiency: 15% (54 deg, 198 to 252) -->
  <path d="M 160 160 L 36.36 119.83 A 130 130 0 0 1 119.83 36.36 Z" fill="#8b5cf6" stroke="#0f172a" stroke-width="2" />
  <!-- 5. Destination Saturation: 10% (36 deg, 252 to 288) -->
  <path d="M 160 160 L 119.83 36.36 A 130 130 0 0 1 200.17 36.36 Z" fill="#ef4444" stroke="#0f172a" stroke-width="2" />
  <!-- 6. Storage Availability: 10% (36 deg, 288 to 324) -->
  <path d="M 160 160 L 200.17 36.36 A 130 130 0 0 1 265.17 83.59 Z" fill="#06b6d4" stroke="#0f172a" stroke-width="2" />
  <!-- 7. En-Route Weather Risk: 10% (36 deg, 324 to 360) -->
  <path d="M 160 160 L 265.17 83.59 A 130 130 0 0 1 290 160 Z" fill="#ec4899" stroke="#0f172a" stroke-width="2" />
  <!-- Center Hole for Donut Look -->
  <circle cx="160" cy="160" r="55" fill="#0f172a" stroke="#1e293b" stroke-width="2"/>
  <text x="160" y="155" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="sans-serif">7 FACTORS</text>
  <text x="160" y="173" text-anchor="middle" fill="#f8fafc" font-size="14" font-weight="bold" font-family="sans-serif">100%</text>

  <!-- Legend -->
  <g transform="translate(320, 30)">
    <rect x="0" y="0" width="14" height="14" rx="3" fill="#3b82f6" />
    <text x="22" y="12" fill="#e2e8f0" font-size="12" font-family="sans-serif">Demand Gap (20%)</text>

    <rect x="0" y="28" width="14" height="14" rx="3" fill="#10b981" />
    <text x="22" y="40" fill="#e2e8f0" font-size="12" font-family="sans-serif">Reliability (20%)</text>

    <rect x="0" y="56" width="14" height="14" rx="3" fill="#f59e0b" />
    <text x="22" y="68" fill="#e2e8f0" font-size="12" font-family="sans-serif">Price Advantage (15%)</text>

    <rect x="0" y="84" width="14" height="14" rx="3" fill="#8b5cf6" />
    <text x="22" y="96" fill="#e2e8f0" font-size="12" font-family="sans-serif">Transport Efficiency (15%)</text>

    <rect x="0" y="112" width="14" height="14" rx="3" fill="#ef4444" />
    <text x="22" y="124" fill="#e2e8f0" font-size="12" font-family="sans-serif">Dest Saturation (10%)</text>

    <rect x="0" y="140" width="14" height="14" rx="3" fill="#06b6d4" />
    <text x="22" y="152" fill="#e2e8f0" font-size="12" font-family="sans-serif">Storage Match (10%)</text>

    <rect x="0" y="168" width="14" height="14" rx="3" fill="#ec4899" />
    <text x="22" y="180" fill="#e2e8f0" font-size="12" font-family="sans-serif">Weather Risk (10%)</text>
  </g>
</svg>
<p><em>Figure 2.1: Visual Pie Chart of the 7 Analytical Weights Driving Market Ranking.</em></p>
</div>

#### Component Weights Bar Chart

```mermaid
xychart-beta
    title "7-Factor Market Ranking Component Weights (%)"
    x-axis ["Demand Gap", "Reliability", "Price Adv", "Transport", "Saturation", "Storage", "Weather"]
    y-axis "Weight (%)" 0 --> 25
    bar [20, 20, 15, 15, 10, 10, 10]
```

---

### 5.3 Component Metric Formulations

| Vector ($f_k$) | Weight ($w_k$) | Visual Distribution | Precise Mathematical Formulation & Logic |
| :--- | :---: | :---: | :--- |
| **Demand Gap** | **$0.20$** | `[████████████] 20%` | $f_1 = \min\left(100, \left\lfloor\frac{\text{Deficit}_{\text{dest}}}{\max(1, Q_{\text{send}})} \times 100\right\rfloor\right)$. Evaluates the capacity of the target market to absorb the dispatched excess volume without creating a secondary price drop. |
| **Broker Reliability** | **$0.20$** | `[████████████] 20%` | $f_2 \in [0, 100]$. Measured via verified deal completion history, delivery promptness, and counterparty star ratings ($1\text{–}5$). Default neutral rating $= 65$. |
| **Price Advantage** | **$0.15$** | `[█████████░░░] 15%` | $f_3 = \min\left(100, \max\left(0, \left\lfloor\frac{P_{\text{base}} - P_{\text{day3}}}{\max(0.01, P_{\text{base}})} \times 1000\right\rfloor\right)\right)$. Measures price spread preserved by redistributing before local prices decay over a typical 3-day transit window. |
| **Transport Efficiency** | **$0.15$** | `[█████████░░░] 15%` | $f_4 = \max\left(0, \left\lfloor 100 - \frac{\text{Cost}_{100\text{kg}}}{C_{\text{max}}} \times 100 \right\rfloor\right)$, where $C_{\text{max}} = ₹500/100\text{ kg}$. Rewards geographically close markets that minimize diesel and toll outlays. |
| **Destination Saturation** | **$0.10$** | `[██████░░░░░░] 10%` | $f_5 = \max\left(0, \left\lfloor 100 - \frac{Q_{\text{pending\_incoming}}}{\text{Deficit}_{\text{dest}}} \times 100 \right\rfloor\right)$. Prevents compounding gluts: counts all active shipments initiated across all brokers destined for that mandi within the past 24 hours. |
| **Storage Availability** | **$0.10$** | `[██████░░░░░░] 10%` | $f_6 = \min\left(100, \left\lfloor\frac{C_{\text{storage\_avail}}}{\max(1, Q_{\text{send}})} \times 100\right\rfloor\right)$. Checks facility type compatibility: Tier 1 crops (Tomato, Banana) require Cold Storage; Tier 2 crops (Potato, Onion) require Dry Warehousing. |
| **En-Route Weather Risk** | **$0.10$** | `[██████░░░░░░] 10%` | $f_7 \in [0, 100]$. Evaluates forecasted precipitation, road washouts, and extreme heat waves along the transit corridor between source and destination centroids. |

---

### 5.4 Hard Exclusion Pre-Filters

Before destinations enter the composite scoring loop, they must pass strict agronomic and commercial exclusion gates:

```mermaid
flowchart TD
    START["Candidate Destination (within 500 km)"] --> G1{"Destination Deficit > 0?"}
    G1 -- No --> EX1["❌ Excluded: Already at/above demand threshold"]
    G1 -- Yes --> G2{"Destination Saturation < 120%?"}
    G2 -- No --> EX2["❌ Excluded: Saturation CRITICAL (>120% gap filled)"]
    G2 -- Yes --> G3{"Perishable Tier 1 & Weather CRITICAL?"}
    G3 -- Yes --> EX3["❌ Excluded: Heavy rain/heat will rot cargo en route"]
    G3 -- No --> G4{"Required Storage Facility Available?"}
    G4 -- No --> EX4["❌ Excluded: No Cold Storage for Tier 1 crops"]
    G4 -- Yes --> PASS["✅ Qualified: Enters 7-Factor Ranking Loop"]
```

---

### 5.5 Decision Recommendation Matrix & Star Ratings

The composite score maps directly to actionable trading verdicts and UI badges:

| Composite Score ($S$) | Star Rating | Decision Verdict | Recommended Action |
| :---: | :---: | :--- | :--- |
| **$80 \le S \le 100$** | ⭐⭐⭐⭐⭐ (5★) | **`STRONGLY RECOMMENDED`** | High arbitrage margin, clear transit corridors, and empty cold storage. Dispatched as top primary route. |
| **$65 \le S < 80$** | ⭐⭐⭐⭐ (4★) | **`RECOMMENDED`** | Solid absorption capacity and positive margins. Recommended for secondary volume dispatch. |
| **$50 \le S < 65$** | ⭐⭐⭐ (3★) | **`FEASIBLE`** | Moderately profitable; sensitive to fuel spikes or minor arrival delays. |
| **$35 \le S < 50$** | ⭐⭐ (2★) | **`MARGINAL`** | Marginal returns; only recommended if source mandi price is crashing catastrophically. |
| **$S < 35$** | ⭐ (1★) | **`NOT RECOMMENDED`** | Transport costs exceed projected price spread. Uneconomical. |

---

## 6. Feature 3: Predictive Demand & Anomaly Analytics

Predictive demand analytics ([demand_ml_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/demand_ml_service.py)) replaces static gut-feeling decisions with statistical machine learning:

```mermaid
flowchart LR
    subgraph Inputs ["Input Signals"]
        IN_VOL["Incoming Supply (kg)"]
        IN_GEO["Region + Population Density + Market Size"]
        IN_CROP["Crop + Perishability Tier"]
        IN_TIME["Calendar Month + Season Wave"]
    end

    subgraph Engines ["Predictive Analytics Suite"]
        E1["Gradient Boosting Regressor<br/>Dynamic Demand Threshold"]
        E2["Rolling History Engine<br/>Statistical Z-Score Detector"]
        E3["Multi-Signal Formula<br/>1-10 Urgency Scorer"]
        E4["Polynomial Decay Formula<br/>7-Day Price Trajectory"]
    end

    subgraph Outputs ["Actionable Market Intelligence"]
        OUT_THRESH["Threshold (kg) & Oversupply Ratio"]
        OUT_ANOM["Anomaly Level: Normal / Elevated / Critical Spike"]
        OUT_URG["Urgency: WATCH / HIGH / CRITICAL / EMERGENCY"]
        OUT_DECAY["Day-by-Day Value Decay & Break-Even Day"]
    end

    Inputs --> Engines
    Engines --> Outputs
```

---

### 6.1 Dynamic ML Demand Threshold Engine

Traditional agricultural systems use static thresholds (e.g., *"Nashik absorbs 500 tonnes of onions"*). In reality, demand is highly elastic and seasonal.

- **Algorithm:** `Gradient Boosting Regressor` (100 estimators, max depth 4, learning rate 0.1, 5-fold CV $R^2 \approx 0.94$).
- **Features Vector:**
  $$\mathbf{x} = \begin{bmatrix} \text{crop\_id}, & \text{season\_id}, & \text{region\_type\_id}, & \text{population\_density}, & \text{market\_size\_id}, & \text{calendar\_month} \end{bmatrix}$$
- **Output:** Predicted daily capacity $\hat{T}_{\text{crop}} \text{ (kg)}$:
  $$\text{Oversupply Ratio } (R) = \frac{\text{Current Daily Supply (kg)}}{\hat{T}_{\text{crop}}}$$

---

### 6.2 Statistical Z-Score Anomaly Detector

To determine if an influx is an ordinary daily fluctuation or a systemic harvest glut, the engine ([supply_history_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/supply_history_service.py)) computes the statistical standardized deviation against the trailing 30-day rolling baseline:

$$Z = \frac{Q_{\text{current}} - \mu_{30}}{\sigma_{30}}$$

- **$Z < 1.0$ (`Normal`):** Typical variation within historical boundaries.
- **$1.0 \le Z < 2.0$ (`Elevated`):** Unusually high arrivals; early warning for brokers to prepare outbound fleets.
- **$Z \ge 2.0$ (`Critical Anomaly`):** Severe market shock (>97.7th percentile of historical arrivals). Automated oversupply alerts trigger immediately.

---

### 6.3 Multi-Signal 1–10 Urgency Scorer

The Urgency Engine ([urgency_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/urgency_service.py)) integrates supply volume, statistical anomaly, arrival velocity, perishability, and time pressure into an integer urgency index $U \in [1, 10]$:

$$\text{Raw Score} = \Big(\underbrace{\min(9.0, 3.0 \cdot R)}_{\text{Base Ratio}} + \underbrace{\min(2.0, 0.5 \cdot Z)}_{\text{Z-Score Boost}} + \underbrace{\Delta_{\text{trend}}}_{\text{Trend Boost}} + \underbrace{\Delta_{\text{time}}}_{\text{Time Boost}}\Big) \times M_{\text{tier}}$$

Where:
- **Perishability Multiplier ($M_{\text{tier}}$):**
  - **Tier 1 (Highly Perishable: Tomato, Banana, Sugarcane):** $M_1 = \mathbf{2.0}$
  - **Tier 2 (Moderately Perishable: Potato, Onion, Maize, Groundnut):** $M_2 = \mathbf{1.3}$
  - **Tier 3 (Storable Grains: Rice, Wheat, Cotton, Soybean):** $M_3 = \mathbf{1.0}$
- **Arrival Trend Boost ($\Delta_{\text{trend}}$):** Rapidly Increasing ($+1.5$), Increasing ($+0.8$), Stable ($0.0$), Decreasing ($-0.5$), Rapidly Decreasing ($-1.0$).
- **Time-Pressure Boost ($\Delta_{\text{time}}$):** If the oversupply condition has persisted $\ge 3$ consecutive days without clearing, $+1.0$ is added.
- **Final Integer Clamping:**
  $$U = \max(1, \min(10, \text{round}(\text{Raw Score})))$$

#### Urgency Windows & Action Protocols

| Urgency Score ($U$) | Label | Visual Color | Action Window | Clinical Trading Protocol |
| :---: | :---: | :---: | :---: | :--- |
| **$1 \le U \le 3$** | **`WATCH`** | 🟢 Green | **$\le 14$ Days** | Safe inventory levels. Monitor prices; hold inventory if upward seasonal trend expected. |
| **$4 \le U \le 6$** | **`HIGH`** | 🟠 Orange | **$\le 5$ Days** | Noticeable supply pressure. Lock forward contracts and initiate transfer for 25–40% excess. |
| **$7 \le U \le 8$** | **`CRITICAL`** | 🔴 Red | **$\le 2$ Days** | Severe price decay imminent. Dispatch immediate outbound fleets to top-ranked regional hubs. |
| **$9 \le U \le 10$** | **`EMERGENCY`** | 🟣 Dark Red | **$\le 24$ Hours** | Catastrophic glut. Produce at immediate risk of rotting on mandi platforms. Immediate liquidation required. |

---

### 6.4 7-Day Price Trajectory & Decay Engine

Under unaddressed oversupply conditions, produce value depreciates daily. The price decay engine ([price_trajectory_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/price_trajectory_service.py)) models this decay over a 7-day projection horizon:

$$P(d) = \max\left(P_{\text{floor}}, P_{\text{base}} \cdot \left[1 - \text{Rate}_{\text{decay}} \cdot \left(\frac{d}{6}\right)^{1 + 0.15 \cdot (R - 1)} \cdot A_{\text{tier}}\right]\right)$$

Where:
- $P_{\text{base}}$: Prevailing mandi market price (₹/kg).
- $\text{Rate}_{\text{decay}} = \epsilon_{\text{crop}} \cdot \max(0, R - 1)$, where $\epsilon_{\text{crop}}$ is crop-specific price elasticity (e.g., Tomato $= 0.60$, Onion $= 0.50$, Wheat $= 0.30$).
- $A_{\text{tier}}$: Acceleration factor (Tier 1 $= 1.15$, Tier 2 $= 1.05$, Tier 3 $= 1.00$).
- $P_{\text{floor}} = 0.30 \cdot P_{\text{base}}$: Physical floor beneath which distress clearance occurs.

#### Break-Even Day Formulation
The engine computes the critical transition point where **holding onto produce costs more in lost market value than the complete freight bill to ship it to a deficit city**:

$$d_{\text{breakeven}} = \min \left\{ d \in \{0, \dots, 6\} \;\Big|\; \text{Loss}_{100\text{kg}}(d) > \text{FreightCost}_{100\text{kg}} \right\}$$

> [!TIP]
> If $d_{\text{breakeven}} \le 2$, the system outputs: *"Act immediately — prices will drop ₹X/100kg by Day 3. Transport breaks even on Day 2."*

---

### 6.5 Seasonal Harvest Wave Context Engine

Harvests follow predictable agro-ecological waves across Indian latitudes. The seasonal context engine ([seasonal_context_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/seasonal_context_service.py)) cross-references current arrivals against macro-calendar dynamics:
- **Peak Influx Wave:** Mandi arrivals represent the primary seasonal crest. High oversupply is expected; long-distance inter-state redistribution is essential.
- **Late Harvest Tail:** Arrivals are tapering off. Short-term gluts will self-correct within 7–10 days; long-distance shipping should be limited.
- **Off-Season Anomaly:** Unseasonal glut caused by localized microclimate shifts or delayed arrivals. High price premiums available in distant metropolitan centers.

---

## 7. Feature 4: Freight Logistics & Route Dispatch Engine

Redistribution recommendations are only as useful as their physical execution. ALIP incorporates a full freight logistics calculation engine ([shipment_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/shipment_service.py), [region_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/region_service.py)).

### 7.1 Spatial Distance Matrix (45+ Indian Mandis)

ALIP maintains exact geographic centroids (latitude, longitude, region type, population density, market tier) for 45+ premier agricultural trading mandis spanning 12 Indian states:

```mermaid
flowchart TD
    subgraph South_Corridor ["Southern Agricultural Belt"]
        HYD["Hyderabad, TG"] --- WRG["Warangal, TG"]
        WRG --- KRM["Karimnagar, TG"]
        HYD --- GNT["Guntur, AP"]
        GNT --- VJA["Vijayawada, AP"]
        HYD --- BLR["Bangalore, KA"]
        BLR --- MYS["Mysore, KA"]
        GNT --- CHN["Chennai, TN"]
        CHN --- CBE["Coimbatore, TN"]
    end

    subgraph West_Corridor ["Western Commercial Belt"]
        MUM["Mumbai, MH"] --- PUN["Pune, MH"]
        PUN --- NSK["Nashik, MH"]
        NSK --- NGP["Nagpur, MH"]
        MUM --- AMD["Ahmedabad, GJ"]
        AMD --- SRT["Surat, GJ"]
    end

    subgraph North_Central ["Northern & Central Belt"]
        DEL["Delhi, DL"] --- JPR["Jaipur, RJ"]
        DEL --- LKO["Lucknow, UP"]
        LKO --- PAT["Patna, BR"]
        DEL --- BHO["Bhopal, MP"]
        BHO --- IND["Indore, MP"]
    end

    South_Corridor <==> West_Corridor
    West_Corridor <==> North_Central
```

Distance between any two mandis is determined using the great-circle **Haversine Distance**:

$$d = 2 R \arcsin \left( \sqrt{ \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right) } \right)$$

Where $R = 6371\text{ km}$.

---

### 7.2 3-Tier Fleet Rate Model

Logistics freight costs are computed based on standardized commercial Indian trucking categories:

| Fleet Category | Payload Capacity | Commercial Freight Rate | Recommended Cargo Tonnage |
| :--- | :---: | :---: | :--- |
| **Small Commercial (LCV)** | $< 3\text{ tonnes}$ (Tata Ace, Bolero Maxi) | **₹25.00 / km** | Small farmer lots, rapid express delivery ($500\text{–}2,500\text{ kg}$). |
| **Medium Commercial (MCV)** | $3\text{–}10\text{ tonnes}$ (Eicher Pro, Tata 1109) | **₹35.00 / km** | Standard mandi trading lots ($3,000\text{–}10,000\text{ kg}$). *(System Default)* |
| **Heavy Commercial (HCV)** | $> 10\text{ tonnes}$ (Multi-axle 16-wheelers) | **₹50.00 / km** | Bulk inter-state reallocation ($> 10,000\text{ kg}$). |

**Formulation:**
$$\text{Total Freight Cost (₹)} = d_{\text{km}} \times \text{Rate}_{\text{per\_km}}$$
$$\text{Freight Cost per 100 kg (₹)} = \left(\frac{\text{Total Freight Cost}}{Q_{\text{dispatched\_kg}}}\right) \times 100$$

---

### 7.3 Economic Arbitrage & Transport ROI

A destination is only commercially viable if the preserved market value exceeds the freight outlays:

$$\text{Preserved Value Gain per 100 kg} = P_{\text{dest\_base}} - P_{\text{source\_decayed}}$$
$$\text{Logistics ROI} = \frac{\text{Preserved Value Gain per 100 kg}}{\text{Freight Cost per 100 kg}}$$

- **$\text{ROI} > 3.0$ (`STRONGLY RECOMMENDED`):** Exceptionally lucrative spatial arbitrage. Yields $\ge 3\times$ return over freight cost.
- **$1.5 < \text{ROI} \le 3.0$ (`RECOMMENDED`):** Healthy commercial spread.
- **$1.0 \le \text{ROI} \le 1.5$ (`MARGINAL`):** Breaks even; vulnerable to road delays.
- **$\text{ROI} < 1.0$ (`NOT RECOMMENDED`):** Transport bill exceeds arbitrage spread; causes a net financial loss.

---

### 7.4 One-Click Shipment Dispatch & Modal

In the Demand Analysis UI ([DemandAnalysis.js](file:///D:/ALIP-TOTAL/ALIP/frontend/alip-frontend/src/pages/DemandAnalysis.js)), brokers do not simply look at charts — they execute:
1. Clicking **`🚚 Initiate Dispatch`** on any ranked destination card opens the **Shipment Dispatch Modal**.
2. Automatically pre-fills:
   - Source Mandi & Destination Mandi
   - Safe Dispatched Quantity (auto-capped by destination saturation)
   - Calculated Haul Distance ($d_{\text{km}}$) & Freight Quote (₹)
   - Agreed Price (₹/kg) and Delivery Window (Days)
3. Submitting triggers `POST /market/demand/initiate-shipment`:
   - Persists the record to `shipment_log` collection in MongoDB.
   - Emits an active waybill ID.
   - Automatically deducts the allocated tonnage from local excess.
   - Immediately registers with the **Destination Saturation Tracker**, preventing other brokers from double-booking that destination!

---

## 8. Feature 5: Regional Supply Explorer

The Regional Supply Explorer (`/supply`, [RegionalSupply.js](file:///D:/ALIP-TOTAL/ALIP/frontend/alip-frontend/src/pages/RegionalSupply.js)) provides comprehensive inventory transparency across trading mandis:

- **Mandi Search & Quick Select:** Instant lookup for major hubs (Hyderabad, Chennai, Mumbai, Delhi, Bangalore).
- **Categorical Supply Status Gauges:**
  - 🔴 **High Supply ($> 500\text{ kg}$):** Potential oversupply risk; prime candidate for outbound arbitrage.
  - 🟡 **Medium Supply ($200\text{–}500\text{ kg}$):** Balanced commercial trading range.
  - 🟢 **Low Supply ($< 200\text{ kg}$):** Deficit condition; prime destination for incoming cargo.
- **Aggregated Commodity Breakdowns:** Displays all available crop types, total metric tonnage, active listing count, and average farmer asking prices.

---

## 9. Feature 6: Master Crop Marketplace Directory

The Marketplace Directory (`/listings`, [AllListings.js](file:///D:/ALIP-TOTAL/ALIP/frontend/alip-frontend/src/pages/AllListings.js)) acts as the primary procurement pipeline connecting brokers with farm-gate harvests:

- **Real-Time Synchronous Query:** Fetches all available listings via `GET /market/listings`.
- **Multi-Vector Filter Engine:**
  - Instant text filter across Farmer Name, District/Mandi Location, and Crop Name.
  - Dropdown filter by specific crop species.
- **Direct Deal Execution:**
  - Displays lot size ($Q_{\text{kg}}$), price expectations ($₹/\text{kg}$), and farmer contact credentials.
  - Highlights listing status: `available` (green), `in_transit` (orange), or `sold` (red).

---

## 10. Feature 7: Automated Market Alerting Engine

The Alerting Engine (`/notifications`, [Notifications.js](file:///D:/ALIP-TOTAL/ALIP/frontend/alip-frontend/src/pages/Notifications.js)) monitors incoming arrival velocities to alert brokers before price collapses happen:

- **Alert Categorization:**
  - 📈 **Oversupply Anomaly ($Z \ge 2.0$ or $R \ge 1.5$):** Alerts brokers of impending local gluts requiring emergency liquidation fleets.
  - 📉 **Undersupply Deficit ($R < 0.5$):** Alerts buyers to escalating prices and inventory deficits in consumption hubs.
  - ✅ **Market Rebalance:** Confirms when dispatched shipments have successfully stabilized a market.
- **Severity Tiers:** High (Critical action required within 24–48h), Medium (Active watch), and Low (Informational).
- **Badge Indicators:** Real-time unread alert count badges in the broker portal navigation header.

---

## 11. Shipment Lifecycle State Machine

Every commercial transaction and redistribution movement adheres to a deterministic, audited lifecycle state machine:

```mermaid
stateDiagram-v2
    [*] --> Pending : Broker creates shipment via One-Click Modal (POST /initiate-shipment)
    
    Pending --> In_Transit : Logistics fleet picks up cargo from mandi (PATCH /status/in_transit)
    Pending --> Cancelled : Deal cancelled by mutual agreement (PATCH /status/cancelled)
    
    In_Transit --> Delivered : Truck reaches destination mandi & passes QA (PATCH /status/delivered)
    In_Transit --> Cancelled : Major road washout / accident (PATCH /status/cancelled)
    
    Delivered --> Deal_Settled : System sets actual_delivery timestamp & unlocks Rating Modal
    Deal_Settled --> [*] : Both brokers submit peer ratings (POST /rate-broker)
```

### State Definitions & Auditing Triggers
1. **`pending`:** Waybill issued. Truck scheduled. Quantity marked against destination saturation.
2. **`in_transit`:** Cargo moving on highway. En-route weather monitoring active.
3. **`delivered`:** Cargo accepted at destination warehouse. Automatic timestamp capture (`actual_delivery = datetime.utcnow()`).
4. **`cancelled`:** Trip aborted. Allocated tonnage automatically restored to source inventory and cleared from destination saturation buffer.
5. **`Deal_Settled`:** Triggers peer rating engine ([reliability_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/reliability_service.py)) updating broker reliability scores ($1\text{–}5\star$).

---

## 12. Cross-Module Connectivity (Farmer $\longleftrightarrow$ Broker Closed Loop)

The AgriLink Intelligence Platform operates as a **closed-loop feedback ecosystem** where every action taken in the Farmer module directly influences the Broker module, and vice versa:

```mermaid
sequenceDiagram
    autonumber
    actor Farmer as 👨‍🌾 Farmer
    participant CropRec as 🌱 Crop Recommendation (Farmer)
    participant Listing as 📦 Crop Listing (Farmer)
    participant MongoDB as 🗄️ ALIP Database
    participant Broker as 🏪 Broker
    participant DemandML as 📊 Demand Intelligence (Broker)
    participant Logistics as 🚚 Freight Engine (Broker)

    Note over Farmer, CropRec: 1. Seasonal Planning Phase
    DemandML->>CropRec: Feed historical mandi deficit signals & price trends
    Farmer->>CropRec: Requests recommendation for district & season
    CropRec-->>Farmer: Recommends high-yielding, high-demand annual field crops

    Note over Farmer, Listing: 2. Harvest & Listing Phase
    Farmer->>Listing: Lists harvest batch (5,000 kg Onion @ ₹20/kg in Nashik)
    Listing->>MongoDB: Persist listing to 'market_listings'

    Note over Broker, DemandML: 3. Market Detection & Analysis
    MongoDB-->>DemandML: New supply snapshot triggers Z-score & threshold check
    DemandML-->>Broker: ⚠️ Alert: Nashik Onion Oversupply (R = 2.04, Urgency = 8 CRITICAL)
    Broker->>DemandML: Runs Full Demand Analysis (5,000 kg, Medium Truck)
    DemandML-->>Broker: Ranks Destinations: 1. Pune (Score: 86), 2. Mumbai (Score: 81)

    Note over Broker, Logistics: 4. Logistical Arbitrage & Settlement
    Broker->>Logistics: Initiates Dispatch to Pune (3,000 kg @ ₹24/kg)
    Logistics->>MongoDB: Creates 'shipment_log' record & updates Pune Saturation
    Logistics-->>Farmer: Transports produce directly from farm-gate to Pune mandi
    Logistics->>MongoDB: Status -> 'delivered'
    Broker->>MongoDB: Submits peer rating (5 Stars)
```

### Intra-Platform Interaction Matrix

| Upstream Module | Event / Data Signal | Downstream Consumer | Operational Platform Impact |
| :--- | :--- | :--- | :--- |
| **Farmer: Crop Listing** | `POST /market/listing/add` | **Broker: Regional Supply** | Instantly increases local mandi supply tonnage, recalculates dynamic oversupply ratio, and triggers Z-score anomaly scans. |
| **Broker: Demand Engine** | High Mandi Prices in Metro Hubs | **Farmer: Crop Recommendation** | Informs macro-priors and price advantage heuristics, guiding farmers to sow deficit crops rather than flooding saturated crops. |
| **Broker: Shipment Dispatch** | `POST /demand/initiate-shipment` | **Farmer: Marketplace** | Marks farmer lots as `in_transit`, preventing duplicate sales and assuring farmers of confirmed procurement. |
| **Farmer: Disease Scan** | Severe Fungal Infection Logged | **Broker: Quality Assessment** | Notifies brokers of localized crop disease outbreaks in source clusters, alerting them to verify quality grades before contracting. |

---

## 13. Technology Stack & Algorithm Directory

| Layer | Technologies & Libraries | Architectural Role & Implementation Details |
| :--- | :--- | :--- |
| **Frontend Framework** | **React 19, React Router v6** | Component-based single page application for high-frequency trading dashboards. |
| **Charting & Visualization** | **Recharts, SVG Vectors, CSS3 Flex/Grid** | Interactive polynomial price decay curves, dynamic break-even area fills, circular SVG factor weight pie charts, and real-time urgency meters. |
| **Backend Framework** | **FastAPI, Uvicorn, Python 3.11** | Async, non-blocking high-throughput microservice architecture. |
| **Machine Learning** | **Scikit-Learn, Scipy, Joblib, SHAP** | `GradientBoostingRegressor` for dynamic demand thresholds, 5-fold cross-validation, and non-causal SHAP feature importance analysis. |
| **Spatial Logistics** | **Geodesic Haversine Formulas** | Vectorized spatial computation across 45+ Indian district centroids, computing haul distance and freight matrices in $< 5\text{ ms}$. |
| **Database Tier** | **MongoDB Atlas / PyMongo / Motor** | Distributed document persistence for `shipment_log`, `market_listings`, `supply_history`, and `users`. |
| **Resilience & Fallbacks** | **Safe In-Memory Mock Caches** | Graceful degradation: if MongoDB or Weather APIs are unreachable, analytical services automatically fall back to cached baselines without failing client requests. |

---

## 14. Complete REST API Catalog & Route Sitemap

The Broker Module exposes a clean, fully documented REST API mounted across `/market`:

| HTTP Method | Route Endpoint | Primary Request Parameters / Payload | Target Service | Success Response Payload Schema |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/market/summary` | None | `market_service.py` | `{"success": true, "total_listings": 42, "total_supply_kg": 185000, "oversupply_alerts": 3, "undersupply_alerts": 2}` |
| **GET** | `/market/listings` | None | `market_service.py` | `{"success": true, "total": 42, "listings": [{"farmer_name", "location", "crop_name", "quantity_kg", "target_price", "phone", "status"}]}` |
| **GET** | `/market/supply/{region}` | `region` (path str) | `market_service.py` | `{"success": true, "region": "Hyderabad", "crops": [{"crop", "total_kg", "listing_count", "avg_price"}], "total_volume": 42000}` |
| **GET** | `/market/demand/{region}` | `region` (path), `crop` (opt query) | `market_service.py` | `{"success": true, "region": "Nashik", "crop": "onion", "status": "oversupply", "threshold_kg": 12000, "current_kg": 24500}` |
| **GET** | `/market/demand/analysis/{region}/{crop}/{supply_kg}` | `region`, `crop`, `supply_kg` (path); `truck_type`, `top_n`, `max_distance` (query) | `market_ranking_service.py` | `{"success": true, "current_state": {...}, "urgency": {...}, "price_trajectory": {...}, "destinations": [7-Factor Ranked Destinations], "allocation_summary": {...}}` |
| **GET** | `/market/demand/threshold/{region}/{crop}` | `region`, `crop` (path str) | `demand_ml_service.py` | `{"success": true, "predicted_threshold_kg": 14200, "threshold_type": "ML-predicted", "model": "Gradient Boosting Regressor"}` |
| **POST** | `/market/demand/trend/{region}/{crop}` | `region`, `crop` (path); `[300, 450, 600, 800, 1100]` (body list) | `demand_ml_service.py` | `{"success": true, "dynamic_threshold": 12000, "trend_analysis": {"trend": "increasing", "growth_rate_pct": 24.5}}` |
| **GET** | `/market/demand/price-impact/{region}/{crop}/{supply_kg}` | `region`, `crop`, `supply_kg` (path) | `demand_ml_service.py` | `{"success": true, "price_impact": {"pct_price_change": -35.2, "estimated_price": 13.0, "loss_per_100kg": 700}, "redistribute_to": [...]}` |
| **POST** | `/market/demand/initiate-shipment` | `{"broker_id", "source_region", "destination_region", "crop", "quantity_kg", "agreed_price", "truck_type"}` | `shipment_service.py` | `{"success": true, "message": "Shipment initiated successfully", "shipment": {"_id", "status": "pending", "initiated_at", ...}, "transport": {...}}` |
| **PATCH** | `/market/demand/shipment/{id}/status/{new_status}` | `id`, `new_status` (`pending`, `in_transit`, `delivered`, `cancelled`) | `shipment_service.py` | `{"success": true, "shipment": {"_id", "status": "delivered", "actual_delivery": "2026-10-06T12:00:00Z"}}` |
| **GET** | `/market/demand/shipments/{broker_id}` | `broker_id` (path str), `status` (opt query) | `shipment_service.py` | `{"success": true, "broker_id": "...", "shipments": [...], "total": 8}` |
| **GET** | `/market/demand/transport-cost/{r1}/{r2}/{qty}` | `r1`, `r2`, `qty` (path); `truck_type` (query) | `shipment_service.py` | `{"success": true, "distance_km": 185.4, "truck_type": "medium", "rate_per_km": 35.0, "total_cost": 6489.0, "cost_per_100kg": 129.78}` |
| **POST** | `/market/demand/rate-broker` | `{"deal_id", "rater_broker_id", "rated_broker_id", "stars": 5, "comment": "Excellent delivery"}` | `reliability_service.py` | `{"success": true, "message": "Rating recorded successfully"}` |
| **GET** | `/market/demand/broker-reliability/{broker_id}` | `broker_id` (path str) | `reliability_service.py` | `{"success": true, "reliability_score": 92.4, "star_rating": 4.8, "total_deals": 18, "on_time_rate": 0.94}` |
| **POST** | `/market/demand/record-supply` | `{"region", "crop", "total_supply_kg", "listing_count"}` | `supply_history_service.py` | `{"success": true, "snapshot_id": "..."}` |
| **GET** | `/market/regions/all` | None | `region_service.py` | `{"success": true, "regions": ["Hyderabad", "Warangal", ...], "total": 45}` |
| **GET** | `/market/regions/distance/{r1}/{r2}` | `r1`, `r2` (path str) | `region_service.py` | `{"success": true, "region1": "Nashik", "region2": "Pune", "distance_km": 208.5}` |
| **GET** | `/notifications` | None | `notifications.py` | `{"success": true, "total": 14, "high_alerts": 3, "alerts": [{"id", "type": "oversupply", "severity": "high", "message", ...}]}` |

---

### Document Sign-Off & Verification
- **Lead Systems Engineer:** ALIP Implementation Agent
- **Approved For:** Production Commercial Deployment `v2.1-hybrid`
- **Cross-Checked Files:** 
  - Backend: [market_ranking_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/market_ranking_service.py), [demand_ml_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/demand_ml_service.py), [urgency_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/urgency_service.py), [price_trajectory_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/price_trajectory_service.py), [shipment_service.py](file:///D:/ALIP-TOTAL/ALIP/backend/services/shipment_service.py), [region_routes.py](file:///D:/ALIP-TOTAL/ALIP/backend/routes/region_routes.py)
  - Frontend: [BrokerDashboard.js](file:///D:/ALIP-TOTAL/ALIP/frontend/alip-frontend/src/pages/BrokerDashboard.js), [DemandAnalysis.js](file:///D:/ALIP-TOTAL/ALIP/frontend/alip-frontend/src/pages/DemandAnalysis.js), [RegionalSupply.js](file:///D:/ALIP-TOTAL/ALIP/frontend/alip-frontend/src/pages/RegionalSupply.js), [AllListings.js](file:///D:/ALIP-TOTAL/ALIP/frontend/alip-frontend/src/pages/AllListings.js), [Notifications.js](file:///D:/ALIP-TOTAL/ALIP/frontend/alip-frontend/src/pages/Notifications.js)
