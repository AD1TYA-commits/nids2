# NIDS 2.0 — SOC Threat Operations Demonstration Guide & Runbook

This guide contains everything needed to present and demonstrate the **Autonomous Network Intrusion Detection System (NIDS 2.0)**.

---

## 1. System Architecture

```text
       LIVE PACKETS (virbr0 / wlp4s0)           OFFLINE BENCHMARK DATASET
                     │                                      │
                     ▼                                      ▼
        RUST FLOW ENGINE (`network-analyzer`)      `demo_traffic.jsonl` / `live_flows.jsonl`
        - libpcap zero-copy capture                         │
        - Stateful TCP lifecycle assembly                   │
        - Welford online statistical moments                │
        - 77 raw flow features                              │
                     │                                      │
                     └──────────────────┬───────────────────┘
                                        ▼
                           FEATURE ADAPTER (`backend/adapter.py`)
                           - Schema validation & normalization
                           - CICIDS2017 semantic compatibility transform
                           - 13 engineered features (90 total)
                                        │
                                        ▼
                           DUAL-TIER INFERENCE ENGINE
                           ┌───────────────────────────────┐
                           │ LightGBM Supervised Classifier │ ──> Attack Category & Confidence
                           └───────────────────────────────┘
                           ┌───────────────────────────────┐
                           │ Autoencoder + DSP Anomaly Det.│ ──> Reconstruction Loss & Subspace Projections
                           └───────────────────────────────┘
                                        │
                                        ▼
                           DETERMINISTIC RISK ENGINE (`risk_engine.py`)
                           - Multi-evidence fusion (supervised + anomaly + DSP prototypes)
                           - Calibrated decision boundary (Threshold = 1.3728)
                           - Risk score computation (0 - 100)
                           - Policy action (ALLOW / MONITOR / BLOCK / QUARANTINE)
                                        │
                                        ▼
                           FASTAPI BACKEND (`http://localhost:8000`)
                           - REST APIs (`/api/status`, `/api/stats`, `/api/history`, `/api/mode`)
                           - Async WebSocket event bus (`/ws`)
                                        │
                                        ▼
                           REAL-TIME SOC DASHBOARD UI
                           - Dark-mode responsive operations console
                           - Real-time KPI counters & risk gauge
                           - Expandable flow inspection & audit trail
```

---

## 2. Quick Start / Running the System

### A. Access the SOC Dashboard
The backend is already running on **port 8000**:
- **Dashboard URL:** [http://localhost:8000](http://localhost:8000)
- **API Status:** [http://localhost:8000/api/status](http://localhost:8000/api/status)
- **Metrics Telemetry:** [http://localhost:8000/api/stats](http://localhost:8000/api/stats)

*(If restarting the backend from scratch):*
```bash
FLOW_REPLAY_MODE=live CAPTURE_INTERFACE=virbr0 python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 3. The Three Operational Modes

Switch between modes using the buttons in the top navigation bar or via the REST API:

### Mode 1: 🧪 DEMO / LAB REPLAY (`mode=demo`)
- **Recommended for primary presentation**: Feeds 1,000 ground-truth benchmark flows with known distributions:
  - **60% Benign / Normal** $\to$ Verified as `NORMAL`, Risk ~5.7, Action `ALLOW`
  - **15% PortScan** $\to$ Verified as `MALICIOUS`, Risk ~82.7, Action `BLOCK & ALERT`
  - **15% DoS & Brute Force** $\to$ Verified as `MALICIOUS`, Risk ~88.5, Action `BLOCK & ALERT`
  - **10% Zero-Day Infiltration** $\to$ Verified as `UNKNOWN ATTACK`, Risk ~87.2, Action `QUARANTINE & DISSECT`
- **API command:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/api/mode?mode=demo"
  ```

### Mode 2: 🟢 LIVE CAPTURE (`mode=live`)
- Captures actual network packets in real time over `virbr0` (or `wlp4s0` / `lo`).
- Rust engine reassembles TCP lifecycles and pipes JSONL into `/api/classify`.
- **API command:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/api/mode?mode=live"
  ```
- **Simulating live traffic during the demo:**
  ```bash
  # In a separate terminal:
  python3 scripts/simulate_vm_traffic.py
  ```

### Mode 3: 📁 CAPTURED REPLAY (`mode=captured`)
- Streams raw uncurated traffic from `live_flows.jsonl`.
- **API command:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/api/mode?mode=captured"
  ```

---

## 4. Key Talking Points for Evaluators

1. **True Dual-Tier Defense (Supervised + Unsupervised):**
   - Supervised models (LightGBM) excel at classifying known signatures (DoS, PortScan, Brute Force).
   - The deep Autoencoder with DSP (Deep Subspace Projection) isolates novel/zero-day attacks that bypass supervised classifiers by detecting out-of-distribution manifold deviations.

2. **Deterministic Risk Engine:**
   - Evaluates multi-source evidence: supervised class confidence, AE reconstruction deviation, and prototype novelty distance.
   - Outputs actionable risk scores (0–100) and strict triage actions (`ALLOW`, `MONITOR`, `BLOCK`, `QUARANTINE`) with transparent audit reasons.

3. **High-Performance Rust Flow Engine:**
   - Zero-copy packet capture, $O(1)$ Welford streaming accumulators, and stateful TCP handshake/teardown tracking.
   - Passes all 19 unit and integration tests.

4. **Production Semantic Compatibility:**
   - Resolved legacy CICFlowMeter Java tooling artifacts (e.g. unrecorded backward PSH flags, teardown FIN discrepancies) at the adapter boundary, ensuring normal TCP traffic conforms to the calibrated model manifold without requiring artificial threshold changes or model retraining.
