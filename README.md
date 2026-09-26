# AE-DSP Network Intrusion Detection System (NIDS)

A production-ready Network Intrusion Detection System built strictly **on the pretrained AE-DSP framework** from the official repository:
👉 **Official AE-DSP GitHub:** [https://github.com/Feiyu0Li/AE-DSP](https://github.com/Feiyu0Li/AE-DSP)

---

## 1. Architectural Overview

```
                      Pretrained AE-DSP
             (AutoEncoder + Semantic Prototypes)
                              │
                              ▼
                AE-DSP representation/features
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
        Component 1:                    Component 2:
         Classifier                   Anomaly Detector
     (Known Attack Types)          (Zero-Day / Novel Attacks)
              │                               │
              └───────────────┬───────────────┘
                              ▼
                         Risk Engine
                   (Deterministic Rules)
                              │
                              ▼
    ┌───────────────────────────────────────────────────┐
    │ NORMAL  │  SUSPICIOUS  │  MALICIOUS  │ UNKNOWN ATTACK │
    └───────────────────────────────────────────────────┘
```

The system strictly avoids unrelated models (no separate Transformers, VAEs, XGBoost, Deep SVDD, or ensemble backbones). Instead, **AE-DSP is the core pretrained foundation**, providing both representation learning and calibrated anomaly scoring.

---

## 2. Core Components

### 2.1 Pretrained AE-DSP Core Foundation (`nids_ae_dsp.core.ae_dsp_model`)
Implements the exact methodology from the official repository:
- **SHAP Contribution-Space Projection**: Network flows are mapped into SHAP attribution space where Global Top-$k$ features are selected by mean absolute attribution.
- **AutoEncoder (AE) Branch**: A PyTorch deep autoencoder trained exclusively on normal traffic in contribution space:
  - Input: $k$ features $\to$ hidden layers $h_1, h_2 \to$ latent bottleneck $z \in \mathbb{R}^{\text{latent\_dim}}$ ($8$ dimensions).
  - Decoder: $z \to h_2 \to h_1 \to$ reconstructed space $\hat{x}$.
  - Squared residuals: $\mathbf{r} = (x - \hat{x})^2$.
  - Uniform reconstruction score: $\text{AE\_score} = \text{mean}(\mathbf{r})$.
  - Robust scaling: $\text{AE}_z = (\text{AE\_score} - \mu_{\text{normal}}) / \sigma_{\text{normal}}$.
- **Deviation Semantic Prototype (DSP) Branch**:
  - $\Delta\text{SHAP} = (x_{\text{top}k} - \mu_{\text{normal}}) / (\sigma_{\text{normal}} + \epsilon)$.
  - $K$ attack prototypes ($K=12$) clustered via K-Means on known attack deviations.
  - Pairwise Euclidean distance to prototypes: $d_k = \|\Delta\text{SHAP} - C_k\|_2$.
  - Prototype similarities: $s_k = \exp(-d_k / \sigma)$.
  - Prototype similarity score: $\text{DSP\_score} = \exp(-\min_k d_k / \sigma) \to \text{DSP}_z$.
- **Calibrated AE-DSP Fusion**:
  - $\text{Fusion Score} = \alpha \cdot \text{AE}_z + \gamma \cdot \text{DSP}_z$ (e.g. $\alpha=0.8, \gamma=1.8$).
  - Decision threshold $\tau$ calibrated on validation normal traffic for a controlled target FPR (e.g. $0.045$).
- **Rich AE-DSP Representation Vector**:
  $$\mathbf{h} = [\mathbf{z} \mid \mathbf{s}_{\text{proto}} \mid \text{AE}_z \mid \text{DSP}_z \mid d_{\min} \mid \text{Score}_{\text{fusion}}]$$

### 2.2 Component 1: Classifier (`nids_ae_dsp.core.classifier`)
- Operates directly on the AE-DSP representation vector $\mathbf{h}$.
- Classifies traffic into known attack types (e.g. `Normal`, `DoS`, `Probe`, `R2L`, `U2R`, `PortScan`, etc.).
- Outputs:
  - `predicted_class`: string name of class.
  - `confidence`: maximum softmax probability ($\max_c P(c)$).
  - `probabilities`: complete distribution across all known classes.
  - `entropy`: predictive entropy ($H = -\sum p_c \ln p_c$).

### 2.3 Component 2: Anomaly Detector (`nids_ae_dsp.core.anomaly_detector`)
- Leverages the official AE-DSP anomaly scoring and prototype deviation metrics:
  - `anomaly_score`: AE-DSP fusion score.
  - `threshold`: $\tau$ calibrated on normal traffic.
  - `is_anomaly`: boolean decision ($\text{Score} \ge \tau$).
  - `ae_z`: AutoEncoder reconstruction deviation.
  - `dsp_z`: Prototype similarity deviation.
  - `nearest_proto_dist`: Distance to closest known attack prototype.
  - `is_novel_pattern`: Triggered when traffic exhibits significant reconstruction error ($\text{AE}_z > 1.5$) and its prototype distance breaches the 95th percentile envelope of known attacks.

### 2.4 Decision Layer: Risk Engine (`nids_ae_dsp.core.risk_engine`)
A transparent, deterministic security decision layer (**not an ML model**). It combines the classifier and anomaly detector outputs into four discrete security states:

| Decision | Anomaly Detector | Classifier Prediction & Confidence | DSP Prototype Evidence | Security Action |
|:---|:---|:---|:---|:---|
| **`NORMAL`** | $\text{Score} < \tau$ (No Anomaly) | `Normal` with confident probability | Conforms to normal SHAP manifold | `ALLOW` |
| **`MALICIOUS`** | Anomaly flagged OR High DSP match | Known attack class with confidence $\ge 55\%$ | Matches known prototype ($DSP_z > 0$) | `BLOCK & ALERT` |
| **`UNKNOWN ATTACK`** | $\text{Score} \ge \tau$ (Anomaly Flagged) | Predicts `Normal` (Evasion/Zero-Day) OR low confidence across known classes OR `is_novel_pattern == True` | Distance exceeds known prototype radius | `QUARANTINE & DISSECT (ZERO-DAY ALERT)` |
| **`SUSPICIOUS`** | Borderline score ($\tau - \delta \le \text{Score} < \tau$) | Low/moderate confidence or conflicting signals | Partial or borderline deviation | `FLAG FOR DEEP INSPECTION & MONITOR` |

---

## 3. Directory Structure

```text
micro/
├── AE-DSP/                             # Cloned official AE-DSP repository
├── nids_ae_dsp/                        # NIDS Python Package
│   ├── __init__.py                     # Public API exports
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ae_dsp_model.py             # Pretrained AE-DSP foundation & representation extractor
│   │   ├── classifier.py               # Component 1: Classifier for known classes
│   │   ├── anomaly_detector.py         # Component 2: Anomaly detector for zero-day/novel attacks
│   │   └── risk_engine.py              # Component 3: Deterministic Risk Engine
│   ├── pipeline.py                     # Unified AEDSP_NIDS orchestrator
│   ├── bundle_io.py                    # Official AE-DSP bundle loader & saver
│   └── trainer.py                      # Training and initialization utilities
├── examples/
│   ├── demo_live_inspection.py         # Live traffic inspection demo with all 4 decisions
│   └── train_and_eval_demo.py          # Leave-one-attack-out held-out zero-day evaluation
├── tests/
│   └── test_nids_architecture.py       # Full unit and integration test suite
└── setup.py                            # Package configuration
```

---

## 4. Quick Start

### 4.1 Running the Live Traffic Inspection Demo
Demonstrates live traffic flows passing through Pretrained AE-DSP $\to$ Representation $\to$ [Classifier, Anomaly Detector] $\to$ Risk Engine:

```bash
python examples/demo_live_inspection.py
```

Sample output:
```text
>>> INCOMING: Flow #5: ZERO-DAY Polymorphic Infiltration (Novel Attack)
  [1] Pretrained AE-DSP Representation:
      - Latent Embedding z   : [-14.744  -0.525 -12.847  -2.383] ... (dim=8)
      - AE Reconstruction res: AE_score=57.5220 | AE_z=179.828
      - Nearest Attack Proto : #2 (Distance=29.571)
      - DSP Similarity Score : DSP_z=-5.489
      - AE-DSP Fusion Score  : 133.9828 (Threshold=3.0447)
  [2] Component 1: Classifier (Known Classes):
      - Predicted Class      : DoS
      - Confidence           : 99.96%
  [3] Component 2: Anomaly Detector (Zero-Day / Novel):
      - Anomaly Status       : ANOMALY DETECTED
      - Novelty / Zero-day   : OUT-OF-DISTRIBUTION NOVEL PATTERN
      - Novelty Score        : 868.234
  [4] Decision Layer: Risk Engine:
      =======================================================
      FINAL SECURITY DECISION: >>> [ UNKNOWN ATTACK ] <<<
      Risk Score             : 100.0 / 100
      Security Action        : QUARANTINE & DISSECT (OUT-OF-DISTRIBUTION ATTACK)
      Audit Explanation      : Novel attack pattern detected. AE-DSP manifold deviation is high (AE_z=179.83), and distance to nearest attack prototype #2 (29.57) exceeds known prototype boundaries.
      =======================================================
```

### 4.2 Running the Held-Out Zero-Day Benchmark Evaluation
Evaluates the framework on a leave-one-attack-type-out protocol where an attack category (`Infiltration`) is completely held out from training:

```bash
python examples/train_and_eval_demo.py
```

Results:
- **Normal Traffic**: $94.67\%$ clean verified as `NORMAL` (FPR $4.33\%$ within target $4.5\%$ threshold envelope).
- **Known Attacks (DoS & Probe)**: $100.00\%$ correctly intercepted as `MALICIOUS`.
- **Held-Out Zero-Day Attacks**: $100.00\%$ successfully isolated as `UNKNOWN ATTACK`.

### 4.3 Running the Test Suite
Executes the comprehensive unit and integration test suite:

```bash
python -m unittest tests/test_nids_architecture.py
```

---

## 5. Using Pretrained AE-DSP Bundles

### Loading an official bundle from `AE-DSP/results`:
```python
from nids_ae_dsp import AEDSP_NIDS

# Load official bundle saved by AE-DSP/experiments/nsl_kdd/run_p6_final_zeroday_sensitive_system.py
pipeline = AEDSP_NIDS.from_pretrained_bundle(
    bundle_dir="path/to/official/model_bundle_seed42",
    class_names=["Normal", "DoS", "Probe", "R2L", "U2R"],
)

# Run complete inference
decisions = pipeline.predict(new_traffic_data)
for d in decisions:
    print(f"Decision: {d.decision.value} | Risk: {d.risk_score} | Action: {d.action}")
```

### Complete Pipeline Serialization:
```python
# Save pipeline (AE-DSP core + Classifier + Anomaly Detector + Risk Engine)
pipeline.save("saved_nids_model")

# Reload anywhere
reloaded = AEDSP_NIDS.load("saved_nids_model")
```
