# Network Anomaly Detection Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Problem Statement

Network operations teams react to incidents **after** they occur. A link silently saturates at 95% for four hours before a user calls the helpdesk. A WAN circuit flaps repeatedly — dropping BGP sessions — while the NOC watches a different screen.

This tool uses unsupervised machine learning to **proactively detect anomalies** in WAN link utilisation data — identifying congestion events, capacity breaches, link failures, and routing instability before they cause user-facing impact.

No labelled training data required. No static thresholds to maintain. The model learns what "normal" looks like for each individual circuit.

---

## Demo
<img width="1901" height="971" alt="ss1" src="https://github.com/user-attachments/assets/99d8e71a-bb8f-4a46-8eab-06dc49d2d486" />
<img width="1872" height="897" alt="ss2" src="https://github.com/user-attachments/assets/70fdfb3b-9397-4820-8375-9ee9d5d928d2" />
<img width="1866" height="845" alt="ss3" src="https://github.com/user-attachments/assets/0cf967a5-b9fa-4322-a3f4-5592d9675f9a" />
<img width="1892" height="930" alt="ss4" src="https://github.com/user-attachments/assets/f95157e2-f3da-49f8-8f6d-6124e7b1700d" />




> *The red markers show anomalies detected by Isolation Forest. Hover over any marker to see a plain-English explanation of why the model flagged it.*

---

## Features

- **Isolation Forest anomaly detection** — unsupervised ML; no labelled examples needed
- **Per-link model training** — each WAN circuit's "normal" is learned independently
- **Feature engineering** — hour-of-day context, rolling mean deviation, volatility signals
- **Interactive Plotly dashboard** — zoom, pan, hover tooltips with plain-English explanations
- **Four anomaly archetypes** — spike, sustained high, link drop, flapping
- **Configurable contamination** — tune sensitivity via sidebar slider
- **Upload your own CSV** — drop in real SNMP poll data with the right column names
- **Ground-truth evaluation** — precision, recall, F1 on synthetic data (for learning/demo)
- **Export anomalies** — one-click CSV download for ticketing system integration

---

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| UI | Streamlit 1.35+ | Reactive web dashboard |
| ML | scikit-learn 1.5+ | IsolationForest implementation |
| Data | Pandas 2.2+ | DataFrame manipulation |
| Data | NumPy 1.26+ | Synthetic signal generation |
| Charts | Plotly 5.22+ | Interactive visualisations |

---

## Project Structure

```
network-anomaly-detection/
├── app.py                          # Streamlit application — UI layer only
├── anomaly_detector.py             # ML model wrapper — IsolationForest + explanations
├── data_generator.py               # Synthetic WAN data with realistic anomaly injection
├── requirements.txt                # Pinned dependencies
├── README.md                       # This file
└── sample_data/
    └── sample_network_data.csv     # Pre-generated 30-day, 5-link dataset (14,400 rows)
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/network-anomaly-detection.git
cd network-anomaly-detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

To regenerate the sample data:
```bash
python data_generator.py
```

---

## How It Works

### 1. Data Generation (`data_generator.py`)

Synthetic WAN utilisation is built from three layers:

```
utilisation = base_load + amplitude × sin(hour) + gaussian_noise
```

**Base signal:** Each link has a different baseline utilisation (30–55%) and amplitude to simulate different circuit capacities.

**Sinusoidal cycle:** The sine wave peaks at 14:00 UTC and troughs at 02:00 UTC, matching a typical enterprise business-hours traffic pattern.

**Injected anomalies:** Four archetypes are randomly placed across the 30-day window:

| Anomaly Type | Real-World Cause | Signal Signature |
|---|---|---|
| Sudden spike | DDoS, routing loop, traffic burst | 1–3 samples at +35–55% above normal |
| Sustained high | Capacity breach, peer link failure | 20–48 samples (5–12 hours) elevated |
| Sudden drop | Link failure, cable cut, interface down | 1–6 samples near 0% |
| Flapping | BGP instability, STP loop, physical fault | Rapid oscillation between ~95% and ~5% |

### 2. Anomaly Detection (`anomaly_detector.py`)

**Algorithm: Isolation Forest**

Isolation Forest works by building random binary decision trees and measuring how quickly each point can be isolated from the rest:

- **Normal points** are densely packed and require many splits to isolate — long path length.
- **Anomalous points** are rare or extreme and can be isolated in very few splits — short path length.

The anomaly score is the normalised average path length across all trees. A more negative raw score = more anomalous.

**Feature Engineering**

Raw utilisation alone ignores context. We engineer six features per sample:

```python
features = [
    utilisation_percent,      # raw measurement
    hour_sin,                 # cyclical hour encoding (sin component)
    hour_cos,                 # cyclical hour encoding (cos component)
    rolling_mean,             # 1-hour rolling average
    rolling_std,              # 1-hour rolling volatility
    deviation_from_rolling,   # current - rolling_mean (detects step changes)
]
```

**Cyclical encoding** is important: if we used raw `hour` (0–23), the model would think midnight (23) and 1am (1) are 22 units apart when they're actually adjacent. Projecting onto `sin`/`cos` fixes this.

**Per-link models** ensure that a link that normally runs at 70% isn't penalised relative to one that runs at 30%.

### 3. Explanation Engine

The `explain_anomaly()` method applies rule-based post-hoc reasoning to produce NOC-friendly text:

- High utilisation during off-hours → flags the contrast
- Near-zero during business hours → suggests link failure
- Large deviation from rolling mean → characterises as spike or drop
- High rolling standard deviation → identifies flapping pattern

---

## Using Your Own Data

Upload a CSV via the sidebar. Required columns:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | ISO 8601 format (e.g. `2024-01-15 14:30:00`) |
| `link_id` | string | Machine-readable circuit ID |
| `link_name` | string | Human-readable circuit name |
| `utilisation_percent` | float | 0–100 |

Optional: `anomaly_injected` (bool) — enables the ground-truth evaluation section.

**Generating from SNMP:** If your NMS (e.g. LibreNMS, Grafana + InfluxDB) can export CSV, reshape the data to match the schema above. The `link_id` can be the SNMP OID or interface index.

---

## Configuration

| Sidebar Control | Effect |
|---|---|
| Contamination threshold | Expected anomaly rate; increase to flag more, decrease for fewer false positives |
| Link multiselect | Filter the dashboard to specific circuits |
| Date range | Zoom into a time window |

---

## Model Performance (on synthetic data)

With default settings (`contamination=0.05`, 30 days, 5 links):

| Metric | Typical Value |
|---|---|
| Precision | 0.70–0.85 |
| Recall | 0.75–0.90 |
| F1 Score | 0.72–0.87 |

Performance varies because anomaly injection is randomised. The model performs best on spikes and drops (extreme values), and slightly less well on sustained anomalies that overlap with legitimate peak-hours traffic.

---

## Extending the Project

- **Replace Isolation Forest** with LSTM Autoencoder for sequence-aware detection
- **Add SHAP values** for model-derived (rather than rule-based) explanations
- **Connect to real data** via InfluxDB, Prometheus, or SNMP polling
- **Add alerting** — POST to a webhook or PagerDuty when a new anomaly is detected
- **Containerise** — a `Dockerfile` is a natural next addition for deployment

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

Built as a portfolio project demonstrating AI/ML skills for network infrastructure engineering roles.
Concepts demonstrated: unsupervised learning, feature engineering, time-series analysis, production-quality Python architecture, interactive data visualisation.
