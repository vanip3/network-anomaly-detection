"""
data_generator.py
-----------------
Generates synthetic WAN link bandwidth utilisation data that mimics real
network traffic patterns. Used both to pre-generate sample_data/ and to
provide on-demand data inside the Streamlit app when no CSV is uploaded.

Design philosophy: The data must be realistic enough that a real NOC engineer
would recognise the patterns. We achieve this by combining:
  1. A sinusoidal base (24-hour business cycle)
  2. Gaussian noise (natural variation)
  3. Four hand-crafted anomaly archetypes (each maps to a real failure mode)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# =============================================================================
# LINK REGISTRY
# Human-readable circuit names give the dashboard a realistic "feel".
# Real WAN links are typically named by their two endpoints and technology.
# =============================================================================
LINK_NAMES: dict[str, str] = {
    "link_1": "LON-NYC-MPLS-01",
    "link_2": "LON-FRA-MPLS-02",
    "link_3": "NYC-SIN-IPLC-01",
    "link_4": "FRA-DXB-MPLS-03",
    "link_5": "SIN-SYD-IPLC-02",
    "link_6": "LON-TYO-MPLS-04",
    "link_7": "NYC-LAX-MPLS-05",
    "link_8": "FRA-JNB-IPLC-03",
}

# =============================================================================
# BASE SIGNAL PARAMETERS
# These control the shape of "normal" traffic on each link.
# Each link gets a slightly different profile to simulate heterogeneous circuits.
# =============================================================================
PEAK_HOUR_UTC: float = 14.0     # Hour at which traffic peaks (2pm UTC ≈ EU/US overlap)
BASE_UTIL_OFFSET: float = 30.0  # Minimum baseline utilisation (%) for link_1
BASE_UTIL_STEP: float = 5.0     # How much each successive link's baseline increases
AMPLITUDE_OFFSET: float = 20.0  # Base amplitude of the business-hours sine swing
AMPLITUDE_STEP: float = 2.0     # Per-link amplitude increment (busier links swing more)
NOISE_STD: float = 3.0          # Gaussian noise std dev (percentage points)
                                 # ±3σ ≈ ±9% — realistic for SNMP-polled interfaces

# =============================================================================
# ANOMALY INJECTION PARAMETERS
# Named as (min, max) tuples to make it clear they are random ranges.
# Each constant maps directly to a real failure mode — documented below.
# =============================================================================

# ---- Spike (DDoS, traffic burst, routing loop) ----
# A brief but extreme jump — often just 1–3 polling intervals wide.
SPIKE_COUNT_RANGE: tuple = (3, 8)          # how many spike events per link
SPIKE_DURATION_RANGE: tuple = (1, 4)       # duration in samples (1 sample = 15 min)
SPIKE_HEIGHT_RANGE: tuple = (35.0, 55.0)   # percentage points pushed above normal

# ---- Sustained high utilisation (capacity breach, backup path overload) ----
# A longer window where traffic stays well above the normal peak.
SUSTAINED_COUNT_RANGE: tuple = (2, 5)
SUSTAINED_DURATION_RANGE: tuple = (20, 48)  # samples ≈ 5–12 hours
SUSTAINED_BOOST_RANGE: tuple = (20.0, 35.0) # percentage point elevation

# ---- Sudden drop (link failure, cable cut, interface down) ----
# Utilisation collapses to near-zero. IF flags this because it's far outside
# the normal distribution even though the VALUE is low, not high.
DROP_COUNT_RANGE: tuple = (2, 5)
DROP_DURATION_RANGE: tuple = (1, 6)        # samples ≈ 15 min – 90 min outage
DROP_UTIL_RANGE: tuple = (0.0, 5.0)        # near-zero range when link is "down"

# ---- Flapping (BGP instability, STP loop, intermittent physical fault) ----
# Rapid oscillation between high and low every other sample.
# Characteristic of a routing protocol neighbour repeatedly going up/down.
FLAP_COUNT_RANGE: tuple = (1, 3)
FLAP_DURATION_RANGE: tuple = (12, 28)      # samples ≈ 3–7 hours of instability
FLAP_HIGH_RANGE: tuple = (85.0, 100.0)     # utilisation when traffic is on-link
FLAP_LOW_RANGE: tuple = (0.0, 15.0)        # utilisation when traffic has rerouted


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _business_hours_sine(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Produce a smooth 24-hour cosine wave that peaks at PEAK_HOUR_UTC.

    Why a cosine? Real aggregate bandwidth on a WAN link rises through the
    morning, peaks mid-afternoon (when Europe and North America overlap), and
    falls through the evening. A shifted cosine is the simplest function that
    captures that shape without over-fitting to any specific dataset.

    Output range is [0, 1]:
      - PEAK_HOUR_UTC (14:00) → 1.0  (maximum)
      - PEAK_HOUR_UTC - 12h  (02:00) → 0.0  (minimum)
    """
    # Convert timestamp to fractional hour (0.0 – 24.0)
    hours = timestamps.hour + timestamps.minute / 60.0

    # Shift the cosine so its peak (cos=1) falls at PEAK_HOUR_UTC.
    # Multiply by 2π/24 to complete exactly one cycle per 24 hours.
    radians = (hours - PEAK_HOUR_UTC) * (2 * np.pi / 24)

    # Rescale from [-1, 1] → [0, 1]
    return (np.cos(radians) + 1) / 2


# =============================================================================
# PUBLIC API
# =============================================================================

def generate_network_data(
    num_links: int = 5,
    days: int = 30,
    interval_minutes: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a DataFrame of synthetic WAN link utilisation readings.

    Parameters
    ----------
    num_links : int
        Number of WAN links to simulate (max = len(LINK_NAMES) = 8).
    days : int
        How many days of historical data to produce.
    interval_minutes : int
        Polling interval in minutes. 15 is standard for SNMP interface counters.
    random_seed : int
        NumPy random seed for reproducibility. Fixes the seed so the dashboard
        shows the same anomaly positions every time "Use sample data" is clicked.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp           : UTC datetime of the polling reading
        link_id             : machine-readable ID (link_1 … link_N)
        link_name           : human-readable circuit name
        utilisation_percent : 0–100 — percentage of link capacity in use
        anomaly_injected    : bool — ground truth label (True = we injected this)
    """
    np.random.seed(random_seed)

    # Clamp to available link names
    num_links = min(num_links, len(LINK_NAMES))

    # -------------------------------------------------------------------------
    # Build the time axis — evenly-spaced timestamps at interval_minutes cadence
    # -------------------------------------------------------------------------
    start_time = (
        datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        - timedelta(days=days)
    )

    # pd.date_range produces a DatetimeIndex — the spine all link data shares
    timestamps = pd.date_range(
        start=start_time,
        periods=int(days * 24 * 60 / interval_minutes),
        freq=f"{interval_minutes}min",
    )
    num_timesteps = len(timestamps)

    all_rows = []

    for link_idx in range(1, num_links + 1):
        link_id   = f"link_{link_idx}"
        link_name = LINK_NAMES[link_id]

        # ---------------------------------------------------------------------
        # STEP 1 — Base signal: sinusoidal business-hours cycle
        # Each link has a slightly different baseline and amplitude to simulate
        # circuits with different capacities and traffic mixes.
        # ---------------------------------------------------------------------
        base_util = BASE_UTIL_OFFSET + link_idx * BASE_UTIL_STEP
        amplitude = AMPLITUDE_OFFSET + link_idx * AMPLITUDE_STEP

        sine_wave = _business_hours_sine(timestamps)

        # Convert to a plain NumPy array immediately.
        # Pandas Series do NOT support in-place item assignment (series[i] = x),
        # which we need for the anomaly injection loop below.
        utilisation: np.ndarray = (
            base_util + amplitude * sine_wave
        ).values.astype(float)

        # ---------------------------------------------------------------------
        # STEP 2 — Gaussian noise: makes the signal look organic, not smooth
        # NOISE_STD=3 means ~95% of readings vary by at most ±6 percentage
        # points from the sine baseline — realistic for a production WAN link.
        # ---------------------------------------------------------------------
        utilisation += np.random.normal(loc=0.0, scale=NOISE_STD, size=num_timesteps)

        # Boolean mask tracking which indices we deliberately perturb
        anomaly_mask = np.zeros(num_timesteps, dtype=bool)

        # ---------------------------------------------------------------------
        # STEP 3 — Inject anomalies
        # We use randint/uniform + the named constants above so tuning the
        # anomaly characteristics only requires changing the constants.
        # ---------------------------------------------------------------------

        # ── 3a. Sudden spike (DDoS, routing loop, traffic burst) ─────────────
        # Brief, extreme utilisation peak. In a real NOC this triggers
        # high-utilisation alarms and often correlates with security events.
        for _ in range(np.random.randint(*SPIKE_COUNT_RANGE)):
            idx        = np.random.randint(10, num_timesteps - 10)
            duration   = np.random.randint(*SPIKE_DURATION_RANGE)
            height     = np.random.uniform(*SPIKE_HEIGHT_RANGE)
            end_idx    = min(idx + duration, num_timesteps)
            utilisation[idx:end_idx] += height
            anomaly_mask[idx:end_idx] = True

        # ── 3b. Sustained high utilisation (capacity breach) ─────────────────
        # Traffic stays elevated for hours — typically caused by a failed peer
        # link rerouting traffic onto a backup circuit, or organic growth
        # breaching the SLA utilisation threshold.
        for _ in range(np.random.randint(*SUSTAINED_COUNT_RANGE)):
            idx        = np.random.randint(10, num_timesteps - 50)
            duration   = np.random.randint(*SUSTAINED_DURATION_RANGE)
            boost      = np.random.uniform(*SUSTAINED_BOOST_RANGE)
            end_idx    = min(idx + duration, num_timesteps)
            utilisation[idx:end_idx] += boost
            anomaly_mask[idx:end_idx] = True

        # ── 3c. Sudden drop (link failure, cable cut, interface down) ─────────
        # Utilisation collapses to near-zero. IF flags this as anomalous
        # even though the value is LOW -- it is far outside the normal distribution.
        for _ in range(np.random.randint(*DROP_COUNT_RANGE)):
            idx      = np.random.randint(10, num_timesteps - 10)
            duration = np.random.randint(*DROP_DURATION_RANGE)
            end_idx  = min(idx + duration, num_timesteps)
            utilisation[idx:end_idx] = np.random.uniform(
                *DROP_UTIL_RANGE, size=end_idx - idx
            )
            anomaly_mask[idx:end_idx] = True

        # Step 3d: Flapping (BGP instability, STP loop, physical fault)
        # Rapid oscillation between high and low every other sample.
        for _ in range(np.random.randint(*FLAP_COUNT_RANGE)):
            idx      = np.random.randint(10, num_timesteps - 30)
            duration = np.random.randint(*FLAP_DURATION_RANGE)
            end_idx  = min(idx + duration, num_timesteps)
            for fi in range(idx, end_idx):
                if (fi - idx) % 2 == 0:
                    utilisation[fi] = np.random.uniform(*FLAP_HIGH_RANGE)
                else:
                    utilisation[fi] = np.random.uniform(*FLAP_LOW_RANGE)
            anomaly_mask[idx:end_idx] = True

        # Step 4: Physical clamp [0, 100]
        utilisation = np.clip(utilisation, 0.0, 100.0)

        # Step 5: Assemble per-link rows
        link_df = pd.DataFrame({
            "timestamp":           timestamps,
            "link_id":             link_id,
            "link_name":           link_name,
            "utilisation_percent": np.round(utilisation, 2),
            "anomaly_injected":    anomaly_mask,
        })
        all_rows.append(link_df)

    df = pd.concat(all_rows, ignore_index=True)
    df = df.sort_values(["timestamp", "link_id"]).reset_index(drop=True)
    return df


def save_sample_data(output_path: str = "sample_data/sample_network_data.csv") -> str:
    """Generate a 30-day, 5-link dataset and write it to CSV."""
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent_dir, exist_ok=True)
    df = generate_network_data(num_links=5, days=30, interval_minutes=15)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows --> {output_path}")
    return output_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_sample_data(os.path.join(script_dir, "sample_data", "sample_network_data.csv"))
