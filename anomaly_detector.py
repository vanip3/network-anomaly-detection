"""
anomaly_detector.py
-------------------
Wraps scikit-learn's IsolationForest in a domain-specific class that speaks
"network operations" rather than "machine learning".

Key design decisions:
  - Feature engineering:  We give the model context (hour-of-day, deviation
    from rolling mean) so it can distinguish "90% util at 3am" (anomalous)
    from "90% util at 2pm" (possibly normal). Raw utilisation alone misses this.
  - Per-link fitting:     Each link has its own traffic profile. Training one
    model per link avoids the model confusing "link_5 always runs hot" with
    an anomaly on link_1.
  - Human-readable output: explain_anomaly() translates numeric scores into
    plain English — critical for a NOC dashboard where the user is a network
    engineer, not a data scientist.

All tuneable values live in the constants block below so there are no magic
numbers buried in method bodies.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
N_ESTIMATORS: int = 200
# 200 trees instead of sklearn's default 100.
# More trees = more stable anomaly scores, at marginal runtime cost.
# Practical rule: double the default for any "production quality" IF model.

ROLLING_WINDOW: int = 4
# 4 samples × 15-minute polling = 1 hour of rolling context.
# This is the window used for rolling_mean and rolling_std features.
# Increase if your polling interval is shorter (e.g. use 12 for 5-min polls).

DEFAULT_CONTAMINATION: float = 0.05
# Prior belief: ~5% of readings are anomalous.
# Contamination only shifts the decision threshold; it does NOT change scores.

DEFAULT_RANDOM_STATE: int = 42
# Fixed seed for reproducibility — same data → same model → same results.
# Always set this in production for regression-testable behaviour.


# =============================================================================
# FEATURE COLUMNS
# Listed once here and referenced in both fit_predict() and get_anomaly_scores()
# to avoid copy-paste drift. If you add a feature, add it here only.
# =============================================================================
FEATURE_COLS: list[str] = [
    "utilisation_percent",   # raw measurement — the primary signal
    "hour_sin",              # cyclical hour encoding — sin component
    "hour_cos",              # cyclical hour encoding — cos component
    "rolling_mean",          # 1-hour rolling average — captures the trend
    "rolling_std",           # 1-hour rolling std dev — captures volatility
    "deviation_from_rolling",# current − rolling_mean — detects step changes
]


# =============================================================================
# EXPLANATION THRESHOLDS
# All numeric cutoffs used by explain_anomaly() are named here.
# This makes it easy to tune the explanation logic without hunting through code.
# =============================================================================

# Utilisation levels (percentage points)
UTIL_CRITICAL: float = 95.0    # near-saturation — link will drop/queue traffic
UTIL_HIGH: float = 85.0        # high-utilisation SLA breach territory
UTIL_ELEVATED: float = 75.0    # elevated off-hours — unusual but not critical
UTIL_NEAR_ZERO: float = 5.0    # near-zero — likely link failure or interface down
UTIL_LOW_BUSINESS: float = 15.0 # low during business hours — possible traffic loss

# Deviation from rolling mean (percentage points)
DEVIATION_LARGE: float = 25.0  # "sudden spike/drop" territory
DEVIATION_MODERATE: float = 15.0  # "sharp change" territory

# Rolling standard deviation (percentage points)
VOLATILITY_HIGH: float = 20.0  # high std dev = flapping / instability

# Off-hours elevated utilisation
UTIL_OFFHOURS_SUSPICIOUS: float = 60.0  # high at night = unusual

# Business hours window (UTC)
BUSINESS_HOURS_START: int = 7   # 07:00 UTC — start of European business day
BUSINESS_HOURS_END: int = 21    # 21:00 UTC — end of US East Coast business day


# =============================================================================
# SEVERITY THRESHOLDS (applied to normalised score ∈ [0, 1])
# =============================================================================
SEVERITY_HIGH_THRESHOLD: float = 0.75
SEVERITY_MEDIUM_THRESHOLD: float = 0.50
# Scores ≥ 0.75 → High (most isolated points — extreme anomalies)
# Scores ≥ 0.50 → Medium (clearly anomalous but less extreme)
# Scores  < 0.50 → Low (borderline — flagged by contamination threshold)


# =============================================================================
# DETECTOR CLASS
# =============================================================================

class NetworkAnomalyDetector:
    """
    Trains and applies an Isolation Forest model to WAN link utilisation data.

    Usage
    -----
    detector = NetworkAnomalyDetector(contamination=0.05)
    result_df = detector.fit_predict(df)
    """

    def __init__(
        self,
        contamination: float = DEFAULT_CONTAMINATION,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        """
        Parameters
        ----------
        contamination : float
            Expected proportion of anomalies (0.01–0.20). This is a prior
            belief — it shifts the decision threshold but does NOT change the
            underlying anomaly scores. Higher → more points flagged.
        random_state : int
            NumPy seed. Always fix this so the same data produces the same
            model — important for reproducible dashboards and CI testing.
        """
        self.contamination = contamination
        self.random_state = random_state

        # One fitted IsolationForest instance per link_id.
        # Per-link models ensure each circuit's "normal" is learned independently.
        self._models: dict = {}

        # Guards against calling get_anomaly_scores() before fit_predict()
        self._is_fitted: bool = False

        # Training metadata — populated during fit_predict(), one entry per link.
        # Exposed via get_training_summary() so the dashboard can show the user
        # that real model training happened, with concrete numbers.
        self._training_meta: list[dict] = []

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw utilisation readings into a richer feature matrix.

        Why not just use raw utilisation?
        Isolation Forest partitions feature space. Raw utilisation can't tell
        the difference between "70% at 3am" (suspicious) and "70% at 2pm"
        (normal peak). Adding temporal and rolling features gives the model
        the context it needs to make that distinction.

        All six features in FEATURE_COLS are computed here.
        """
        feat = df.copy()

        # Coerce timestamp to datetime — handles both native datetime and
        # strings (e.g. after a pd.read_csv / pd.read_json round-trip)
        feat["timestamp"] = pd.to_datetime(feat["timestamp"])

        hour = feat["timestamp"].dt.hour.astype(float)

        # Cyclical encoding: project hour onto a unit circle.
        # Without this, the model thinks hour=23 and hour=0 are 23 units apart
        # when they are actually 1 hour apart in the daily cycle.
        # With sin/cos encoding, both are represented as adjacent 2D points.
        feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Sort so that rolling windows are computed in time order per link
        feat = feat.sort_values(["link_id", "timestamp"])
        grp = feat.groupby("link_id")["utilisation_percent"]

        # ROLLING_WINDOW samples = 1 hour at 15-minute polling cadence.
        # min_periods=1 ensures the first few rows aren't dropped as NaN.
        feat["rolling_mean"] = grp.transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        feat["rolling_std"] = grp.transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1)
            .std()
            .fillna(0)  # std of a single value is NaN — replace with 0
        )

        # Deviation: how far is this reading from its own recent average?
        # Large positive → sudden spike; large negative → sudden drop.
        feat["deviation_from_rolling"] = (
            feat["utilisation_percent"] - feat["rolling_mean"]
        )

        return feat

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train one Isolation Forest per link and return the enriched DataFrame.

        The algorithm (per link):
          1. Engineer features from raw data
          2. Fit IsolationForest on all samples (unsupervised — no labels needed)
          3. Call decision_function() to get a continuous anomaly score
          4. Call predict() to get binary classification (+1 normal / -1 anomaly)
          5. Normalise scores to [0, 1] and generate plain-English explanations

        Returns
        -------
        Original DataFrame plus four new columns:
            is_anomaly       : bool
            anomaly_score    : float (raw IF score; more negative = more anomalous)
            score_normalised : float ∈ [0, 1]  (1.0 = most anomalous)
            explanation      : str
        """
        enriched = self._engineer_features(df)

        all_predictions: list = []
        all_scores: list = []

        for link_id, link_df in enriched.groupby("link_id"):
            X = link_df[FEATURE_COLS].values

            # ------------------------------------------------------------------
            # Isolation Forest — how it works:
            #
            # 1. Build N_ESTIMATORS random trees by repeatedly:
            #    a) Choosing a random feature
            #    b) Choosing a random split value between that feature's min/max
            #    c) Splitting the data and recursing on each half
            # 2. Stop when a point is isolated in its own leaf node.
            # 3. Count the number of splits needed to isolate each point.
            #    - Normal points are densely packed → many splits needed (long path)
            #    - Anomalous points are sparse/extreme → few splits needed (short path)
            # 4. The anomaly score is the normalised average path length across trees.
            #    - Score close to -1 (sklearn convention): very anomalous
            #    - Score close to +1: very normal
            # ------------------------------------------------------------------
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=N_ESTIMATORS,
                max_samples="auto",   # uses min(256, n_samples) — memory-efficient
                random_state=self.random_state,
                n_jobs=-1,            # parallelise tree building across all CPU cores
            )

            # fit() ingests ALL data for this link — unsupervised, no labels needed.
            # In production you'd fit on a clean historical baseline window, then
            # call predict() on new incoming data (streaming anomaly detection).
            t_start = time.perf_counter()
            model.fit(X)
            t_end = time.perf_counter()

            # predict() returns +1 (normal) or -1 (anomaly) using the contamination
            # threshold to decide where the boundary falls.
            preds = model.predict(X)

            # decision_function() gives the raw continuous score.
            # We keep this because the binary prediction discards information —
            # the dashboard wants to show "how anomalous" not just "anomalous/not".
            scores = model.decision_function(X)

            # Record training metadata for this link so the dashboard can prove
            # that real model training occurred (not just threshold rules).
            # model.offset_ is the learned decision threshold — the score value
            # below which a point is classified as anomalous. This is derived
            # from the contamination parameter and the actual data distribution.
            # Seeing it vary across links shows the model adapted to each circuit.
            self._training_meta.append({
                "link_id":          link_id,
                "samples_trained":  len(X),
                "trees_built":      len(model.estimators_),
                "features_used":    model.n_features_in_,
                "decision_threshold": round(float(model.offset_), 5),
                "training_time_ms": round((t_end - t_start) * 1000, 1),
            })

            self._models[link_id] = model
            all_predictions.extend(preds)
            all_scores.extend(scores)

        self._is_fitted = True

        # ------------------------------------------------------------------
        # Map results back onto the original DataFrame row order
        # ------------------------------------------------------------------
        result = df.copy()

        # groupby() iterates in the same order as sort_values(["link_id", "timestamp"])
        sort_idx = enriched.sort_values(["link_id", "timestamp"]).index
        result_sorted = result.loc[sort_idx].copy()

        result_sorted["is_anomaly"]   = np.array(all_predictions) == -1
        result_sorted["anomaly_score"] = np.array(all_scores)

        # Normalise to [0, 1] with inversion so 1.0 = most anomalous.
        # This is more intuitive for a "severity score" in the dashboard.
        raw = result_sorted["anomaly_score"].values
        score_min, score_max = raw.min(), raw.max()
        if score_max > score_min:
            result_sorted["score_normalised"] = 1.0 - (raw - score_min) / (
                score_max - score_min
            )
        else:
            result_sorted["score_normalised"] = 0.0

        # Generate explanations for every row (not just anomalies) so the
        # tooltip on every hover is populated.
        enriched_sorted = enriched.loc[sort_idx].copy()
        result_sorted["explanation"] = [
            self.explain_anomaly(row)
            for _, row in enriched_sorted.iterrows()
        ]

        # Restore the caller's original row order
        return result_sorted.loc[result.index]

    def get_anomaly_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        Return raw Isolation Forest decision scores for every row.

        More negative = more anomalous. Useful if you want to plot the score
        distribution or implement a custom threshold.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_predict() before get_anomaly_scores().")

        enriched = self._engineer_features(df)
        scores: list = []

        for link_id, link_df in enriched.groupby("link_id"):
            if link_id in self._models:
                X = link_df[FEATURE_COLS].values
                scores.extend(self._models[link_id].decision_function(X))
            else:
                # Link not seen during training — return neutral score
                scores.extend([0.0] * len(link_df))

        return pd.Series(scores, index=enriched.index)

    def get_training_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame describing what was trained during fit_predict().

        This is the evidence that REAL model training happened — not threshold
        rules. Each row is one trained IsolationForest model (one per link):

        Columns
        -------
        link_id            : machine-readable circuit identifier
        samples_trained    : number of 15-min polling rows the model learned from
        trees_built        : always N_ESTIMATORS (200) — confirms ensemble was built
        features_used      : number of engineered features fed to IF (should be 6)
        decision_threshold : the learned score boundary below which a point is
                             anomalous. This is derived from the contamination
                             parameter AND the actual data distribution, so it
                             differs per link. Seeing variation here proves the
                             model adapted to each circuit's individual profile.
        training_time_ms   : wall-clock time to fit this link's model in milliseconds

        Why does decision_threshold vary per link?
        IsolationForest computes threshold as the contamination-th percentile of
        the anomaly scores on the TRAINING data. Each link has a different score
        distribution, so each threshold will be slightly different — exactly what
        you'd expect from a model that learned from the data rather than applying
        a fixed rule.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_predict() before get_training_summary().")
        return pd.DataFrame(self._training_meta)

    def explain_anomaly(self, row: pd.Series) -> str:
        """
        Produce a plain-English reason for why a data point is anomalous.

        This is rule-based post-hoc reasoning, NOT SHAP attribution.
        The rules inspect the feature values and select the most salient
        narrative. In production you'd add SHAP on top, but this approach is:
          a) Transparent and readable by any engineer
          b) Fast — no extra model inference call
          c) Directly maps to real network failure modes

        Parameters
        ----------
        row : pd.Series
            A single row from the feature-engineered DataFrame.
            Expected columns: utilisation_percent, deviation_from_rolling,
            rolling_std, hour_sin, hour_cos.
        """
        util       = float(row.get("utilisation_percent", 50))
        deviation  = float(row.get("deviation_from_rolling", 0))
        volatility = float(row.get("rolling_std", 0))

        # Recover the hour from cyclical encoding.
        # arctan2(sin, cos) → angle in radians → convert to hours (0–23).
        if "hour_sin" in row and "hour_cos" in row:
            angle_deg = np.degrees(np.arctan2(row["hour_sin"], row["hour_cos"])) % 360
            hour = int(round(angle_deg * 24 / 360)) % 24
        else:
            hour = 12  # safe default if features are missing

        is_off_hours = hour < BUSINESS_HOURS_START or hour > BUSINESS_HOURS_END

        reasons: list[str] = []

        # --- Rule 1: High utilisation ---
        if util >= UTIL_CRITICAL:
            reasons.append(
                f"Critical utilisation ({util:.1f}%) -- link near saturation"
            )
        elif util >= UTIL_HIGH:
            reasons.append(
                f"High utilisation ({util:.1f}%) -- approaching capacity limit"
            )
        elif util >= UTIL_ELEVATED and is_off_hours:
            reasons.append(
                f"Elevated utilisation ({util:.1f}%) during off-hours "
                f"({hour:02d}:00 UTC)"
            )

        # --- Rule 2: Near-zero utilisation (possible link failure) ---
        if util < UTIL_NEAR_ZERO:
            reasons.append(
                f"Near-zero utilisation ({util:.1f}%) -- "
                "possible link failure or interface down"
            )
        elif util < UTIL_LOW_BUSINESS and not is_off_hours:
            reasons.append(
                f"Unexpectedly low utilisation ({util:.1f}%) during business hours "
                "-- traffic may have been rerouted"
            )

        # --- Rule 3: Large deviation from recent rolling mean ---
        if abs(deviation) > DEVIATION_LARGE:
            direction = "spike" if deviation > 0 else "drop"
            reasons.append(
                f"Sudden {direction} of {abs(deviation):.1f}% vs. recent average "
                "-- consistent with DDoS, congestion event, or link flap"
            )
        elif abs(deviation) > DEVIATION_MODERATE:
            direction = "increase" if deviation > 0 else "decrease"
            reasons.append(
                f"Sharp {direction} ({deviation:+.1f}%) from recent rolling mean"
            )

        # --- Rule 4: High volatility (flapping / instability) ---
        if volatility > VOLATILITY_HIGH:
            reasons.append(
                f"High signal volatility (std={volatility:.1f}%) -- "
                "possible BGP instability or physical-layer flapping"
            )

        # --- Rule 5: Off-hours elevation with no other trigger ---
        if is_off_hours and util > UTIL_OFFHOURS_SUSPICIOUS and not reasons:
            reasons.append(
                f"Abnormally high utilisation ({util:.1f}%) at {hour:02d}:00 UTC "
                "(off-hours -- expected traffic is low)"
            )

        # --- Fallback: statistically rare combination ---
        if not reasons:
            reasons.append(
                f"Unusual combination of metrics (util={util:.1f}%, "
                f"deviation={deviation:+.1f}%) -- flagged by Isolation Forest "
                "as statistically rare pattern"
            )

        return "; ".join(reasons)

    def get_severity(self, score_normalised: float) -> str:
        """Map normalised score [0,1] to a NOC severity tier."""
        if score_normalised >= SEVERITY_HIGH_THRESHOLD:
            return "High"
        if score_normalised >= SEVERITY_MEDIUM_THRESHOLD:
            return "Medium"
        return "Low"
