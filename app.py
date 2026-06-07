"""
app.py
------
Main Streamlit application — the UI layer for the Network Anomaly Detection
Dashboard. This file's only job is presentation and user interaction; all
ML logic lives in anomaly_detector.py and all data logic in data_generator.py.

Run with:
    streamlit run app.py

Architecture note:
    Streamlit re-executes this entire script on every user interaction.
    We use @st.cache_data to avoid re-running expensive operations (IF training,
    data generation) on every slider drag. The cache key is the function's
    arguments, so changing contamination or date range triggers re-computation.

Sidebar data-flow contract (important):
    render_sidebar() resolves the ACTIVE data source (upload or sample) FIRST,
    then builds all filter controls (link multiselect, date picker) from that
    active source. This ensures filters always reflect the data being shown —
    not the sample data defaults. See render_sidebar() for the full explanation.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import io

from data_generator import generate_network_data
from anomaly_detector import NetworkAnomalyDetector, N_ESTIMATORS, DEFAULT_RANDOM_STATE


# =============================================================================
# PAGE CONFIGURATION
# Must be the FIRST Streamlit call — Streamlit enforces this ordering.
# =============================================================================
st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONSTANTS
# All hardcoded values live here. Changing a threshold means editing one line.
# =============================================================================

# --- CSV upload validation ---
REQUIRED_COLUMNS: set[str] = {
    "timestamp", "link_id", "link_name", "utilisation_percent"
}
# The anomaly_injected column is optional — only present in sample data.
OPTIONAL_COLUMNS: set[str] = {"anomaly_injected"}

# --- Model defaults ---
CONTAMINATION_DEFAULT: float = 0.05   # 5% expected anomaly rate
CONTAMINATION_MIN: float = 0.01
CONTAMINATION_MAX: float = 0.20
CONTAMINATION_STEP: float = 0.01

# --- Date filter ---
DEFAULT_DATE_LOOKBACK_DAYS: int = 7   # initial date range shows last 7 days

# --- SLA reference lines on the time-series chart ---
SLA_WARN_PCT: int = 80     # orange dashed line — common SLA warning threshold
SLA_CRIT_PCT: int = 95     # red dashed line — critical / near-saturation

# --- Chart dimensions (pixels) ---
TIMESERIES_HEIGHT: int = 480
INSIGHT_CHART_HEIGHT: int = 360
ANOMALY_TABLE_HEIGHT: int = 350

# --- Colour palette ---
# Set2 is colourblind-friendly and distinguishes up to 8 series clearly.
LINK_COLOURS = px.colors.qualitative.Set2
ANOMALY_COLOUR: str = "#FF4444"    # red — anomaly markers and histogram bars
NORMAL_COLOUR: str = "#4FC3F7"     # light blue — normal data histogram bars
HIGH_SEV_COLOUR: str = "#FF4444"   # red  — High severity rows
MED_SEV_COLOUR: str = "#FF9800"    # amber — Medium severity rows


# =============================================================================
# CSV VALIDATION
# =============================================================================

def _validate_uploaded_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that an uploaded DataFrame has the expected shape and content.

    Returns
    -------
    (is_valid : bool, error_message : str)
    If is_valid is True, error_message is "".
    """
    # 1. Empty file
    if df.empty:
        return False, "The uploaded file is empty — no rows found."

    # 2. Required columns present
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sorted_missing = sorted(missing)
        sorted_present = sorted(df.columns.tolist())
        return False, (
            f"Missing required column(s): **{', '.join(sorted_missing)}**. "
            f"Columns found: {', '.join(sorted_present)}. "
            f"Required: {', '.join(sorted(REQUIRED_COLUMNS))}."
        )

    # 3. Timestamp parseable
    try:
        pd.to_datetime(df["timestamp"])
    except Exception as exc:
        return False, (
            f"Could not parse 'timestamp' column as datetime: {exc}. "
            "Use ISO 8601 format, e.g. `2024-01-15 14:30:00`."
        )

    # 4. utilisation_percent is numeric
    if not pd.api.types.is_numeric_dtype(df["utilisation_percent"]):
        return False, (
            "'utilisation_percent' must be a numeric column (0–100). "
            f"Got dtype: {df['utilisation_percent'].dtype}."
        )

    # 5. utilisation_percent has at least some valid (non-NaN) values
    if df["utilisation_percent"].isna().all():
        return False, "'utilisation_percent' column contains only NaN values."

    # 6. Range warning (not a failure — we clamp silently, but tell the user)
    out_of_range = (~df["utilisation_percent"].between(0, 100)).sum()
    if out_of_range > 0:
        # Return valid=True but include a warning hint in the message field.
        # The caller displays this as st.warning, not st.error.
        return True, (
            f"⚠️  {out_of_range:,} row(s) have utilisation_percent outside [0, 100]. "
            "They will be clamped before modelling."
        )

    return True, ""


# =============================================================================
# CACHED DATA FUNCTIONS
# @st.cache_data memoises by argument hash — same inputs = instant return.
# =============================================================================

@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    """
    Load the pre-generated sample CSV if it exists, else generate fresh data.

    Cached so page refreshes don't regenerate 14,000+ rows unnecessarily.
    The cache is keyed on the function arguments — since there are none, this
    runs at most once per Streamlit process lifetime.
    """
    csv_path = os.path.join(
        os.path.dirname(__file__), "sample_data", "sample_network_data.csv"
    )
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Fallback: generate on the fly (e.g. first run before data_generator.py)
    return generate_network_data(num_links=5, days=30, interval_minutes=15)


@st.cache_data(show_spinner=False)
def run_detection(
    data_key: str,       # "sample" or "uploaded" — part of the cache key
    contamination: float,
    serialised_df: str,  # DataFrame as a JSON string — hashable cache key
) -> pd.DataFrame:
    """
    Run Isolation Forest and return the enriched DataFrame.

    Why pass serialised_df as a string?
    st.cache_data hashes every argument. DataFrames are mutable and unhashable,
    so we serialise to JSON first. orient='records' + date_format='iso' ensures
    timestamps survive the round-trip without precision loss.

    Why keep data_key as a separate argument?
    It forces a cache miss when the user switches between sample and uploaded
    data, even if (by coincidence) the JSON strings happened to be identical.
    """
    df = pd.read_json(io.StringIO(serialised_df), orient="records")
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # restore datetime dtype

    detector = NetworkAnomalyDetector(contamination=contamination)
    result_df = detector.fit_predict(df)

    # Return both the enriched results AND the training summary so the dashboard
    # can show concrete evidence of what the model learned (samples, trees,
    # per-link decision thresholds, training time).
    training_summary = detector.get_training_summary()
    return result_df, training_summary


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar(base_df: pd.DataFrame) -> tuple:
    """
    Render all sidebar controls and return (filtered_df, contamination, data_source).

    DATA-FLOW CONTRACT
    ------------------
    The sidebar renders in this strict order:
      1. Contamination slider  (no data dependency)
      2. Data source controls  (file uploader + "use sample" button)
      3. Resolve active_df     (upload if valid, else sample)
      4. Filter controls       (link multiselect + date picker) built from active_df

    Step 4 MUST come after step 3. If we built filters from base_df (sample)
    before checking the upload, a CSV with different links or date ranges would
    be silently filtered by the wrong options — a hard-to-spot logic bug.
    """
    st.sidebar.title("⚙️ Controls")
    st.sidebar.markdown("---")

    # ── 1. Model settings ─────────────────────────────────────────────────────
    st.sidebar.subheader("Model Settings")
    contamination = st.sidebar.slider(
        "Contamination threshold",
        min_value=CONTAMINATION_MIN,
        max_value=CONTAMINATION_MAX,
        value=CONTAMINATION_DEFAULT,
        step=CONTAMINATION_STEP,
        help=(
            "Your prior belief about the fraction of anomalous readings. "
            f"Default {CONTAMINATION_DEFAULT:.0%} = 'I expect 1 in 20 readings to be anomalous.' "
            "Higher → more points flagged; lower → more conservative."
        ),
    )

    st.sidebar.markdown("---")

    # ── 2. Data source controls ───────────────────────────────────────────────
    st.sidebar.subheader("Data Source")

    uploaded_file = st.sidebar.file_uploader(
        "Upload your own CSV",
        type=["csv"],
        help=(
            f"Required columns: {', '.join(sorted(REQUIRED_COLUMNS))}. "
            "Optional: anomaly_injected (bool) enables ground-truth evaluation."
        ),
    )
    use_sample = st.sidebar.button("↩ Use sample data", use_container_width=True)

    # ── 3. Resolve active dataset ─────────────────────────────────────────────
    # Default to sample data; override if a valid upload is present.
    active_df = base_df
    data_source = "sample"

    if uploaded_file is not None and not use_sample:
        try:
            raw_upload = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.sidebar.error(f"Could not read CSV file: {exc}")
            raw_upload = None

        if raw_upload is not None:
            is_valid, message = _validate_uploaded_csv(raw_upload)
            if is_valid:
                raw_upload["timestamp"] = pd.to_datetime(raw_upload["timestamp"])
                # Clamp utilisation silently (validation already warned the user)
                raw_upload["utilisation_percent"] = raw_upload[
                    "utilisation_percent"
                ].clip(0, 100)
                active_df = raw_upload
                data_source = "uploaded"
                if message:
                    # message is non-empty only for the out-of-range warning case
                    st.sidebar.warning(message)
                else:
                    st.sidebar.success(
                        f"Loaded {len(active_df):,} rows "
                        f"across {active_df['link_name'].nunique()} link(s)."
                    )
            else:
                # Show the specific validation error so the user knows how to fix it
                st.sidebar.error(message)
                st.sidebar.info("Falling back to sample data.")

    # ── 4. Filter controls — built from ACTIVE dataset ────────────────────────
    # These MUST be computed after active_df is resolved (see docstring above).
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Filters")

    available_links = sorted(active_df["link_name"].unique().tolist())
    selected_links = st.sidebar.multiselect(
        "Select links to display",
        options=available_links,
        default=available_links,
        help="Filter the dashboard to specific WAN circuits.",
    )

    min_date = active_df["timestamp"].min().date()
    max_date = active_df["timestamp"].max().date()
    default_start = max(min_date, max_date - timedelta(days=DEFAULT_DATE_LOOKBACK_DAYS))

    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Zoom into a specific time window.",
    )

    # ── 5. Apply filters ──────────────────────────────────────────────────────
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_dt = pd.Timestamp(date_range[0])
        end_dt   = pd.Timestamp(date_range[1]) + timedelta(days=1)
        active_df = active_df[
            (active_df["timestamp"] >= start_dt) & (active_df["timestamp"] < end_dt)
        ]

    if selected_links:
        active_df = active_df[active_df["link_name"].isin(selected_links)]

    return active_df, contamination, data_source


# =============================================================================
# SECTION 0 — Model Training (shown BEFORE anomaly results)
# =============================================================================

# Feature descriptions — shown in the training section so users understand
# what the model was actually fed. This answers "where is the AI?" concretely.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "utilisation_percent":    "Raw bandwidth utilisation (0–100%). The primary signal.",
    "hour_sin":               "Cyclical hour encoding — sin component. Lets the model understand time-of-day without treating midnight and 1am as far apart.",
    "hour_cos":               "Cyclical hour encoding — cos component. Together with hour_sin, encodes the full 24-hour business cycle.",
    "rolling_mean":           "1-hour rolling average. Captures the recent trend so the model knows what 'normal' looks like right now.",
    "rolling_std":            "1-hour rolling standard deviation. High values = flapping or instability.",
    "deviation_from_rolling": "Current value minus rolling mean. Large positive = spike; large negative = drop.",
}


def render_model_training(training_summary: pd.DataFrame, contamination: float) -> None:
    """
    Show concrete evidence that ML model training happened.

    This section answers the question: "Where is the AI?"

    Isolation Forest is an unsupervised machine learning algorithm. Unlike a
    threshold rule, it LEARNS the distribution of normal traffic from data.
    The per-link decision thresholds below prove this — each link's threshold
    is computed from its own data, not set manually.
    """
    st.subheader("🤖 Model Training")
    st.caption(
        "Isolation Forest is an unsupervised ML algorithm — it learns what "
        "'normal' looks like from the data itself. No labelled examples needed. "
        "One model was trained per WAN link so each circuit's unique traffic "
        "profile is learned independently."
    )

    # ── KPI row: total training numbers ──────────────────────────────────────
    total_samples  = int(training_summary["samples_trained"].sum())
    total_trees    = int(training_summary["trees_built"].sum())
    total_time_ms  = float(training_summary["training_time_ms"].sum())
    num_models     = len(training_summary)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models trained",      str(num_models),
              help="One IsolationForest per WAN link")
    c2.metric("Total samples learned", f"{total_samples:,}",
              help="15-minute polling rows ingested across all links")
    c3.metric("Decision trees built",  f"{total_trees:,}",
              help=f"{N_ESTIMATORS} trees × {num_models} link models")
    c4.metric("Total training time",   f"{total_time_ms:.0f} ms",
              help="Wall-clock time to fit all models")

    # ── Per-link training table ───────────────────────────────────────────────
    # The key evidence: decision_threshold differs per link.
    # This PROVES the model adapted to each circuit — a static rule would
    # show the same threshold for every link.
    st.markdown("**Per-Link Training Results**")
    st.caption(
        "The **Decision Threshold** column is the most important: it is the "
        "anomaly score boundary the model LEARNED from each link's data. "
        "It varies per link because each circuit has a different traffic "
        "profile — proof that this is learned ML, not a hardcoded rule."
    )

    # Merge with result_df link names if available (link_id → link_name lookup)
    display = training_summary.copy()
    display = display.rename(columns={
        "link_id":           "Link ID",
        "samples_trained":   "Samples Trained",
        "trees_built":       "Trees Built",
        "features_used":     "Features Used",
        "decision_threshold":"Decision Threshold (learned)",
        "training_time_ms":  "Training Time (ms)",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Features used ─────────────────────────────────────────────────────────
    with st.expander("🔍 What did the model learn from? (Feature descriptions)"):
        st.markdown(
            "Isolation Forest was trained on **6 engineered features** per "
            "polling sample — not just raw utilisation. Adding temporal context "
            "means the model can flag '90% util at 3am' as anomalous while "
            "accepting '90% util at 2pm' as normal peak traffic.\n"
        )
        for feat, desc in FEATURE_DESCRIPTIONS.items():
            st.markdown(f"- **`{feat}`** — {desc}")

    # ── Model parameters used ─────────────────────────────────────────────────
    with st.expander("⚙️ Model hyperparameters"):
        params = {
            "Algorithm":         "IsolationForest (sklearn.ensemble)",
            "n_estimators":      f"{N_ESTIMATORS} trees per model",
            "contamination":     f"{contamination:.0%} (set via sidebar slider)",
            "max_samples":       "auto (min(256, n_samples))",
            "random_state":      str(DEFAULT_RANDOM_STATE),
            "Training strategy": "One model per WAN link (per-link fit)",
            "Features":          f"{len(FEATURE_DESCRIPTIONS)} engineered features",
            "Labels required":   "None — fully unsupervised learning",
        }
        for k, v in params.items():
            st.markdown(f"- **{k}**: {v}")


# =============================================================================
# SECTION 1 — Summary Metrics
# =============================================================================

def render_summary_metrics(result_df: pd.DataFrame) -> None:
    """
    Four KPI tiles at the top of the dashboard.
    st.metric() renders a bold number with an optional delta indicator.
    """
    st.subheader("📊 Summary")

    total_points  = len(result_df)
    anomaly_df    = result_df[result_df["is_anomaly"]]
    anomaly_count = len(anomaly_df)
    anomaly_pct   = (anomaly_count / total_points * 100) if total_points > 0 else 0.0
    links_with_anomalies = anomaly_df["link_name"].nunique() if anomaly_count > 0 else 0

    most_recent_str = (
        anomaly_df["timestamp"].max().strftime("%Y-%m-%d %H:%M UTC")
        if anomaly_count > 0
        else "None detected"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="📡 Data points analysed", value=f"{total_points:,}")

    with col2:
        st.metric(
            label="🚨 Anomalies detected",
            value=f"{anomaly_count:,}",
            delta=f"{anomaly_pct:.1f}% of total",
            delta_color="inverse",  # red delta = bad (high anomaly rate is undesirable)
        )

    with col3:
        st.metric(label="🔗 Links with anomalies", value=str(links_with_anomalies))

    with col4:
        st.metric(label="⏱️ Most recent anomaly", value=most_recent_str)


# =============================================================================
# SECTION 2 — Time Series Chart
# =============================================================================

def render_time_series(result_df: pd.DataFrame) -> None:
    """
    Main interactive chart: one line per link + red anomaly markers on top.

    Why go.Figure() instead of px.line()?
    We need to mix two trace types (line + scatter) in one chart. Plotly
    Express is a high-level shortcut; go.Figure() (graph_objects) gives us
    full control over each trace's properties.

    Why red scatter markers ON TOP of the continuous line?
    Drawing the line through anomalous points preserves the visual continuity
    of the time series. Overlaying markers makes anomalies immediately visible
    without breaking the reader's sense of the underlying trend.
    """
    st.subheader("📈 WAN Link Utilisation — Anomalies Highlighted")

    fig = go.Figure()

    for i, link_name in enumerate(sorted(result_df["link_name"].unique())):
        link_data = (
            result_df[result_df["link_name"] == link_name]
            .sort_values("timestamp")
        )
        colour = LINK_COLOURS[i % len(LINK_COLOURS)]

        # Continuous utilisation line — includes anomalous points for continuity
        fig.add_trace(go.Scatter(
            x=link_data["timestamp"],
            y=link_data["utilisation_percent"],
            mode="lines",
            name=link_name,
            line=dict(color=colour, width=1.5),
            opacity=0.7,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M UTC}<br>"
                "Utilisation: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
            customdata=link_data[["link_name"]].values,
        ))

        # Anomaly markers — red dots overlaid on the line at flagged points.
        # The white border (line dict) improves visibility against dark backgrounds.
        anomalies = link_data[link_data["is_anomaly"]]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["utilisation_percent"],
                mode="markers",
                name=f"{link_name} — anomaly",
                marker=dict(
                    color=ANOMALY_COLOUR,
                    size=10,
                    symbol="circle",
                    line=dict(color="white", width=1.5),
                ),
                hovertemplate=(
                    "<b>⚠️ ANOMALY — %{customdata[0]}</b><br>"
                    "Time: %{x|%Y-%m-%d %H:%M UTC}<br>"
                    "Utilisation: %{y:.1f}%<br>"
                    "Score: %{customdata[1]:.3f}<br>"
                    "<br><i>%{customdata[2]}</i><br>"
                    "<extra></extra>"
                ),
                customdata=anomalies[["link_name", "score_normalised", "explanation"]].values,
                showlegend=False,  # keeps legend readable — links only, not per-link anomaly series
            ))

    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Utilisation (%)",
        yaxis=dict(range=[0, 105]),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=TIMESERIES_HEIGHT,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,17,0.8)",
        margin=dict(l=50, r=20, t=40, b=50),
    )

    # SLA reference lines — helps engineers contextualise anomalies vs. thresholds
    fig.add_hline(
        y=SLA_WARN_PCT,
        line_dash="dash", line_color="orange", opacity=0.6,
        annotation_text=f"{SLA_WARN_PCT}% SLA threshold",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=SLA_CRIT_PCT,
        line_dash="dash", line_color="red", opacity=0.6,
        annotation_text=f"{SLA_CRIT_PCT}% critical",
        annotation_position="bottom right",
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SECTION 3 — Anomaly Details Table
# =============================================================================

def render_anomaly_table(result_df: pd.DataFrame) -> None:
    """
    Colour-coded table of anomalous rows with download button.

    Why Pandas Styler instead of st.table()?
    Styler lets us apply per-cell background colours to communicate severity
    without adding a separate visual element. The user can instantly distinguish
    High (red) from Medium (amber) anomalies at a glance.

    Note: Styler.map() is used here — not .applymap(), which was deprecated
    in pandas 2.1.0 and will be removed in a future release.
    """
    st.subheader("🔍 Anomaly Details")

    anomaly_df = result_df[result_df["is_anomaly"]].copy()

    if anomaly_df.empty:
        st.info(
            "No anomalies detected with the current settings. "
            "Try increasing the contamination threshold in the sidebar."
        )
        return

    # Add severity using the detector's threshold constants
    detector = NetworkAnomalyDetector()
    anomaly_df["severity"] = anomaly_df["score_normalised"].apply(detector.get_severity)

    # Build display-friendly column set
    display_map = {
        "timestamp":           "Timestamp",
        "link_name":           "Link Name",
        "utilisation_percent": "Utilisation (%)",
        "score_normalised":    "Anomaly Score",
        "explanation":         "Explanation",
        "severity":            "Severity",
    }
    display_df = (
        anomaly_df[list(display_map.keys())]
        .rename(columns=display_map)
        .copy()
    )
    display_df["Timestamp"]        = display_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
    display_df["Utilisation (%)"]  = display_df["Utilisation (%)"].round(1)
    display_df["Anomaly Score"]    = display_df["Anomaly Score"].round(3)
    display_df = display_df.sort_values("Anomaly Score", ascending=False)

    def _colour_severity(val: str) -> str:
        """Return CSS style string for a severity cell value."""
        if val == "High":
            return f"background-color: {HIGH_SEV_COLOUR}; color: white; font-weight: bold"
        if val == "Medium":
            return f"background-color: {MED_SEV_COLOUR}; color: white"
        return ""

    # .map() is the pandas 2.1+ replacement for the deprecated .applymap()
    styled = display_df.style.map(_colour_severity, subset=["Severity"])

    st.dataframe(styled, use_container_width=True, height=ANOMALY_TABLE_HEIGHT)

    # Download button — lets NOC engineers export for ticketing systems (JIRA, ServiceNow)
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download anomalies as CSV",
        data=csv_bytes,
        file_name=f"anomalies_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# =============================================================================
# SECTION 4 — Model Insights
# =============================================================================

def render_model_insights(result_df: pd.DataFrame) -> None:
    """
    Two side-by-side charts giving a statistical view of the detection results.

    Left:  Score distribution histogram — shows how well the model separates
           normal from anomalous (a bimodal distribution is ideal).
    Right: Anomaly count by link bar chart — shows which circuits need attention.
    """
    st.subheader("🧠 Model Insights")

    col_left, col_right = st.columns(2)

    # ── Left: Score distribution ───────────────────────────────────────────
    with col_left:
        st.markdown("**Anomaly Score Distribution**")
        st.caption(
            "A well-separated bimodal distribution (two humps) means the model "
            "found a clear boundary between normal and anomalous traffic."
        )

        normal_scores  = result_df[~result_df["is_anomaly"]]["score_normalised"]
        anomaly_scores = result_df[ result_df["is_anomaly"]]["score_normalised"]

        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=normal_scores, name="Normal",
            marker_color=NORMAL_COLOUR, opacity=0.7, nbinsx=40,
        ))
        hist_fig.add_trace(go.Histogram(
            x=anomaly_scores, name="Anomaly",
            marker_color=ANOMALY_COLOUR, opacity=0.8, nbinsx=40,
        ))
        hist_fig.update_layout(
            barmode="overlay",
            xaxis_title="Normalised Anomaly Score",
            yaxis_title="Count",
            height=INSIGHT_CHART_HEIGHT,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,17,17,0.8)",
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    # ── Right: Anomalies by link ───────────────────────────────────────────
    with col_right:
        st.markdown("**Anomalies by Link**")
        st.caption(
            "Links with many anomalies warrant a capacity review or circuit "
            "health investigation."
        )

        anomaly_counts = (
            result_df[result_df["is_anomaly"]]
            .groupby("link_name")
            .size()
            .reset_index(name="anomaly_count")
            .sort_values("anomaly_count", ascending=True)
        )

        if anomaly_counts.empty:
            st.info("No anomalies to display.")
        else:
            bar_fig = go.Figure(go.Bar(
                x=anomaly_counts["anomaly_count"],
                y=anomaly_counts["link_name"],
                orientation="h",
                marker=dict(
                    color=anomaly_counts["anomaly_count"],
                    colorscale="Reds",
                    showscale=False,
                ),
                text=anomaly_counts["anomaly_count"],
                textposition="outside",
            ))
            bar_fig.update_layout(
                xaxis_title="Number of Anomalies",
                yaxis_title="",
                height=INSIGHT_CHART_HEIGHT,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,17,17,0.8)",
                margin=dict(l=20, r=40, t=30, b=40),
            )
            st.plotly_chart(bar_fig, use_container_width=True)


# =============================================================================
# SECTION 5 — Ground Truth Evaluation (sample data only)
# =============================================================================

def render_ground_truth_comparison(result_df: pd.DataFrame) -> None:
    """
    Compare model predictions against injected anomaly labels.

    Only shown when anomaly_injected column is present (sample data).
    Demonstrates ML evaluation skills — precision, recall, F1.
    """
    if "anomaly_injected" not in result_df.columns:
        return

    st.subheader("🎯 Model Evaluation (Sample Data Only)")
    st.caption(
        "Since this is synthetic data, we know the ground truth. "
        "Real production data wouldn't have these labels."
    )

    y_true = result_df["anomaly_injected"].astype(bool)
    y_pred = result_df["is_anomaly"].astype(bool)

    tp = int(( y_true &  y_pred).sum())   # caught real anomalies
    fp = int((~y_true &  y_pred).sum())   # false alarms
    fn = int(( y_true & ~y_pred).sum())   # missed anomalies
    tn = int((~y_true & ~y_pred).sum())   # correctly identified normal

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", f"{precision:.1%}", help="Of flagged anomalies, how many were real?")
    c2.metric("Recall",    f"{recall:.1%}",    help="Of real anomalies, how many did we catch?")
    c3.metric("F1 Score",  f"{f1:.3f}",        help="Harmonic mean of precision and recall")
    c4.metric("True Positives",  str(tp))
    c5.metric("False Positives", str(fp))

    with st.expander("What do these metrics mean?"):
        st.markdown(f"""
**Precision** — "When the alarm fires, how often is it real?"
Low precision = too many false alarms. NOC engineers start ignoring the dashboard.

**Recall** — "How many real outages did we catch?"
Low recall = silent failures. Incidents go undetected until users complain.

**F1 Score** — harmonic mean of precision and recall.
The right single metric when both false positives AND false negatives have real costs.

> In network operations, **recall is usually more important than precision** —
> it's better to generate some false alarms than to miss a capacity breach or
> link failure that impacts users.

Current: Precision={precision:.1%}, Recall={recall:.1%}, F1={f1:.3f}
— missed {fn} real anomalies, generated {fp} false alarms out of {tp+fp+fn+tn:,} data points.
        """)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # --- Header ---
    st.title("🔬 Network Anomaly Detection Dashboard")
    st.markdown("*AI-powered WAN link health monitoring using Isolation Forest*")
    st.markdown("---")

    # --- Load sample data (cached after first call) ---
    with st.spinner("Loading sample data…"):
        base_df = load_sample_data()

    # --- Sidebar: resolves active data source and filters ---
    filtered_df, contamination, data_source = render_sidebar(base_df)

    if filtered_df.empty:
        st.warning(
            "No data matches the current filters. "
            "Adjust the date range or link selection in the sidebar."
        )
        return

    # Minimum rows check: IsolationForest needs enough data to be meaningful.
    # 10 rows per link is a conservative lower bound.
    min_rows_per_link = (
        filtered_df.groupby("link_id").size().min()
        if not filtered_df.empty else 0
    )
    if min_rows_per_link < 10:
        st.warning(
            f"At least one link has fewer than 10 data points after filtering "
            f"(min={min_rows_per_link}). Widen the date range for reliable results."
        )

    # --- Run detection (cached by contamination + serialised data fingerprint) ---
    with st.spinner("Running Isolation Forest…"):
        # orient='records' + date_format='iso' ensures timestamps survive the
        # JSON round-trip inside run_detection() without precision loss.
        serialised = filtered_df.to_json(orient="records", date_format="iso")
        result_df, training_summary = run_detection(data_source, contamination, serialised)

    # --- Render dashboard sections ---
    # Section 0: Model Training — make the AI visible before showing results
    render_model_training(training_summary, contamination)
    st.markdown("---")

    render_summary_metrics(result_df)
    st.markdown("---")

    render_time_series(result_df)
    st.markdown("---")

    render_anomaly_table(result_df)
    st.markdown("---")

    render_model_insights(result_df)

    # Ground truth section only makes sense when we have labelled data
    if "anomaly_injected" in result_df.columns:
        st.markdown("---")
        render_ground_truth_comparison(result_df)


if __name__ == "__main__":
    main()
