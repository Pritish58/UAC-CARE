import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Risk & Anomalies | UAC-CareAI",
    page_icon="⚠️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    BASE_DIR / "Data" / "processed"
)

REPORT_DIR = (
    BASE_DIR / "reports"
)


STRESS_FILE = (
    PROCESSED_DIR / "uac_stress_labels.csv"
)

ANOMALY_FILE = (
    PROCESSED_DIR / "uac_anomaly_results.csv"
)

STRESS_RESULTS_FILE = (
    REPORT_DIR / "stress_model_results.csv"
)


# ============================================================
# HEADER
# ============================================================

st.title("⚠️ Risk & Anomaly Intelligence")

st.caption(
    "Operational stress classification and detection of "
    "unusual system conditions"
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

stress_df = None
anomaly_df = None
stress_results = None


if STRESS_FILE.exists():

    stress_df = pd.read_csv(
        STRESS_FILE
    )

    if "Date" in stress_df.columns:
        stress_df["Date"] = pd.to_datetime(
            stress_df["Date"]
        )


if ANOMALY_FILE.exists():

    anomaly_df = pd.read_csv(
        ANOMALY_FILE
    )

    if "Date" in anomaly_df.columns:
        anomaly_df["Date"] = pd.to_datetime(
            anomaly_df["Date"]
        )


if STRESS_RESULTS_FILE.exists():

    stress_results = pd.read_csv(
        STRESS_RESULTS_FILE
    )


# ============================================================
# DATA AVAILABILITY CHECK
# ============================================================

if stress_df is None:

    st.error(
        "Stress classification dataset was not found."
    )

    st.stop()


if anomaly_df is None:

    st.error(
        "Anomaly detection dataset was not found."
    )

    st.stop()


# ============================================================
# FIND STRESS COLUMN
# ============================================================

possible_stress_columns = [
    "Stress Level",
    "Stress Category",
    "Stress Class",
    "Stress Label",
    "Risk Level",
    "Risk Category"
]

stress_column = next(
    (
        col
        for col in possible_stress_columns
        if col in stress_df.columns
    ),
    None
)


# ============================================================
# FIND ANOMALY COLUMNS
# ============================================================

anomaly_flag_column = (
    "Anomaly Flag"
    if "Anomaly Flag" in anomaly_df.columns
    else None
)

anomaly_status_column = (
    "Anomaly Status"
    if "Anomaly Status" in anomaly_df.columns
    else None
)

anomaly_severity_column = (
    "Anomaly Severity"
    if "Anomaly Severity" in anomaly_df.columns
    else None
)

anomaly_score_column = (
    "Anomaly Score"
    if "Anomaly Score" in anomaly_df.columns
    else None
)


# ============================================================
# STRESS OVERVIEW
# ============================================================

st.subheader("Operational Stress Overview")


if stress_column is not None:

    stress_counts = (
        stress_df[stress_column]
        .value_counts()
        .reset_index()
    )

    stress_counts.columns = [
        "Stress Level",
        "Observations"
    ]

    total_observations = len(
        stress_df
    )

    critical_count = int(
        stress_counts.loc[
            stress_counts["Stress Level"]
            .astype(str)
            .str.upper()
            == "CRITICAL",
            "Observations"
        ].sum()
    )

    high_count = int(
        stress_counts.loc[
            stress_counts["Stress Level"]
            .astype(str)
            .str.upper()
            == "HIGH",
            "Observations"
        ].sum()
    )

else:

    stress_counts = pd.DataFrame()

    total_observations = len(
        stress_df
    )

    critical_count = 0
    high_count = 0


# ============================================================
# ANOMALY METRICS
# ============================================================

if anomaly_flag_column is not None:

    anomaly_count = int(
        pd.to_numeric(
            anomaly_df[anomaly_flag_column],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )

else:

    anomaly_count = 0


anomaly_rate = (
    anomaly_count
    / len(anomaly_df)
    * 100
)


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Observations",
        f"{total_observations:,}"
    )


with col2:

    st.metric(
        "High Stress",
        f"{high_count:,}"
    )


with col3:

    st.metric(
        "Critical Stress",
        f"{critical_count:,}"
    )


with col4:

    st.metric(
        "Anomalies",
        f"{anomaly_count:,}"
    )


with col5:

    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )


# ============================================================
# STRESS DISTRIBUTION
# ============================================================

if not stress_counts.empty:

    st.subheader("Stress Level Distribution")

    fig = px.bar(
        stress_counts,
        x="Stress Level",
        y="Observations",
        labels={
            "Stress Level": "Operational Stress",
            "Observations": "Number of Observations"
        }
    )

    fig.update_layout(
        height=380,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# STRESS OVER TIME
# ============================================================

if (
    stress_column is not None
    and "Date" in stress_df.columns
):

    st.subheader("Stress Classification Over Time")

    stress_time = (
        stress_df[
            [
                "Date",
                stress_column
            ]
        ]
        .rename(
            columns={
                stress_column: "Stress Level"
            }
        )
    )

    stress_time["Stress Code"] = (
        stress_time["Stress Level"]
        .astype(str)
        .str.upper()
        .map({
            "LOW": 1,
            "MODERATE": 2,
            "HIGH": 3,
            "CRITICAL": 4
        })
    )

    fig = px.scatter(
        stress_time,
        x="Date",
        y="Stress Code",
        color="Stress Level",
        hover_data=["Stress Level"],
        labels={
            "Stress Code":
                "Operational Stress Level",
            "Date":
                "Date"
        }
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[1, 2, 3, 4],
        ticktext=[
            "Low",
            "Moderate",
            "High",
            "Critical"
        ]
    )

    fig.update_layout(
        height=400,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# ANOMALY ANALYSIS
# ============================================================

st.divider()

st.subheader("Anomaly Detection")


if (
    anomaly_flag_column is not None
    and "Date" in anomaly_df.columns
):

    anomaly_plot = anomaly_df.copy()

    anomaly_plot["Anomaly Label"] = (
        anomaly_plot[anomaly_flag_column]
        .map({
            0: "NORMAL",
            1: "ANOMALY"
        })
    )

    fig = px.scatter(
        anomaly_plot,
        x="Date",
        y="Total System Load"
        if "Total System Load" in anomaly_plot.columns
        else anomaly_score_column,
        color="Anomaly Label",
        hover_data=[
            col
            for col in [
                "Total System Load",
                "Net Intake Pressure",
                anomaly_score_column,
                anomaly_severity_column
            ]
            if col is not None
            and col in anomaly_plot.columns
        ],
        labels={
            "Date": "Date",
            "Total System Load":
                "System Load"
        }
    )

    fig.update_layout(
        height=420,
        hovermode="closest"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# ANOMALY SEVERITY
# ============================================================

if anomaly_severity_column is not None:

    st.subheader("Anomaly Severity Distribution")

    severity_counts = (
        anomaly_df[
            anomaly_severity_column
        ]
        .value_counts()
        .reset_index()
    )

    severity_counts.columns = [
        "Severity",
        "Observations"
    ]

    fig = px.bar(
        severity_counts,
        x="Severity",
        y="Observations",
        labels={
            "Severity":
                "Anomaly Severity",
            "Observations":
                "Observations"
        }
    )

    fig.update_layout(
        height=350,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TOP ANOMALOUS DAYS
# ============================================================

if (
    anomaly_flag_column is not None
    and anomaly_score_column is not None
):

    st.divider()

    st.subheader(
        "Highest-Risk Anomalous Days"
    )

    top_anomalies = (
        anomaly_df[
            anomaly_df[anomaly_flag_column] == 1
        ]
        .sort_values(
            anomaly_score_column,
            ascending=False
        )
        .head(15)
        .copy()
    )

    display_columns = [
        col
        for col in [
            "Date",
            "Total System Load",
            "Net Intake Pressure",
            anomaly_score_column,
            anomaly_severity_column
        ]
        if col is not None
        and col in top_anomalies.columns
    ]

    if display_columns:

        display_df = top_anomalies[
            display_columns
        ].copy()

        if "Date" in display_df.columns:

            display_df["Date"] = (
                display_df["Date"]
                .dt.strftime("%d %b %Y")
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

if stress_results is not None:

    st.divider()

    st.subheader(
        "Stress Classification Model Performance"
    )

    st.dataframe(
        stress_results,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# METHODOLOGY NOTE
# ============================================================

st.divider()

st.info(
    """
    **Interpretation:** Stress levels are derived from
    operational indicators and used as a decision-support
    classification. Anomaly detection uses Isolation Forest
    to identify observations that differ substantially from
    learned historical operational patterns.

    These indicators identify conditions requiring attention;
    they are not intended to replace operational judgment.
    """
)