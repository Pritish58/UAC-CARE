import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Operational Patterns | UAC-CareAI",
    page_icon="🔎",
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

CLUSTER_FILE = (
    PROCESSED_DIR
    / "uac_operational_clusters.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "cluster_summary.csv"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔎 Operational Patterns")

st.caption(
    "Data-driven identification of recurring system "
    "capacity and intake conditions"
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

if not CLUSTER_FILE.exists():

    st.error(
        "Operational cluster dataset was not found."
    )

    st.stop()


df = pd.read_csv(
    CLUSTER_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"]
)


summary = None

if SUMMARY_FILE.exists():

    summary = pd.read_csv(
        SUMMARY_FILE
    )


# ============================================================
# IDENTIFY CLUSTER COLUMN
# ============================================================

cluster_column = (
    "Operational Cluster"
    if "Operational Cluster" in df.columns
    else None
)

pattern_column = (
    "Operational Pattern"
    if "Operational Pattern" in df.columns
    else None
)


if cluster_column is None:

    st.error(
        "Operational Cluster column was not found."
    )

    st.stop()


# ============================================================
# BASIC STATISTICS
# ============================================================

cluster_count = (
    df[cluster_column]
    .nunique()
)

largest_cluster = (
    df[cluster_column]
    .value_counts()
    .idxmax()
)

largest_cluster_size = (
    df[cluster_column]
    .value_counts()
    .max()
)


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Operational Patterns",
        f"{cluster_count}"
    )


with col2:

    st.metric(
        "Largest Cluster",
        f"Cluster {largest_cluster}"
    )


with col3:

    st.metric(
        "Largest Cluster Size",
        f"{largest_cluster_size:,}"
    )


# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

st.subheader(
    "Operational Pattern Distribution"
)

cluster_counts = (
    df[cluster_column]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_counts.columns = [
    "Operational Cluster",
    "Observations"
]

fig = px.bar(
    cluster_counts,
    x="Operational Cluster",
    y="Observations",
    text="Observations",
    labels={
        "Operational Cluster":
            "Cluster",
        "Observations":
            "Number of Observations"
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
# CLUSTER BEHAVIOR
# ============================================================

st.divider()

st.subheader(
    "Cluster Load & Intake Pressure"
)

if summary is not None:

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if "Total System Load" in summary.columns:

        fig = px.bar(
            summary,
            x="Operational Cluster",
            y="Total System Load",
            color="Operational Pattern"
            if "Operational Pattern"
            in summary.columns
            else None,
            text="Total System Load",
            labels={
                "Total System Load":
                    "Average System Load",
                "Operational Cluster":
                    "Cluster"
            }
        )

        fig.update_layout(
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Intake Pressure
    # --------------------------------------------------------

    if "Net Intake Pressure" in summary.columns:

        fig = px.bar(
            summary,
            x="Operational Cluster",
            y="Net Intake Pressure",
            color="Operational Pattern"
            if "Operational Pattern"
            in summary.columns
            else None,
            text="Net Intake Pressure",
            labels={
                "Net Intake Pressure":
                    "Average Net Intake Pressure",
                "Operational Cluster":
                    "Cluster"
            }
        )

        fig.update_layout(
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# CLUSTER OVER TIME
# ============================================================

st.divider()

st.subheader(
    "Operational Pattern Over Time"
)

time_df = df[
    [
        "Date",
        cluster_column
    ]
].copy()

fig = px.scatter(
    time_df,
    x="Date",
    y=cluster_column,
    color=cluster_column,
    labels={
        "Date": "Date",
        cluster_column: "Operational Cluster"
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
# PATTERN DESCRIPTIONS
# ============================================================

if (
    summary is not None
    and "Operational Pattern" in summary.columns
):

    st.divider()

    st.subheader(
        "Operational Pattern Interpretation"
    )

    for _, row in summary.iterrows():

        cluster = row[
            "Operational Cluster"
        ]

        pattern = row[
            "Operational Pattern"
        ]

        load = row.get(
            "Total System Load",
            None
        )

        pressure = row.get(
            "Net Intake Pressure",
            None
        )

        observations = row.get(
            "Observations",
            None
        )

        st.markdown(
            f"""
            **Cluster {cluster} — {pattern}**

            - Observations: `{observations:,.0f}`
            - Average system load:
              `{load:,.1f}` if available
            - Average intake pressure:
              `{pressure:,.1f}` if available
            """
        )


# ============================================================
# DETAILED SUMMARY
# ============================================================

if summary is not None:

    st.divider()

    st.subheader(
        "Detailed Cluster Summary"
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# METHODOLOGY NOTE
# ============================================================

st.divider()

st.info(
    """
    **Methodology:** K-Means clustering groups observations
    according to similarities across operational load,
    custody, transfer, care and intake-pressure indicators.

    Cluster labels are analytical group identifiers rather
    than predefined official categories. Their interpretation
    is based on the characteristics of each cluster.
    """
)