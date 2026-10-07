import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from dashboard.data_loader import (
    load_main_data,
    load_anomaly_data,
    load_stress_data,
    load_cluster_data
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Overview | UAC-CareAI",
    page_icon="🏛️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_main_data().copy()

try:
    anomaly_df = load_anomaly_data().copy()
except Exception:
    anomaly_df = pd.DataFrame()

try:
    stress_df = load_stress_data().copy()
except Exception:
    stress_df = pd.DataFrame()

try:
    cluster_df = load_cluster_data().copy()
except Exception:
    cluster_df = pd.DataFrame()


# ============================================================
# BASIC PREPARATION
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = (
    df
    .dropna(subset=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns(
    [4, 1]
)


with header_col1:

    st.markdown(
        "## 🏛️ UAC-CareAI"
    )

    st.markdown(
        "**System Capacity & Care-Load Analytics Platform**"
    )

    st.caption(
        "Data-driven insights for operational planning, "
        "risk assessment and resource management"
    )


with header_col2:

    st.markdown(
        "### DATA AVAILABLE"
    )

    st.caption(
        f"{df['Date'].min():%d %b %Y} → "
        f"{df['Date'].max():%d %b %Y}"
    )


st.divider()


# ============================================================
# DATE FILTER
# ============================================================

filter_col1, filter_col2 = st.columns(
    [3, 1]
)


with filter_col1:

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    selected_dates = st.date_input(
        "Reporting Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )


with filter_col2:

    st.markdown(
        "#### Dataset Coverage"
    )

    st.write(
        f"**{len(df):,} observations**"
    )


# ============================================================
# HANDLE DATE SELECTION
# ============================================================

if (
    isinstance(selected_dates, (tuple, list))
    and len(selected_dates) == 2
):

    start_date = selected_dates[0]
    end_date = selected_dates[1]

    data = df[
        (df["Date"].dt.date >= start_date)
        &
        (df["Date"].dt.date <= end_date)
    ].copy()

else:

    data = df.copy()


# ============================================================
# SAFETY CHECK
# ============================================================

if data.empty:

    st.warning(
        "No observations are available for the selected period."
    )

    st.stop()


# ============================================================
# LATEST OBSERVATION
# ============================================================

latest = data.iloc[-1]


current_load = latest["Total System Load"]

current_cbp = latest["Children in CBP custody"]

current_hhs = latest["Children in HHS Care"]

total_transfers = data[
    "Children transferred out of CBP custody"
].sum()

total_discharges = data[
    "Children discharged from HHS Care"
].sum()

current_pressure = latest[
    "Net Intake Pressure"
]


# ============================================================
# OPERATIONAL STATUS
# ============================================================

status = "NORMAL"

status_reason = "Current indicators are within the observed range."

if not anomaly_df.empty:

    anomaly_df["Date"] = pd.to_datetime(
        anomaly_df["Date"],
        errors="coerce"
    )

    anomaly_df = anomaly_df.sort_values("Date")

    recent_anomaly = anomaly_df[
        anomaly_df["Date"] <= latest["Date"]
    ]

    if not recent_anomaly.empty:

        recent_row = recent_anomaly.iloc[-1]

        if "Anomaly Severity" in recent_row.index:

            severity = str(
                recent_row["Anomaly Severity"]
            ).upper()

            if severity in ["CRITICAL", "HIGH"]:

                status = severity

                status_reason = (
                    "Recent observation flagged as "
                    f"{severity.lower()} anomaly."
                )


# ============================================================
# EXECUTIVE HEADER
# ============================================================

st.markdown(
    "### EXECUTIVE OVERVIEW"
)

st.title(
    "System Capacity & Care-Load"
)

st.caption(
    "Operational intelligence, capacity indicators "
    "and risk-oriented analytics"
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader(
    "Key Performance Indicators"
)


k1, k2, k3, k4, k5, k6 = st.columns(6)


with k1:

    st.metric(
        "Current System Load",
        f"{current_load:,.0f}"
    )


with k2:

    st.metric(
        "CBP Custody",
        f"{current_cbp:,.0f}"
    )


with k3:

    st.metric(
        "HHS Care",
        f"{current_hhs:,.0f}"
    )


with k4:

    st.metric(
        "Total Transfers",
        f"{total_transfers:,.0f}"
    )


with k5:

    st.metric(
        "Total Discharges",
        f"{total_discharges:,.0f}"
    )


with k6:

    if status in ["CRITICAL", "HIGH"]:

        st.error(
            f"⚠ {status}"
        )

    else:

        st.success(
            "● NORMAL"
        )

    st.caption(
        "Operational Status"
    )


st.divider()


# ============================================================
# MAIN ANALYTICS ROW
# ============================================================

left_col, right_col = st.columns(
    [2, 1]
)


# ============================================================
# SYSTEM LOAD TREND
# ============================================================

with left_col:

    st.subheader(
        "System Load Trend"
    )

    chart_df = data[
        [
            "Date",
            "Total System Load"
        ]
    ].copy()

    chart_df["7-Day Moving Average"] = (
        chart_df["Total System Load"]
        .rolling(7)
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["Date"],
            y=chart_df["Total System Load"],
            mode="lines",
            name="System Load",
            line=dict(
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["Date"],
            y=chart_df["7-Day Moving Average"],
            mode="lines",
            name="7-Day Moving Average",
            line=dict(
                width=2,
                dash="dash"
            )
        )
    )

    fig.update_layout(
        height=390,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        hovermode="x unified",
        xaxis_title="",
        yaxis_title="System Load",
        legend=dict(
            orientation="h"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# CBP VS HHS
# ============================================================

with right_col:

    st.subheader(
        "CBP Custody vs HHS Care"
    )

    population_df = data[
        [
            "Date",
            "Children in CBP custody",
            "Children in HHS Care"
        ]
    ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=population_df["Date"],
            y=population_df[
                "Children in CBP custody"
            ],
            mode="lines",
            name="CBP Custody",
            stackgroup="one"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=population_df["Date"],
            y=population_df[
                "Children in HHS Care"
            ],
            mode="lines",
            name="HHS Care",
            stackgroup="one"
        )
    )

    fig.update_layout(
        height=390,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        hovermode="x unified",
        xaxis_title="",
        yaxis_title="Children"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# SECOND ANALYTICS ROW
# ============================================================

left_col, right_col = st.columns(
    [2, 1]
)


# ============================================================
# TRANSFERS VS DISCHARGES
# ============================================================

with left_col:

    st.subheader(
        "Transfers vs Discharges"
    )

    flow_df = data[
        [
            "Date",
            "Children transferred out of CBP custody",
            "Children discharged from HHS Care"
        ]
    ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=flow_df["Date"],
            y=flow_df[
                "Children transferred out of CBP custody"
            ],
            mode="lines",
            name="Transfers"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=flow_df["Date"],
            y=flow_df[
                "Children discharged from HHS Care"
            ],
            mode="lines",
            name="Discharges"
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        hovermode="x unified",
        xaxis_title="",
        yaxis_title="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# KEY INDICATORS
# ============================================================

with right_col:

    st.subheader(
        "Key Indicators"
    )

    st.metric(
        "Net Intake Pressure",
        f"{current_pressure:,.0f}"
    )

    if "7-Day Net Intake" in latest.index:

        st.metric(
            "7-Day Net Intake",
            f"{latest['7-Day Net Intake']:,.0f}"
        )

    if "Load Volatility 7D" in latest.index:

        st.metric(
            "7-Day Load Volatility",
            f"{latest['Load Volatility 7D']:,.1f}"
        )

    if "Daily Load Change" in latest.index:

        st.metric(
            "Daily Load Change",
            f"{latest['Daily Load Change']:,.0f}"
        )


# ============================================================
# THIRD ROW
# ============================================================

left_col, right_col = st.columns(
    [2, 1]
)


# ============================================================
# INTAKE PRESSURE
# ============================================================

with left_col:

    st.subheader(
        "Net Intake Pressure"
    )

    pressure_df = data[
        [
            "Date",
            "Net Intake Pressure"
        ]
    ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pressure_df["Date"],
            y=pressure_df[
                "Net Intake Pressure"
            ],
            mode="lines",
            fill="tozeroy",
            name="Net Intake Pressure"
        )
    )

    fig.update_layout(
        height=340,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        hovermode="x unified",
        xaxis_title="",
        yaxis_title="Pressure"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# RECENT ALERTS
# ============================================================

with right_col:

    st.subheader(
        "Recent Alerts & Anomalies"
    )

    if not anomaly_df.empty:

        recent = anomaly_df[
            anomaly_df["Date"] <= latest["Date"]
        ].copy()

        if "Anomaly Flag" in recent.columns:

            recent = recent[
                recent["Anomaly Flag"] == 1
            ]

        recent = (
            recent
            .sort_values("Date", ascending=False)
            .head(5)
        )


        if recent.empty:

            st.success(
                "No recent anomalies detected."
            )

        else:

            for _, row in recent.iterrows():

                severity = str(
                    row.get(
                        "Anomaly Severity",
                        "HIGH"
                    )
                ).upper()

                date_text = row["Date"].strftime(
                    "%d %b %Y"
                )

                if severity == "CRITICAL":

                    st.error(
                        f"**CRITICAL** — "
                        f"Unusual operational observation "
                        f"({date_text})"
                    )

                elif severity == "HIGH":

                    st.warning(
                        f"**HIGH** — "
                        f"Unusual operational observation "
                        f"({date_text})"
                    )

                else:

                    st.info(
                        f"**{severity}** — "
                        f"Unusual operational observation "
                        f"({date_text})"
                    )

    else:

        st.info(
            "Anomaly results are not available."
        )


# ============================================================
# RECENT OPERATIONAL ACTIVITY
# ============================================================

st.divider()

st.subheader(
    "Recent Operational Activity"
)


activity = data[
    [
        "Date",
        "Total System Load",
        "Children in CBP custody",
        "Children in HHS Care",
        "Net Intake Pressure"
    ]
].tail(10).copy()


activity = activity.sort_values(
    "Date",
    ascending=False
)


activity["Date"] = (
    activity["Date"]
    .dt.strftime("%d %b %Y")
)


activity = activity.rename(
    columns={
        "Date": "Date",
        "Total System Load": "System Load",
        "Children in CBP custody": "CBP Custody",
        "Children in HHS Care": "HHS Care",
        "Net Intake Pressure": "Net Intake Pressure"
    }
)


st.dataframe(
    activity,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA COVERAGE
# ============================================================

st.divider()

st.subheader(
    "Data Coverage"
)


coverage_col1, coverage_col2, coverage_col3 = st.columns(3)


with coverage_col1:

    st.metric(
        "Observations",
        f"{len(df):,}"
    )


with coverage_col2:

    st.metric(
        "Reporting Start",
        df["Date"].min().strftime(
            "%d %b %Y"
        )
    )


with coverage_col3:

    st.metric(
        "Reporting End",
        df["Date"].max().strftime(
            "%d %b %Y"
        )
    )


# ============================================================
# METHODOLOGY NOTE
# ============================================================

with st.expander(
    "About this dashboard"
):

    st.write(
        """
        **UAC-CareAI** is an analytical decision-support
        prototype for examining system capacity and care-load
        conditions.

        The dashboard combines cleaned historical operational
        data with derived indicators including system load,
        net intake pressure, rolling averages and operational
        flows.

        Additional project modules provide machine-learning
        prediction, time-series forecasting, anomaly detection,
        operational clustering and explainable AI.

        Dashboard indicators are intended for analytical
        decision support and should be interpreted alongside
        operational context and data-quality considerations.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "UAC-CareAI • System Capacity & Care-Load Analytics "
    "Platform • Analytical Decision-Support Prototype"
)