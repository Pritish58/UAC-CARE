import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from dashboard.data_loader import load_main_data


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Capacity & Load | UAC-CareAI",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_main_data().copy()

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

st.markdown(
    "### CAPACITY & LOAD ANALYTICS"
)

st.title(
    "📊 System Capacity & Load"
)

st.caption(
    "Detailed analysis of system load, intake pressure, "
    "operational flows and short-term capacity conditions."
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
        "Analysis Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )


with filter_col2:

    st.markdown(
        "#### Dataset Coverage"
    )

    st.write(
        f"{len(df):,} observations"
    )


# ============================================================
# FILTER DATA
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


if data.empty:

    st.warning(
        "No data is available for the selected period."
    )

    st.stop()


# ============================================================
# CURRENT VALUES
# ============================================================

latest = data.iloc[-1]

current_load = latest["Total System Load"]

average_load = data[
    "Total System Load"
].mean()

peak_load = data[
    "Total System Load"
].max()

average_pressure = data[
    "Net Intake Pressure"
].mean()


# ============================================================
# KPI SECTION
# ============================================================

st.subheader(
    "Capacity Indicators"
)


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.metric(
        "Current System Load",
        f"{current_load:,.0f}"
    )


with k2:

    st.metric(
        "Average System Load",
        f"{average_load:,.0f}"
    )


with k3:

    st.metric(
        "Peak System Load",
        f"{peak_load:,.0f}"
    )


with k4:

    st.metric(
        "Average Intake Pressure",
        f"{average_pressure:,.1f}"
    )


st.divider()


# ============================================================
# SYSTEM LOAD ANALYSIS
# ============================================================

st.subheader(
    "System Load Analysis"
)


load_df = data[
    [
        "Date",
        "Total System Load"
    ]
].copy()


load_df["7-Day Average"] = (
    load_df["Total System Load"]
    .rolling(7)
    .mean()
)


load_df["14-Day Average"] = (
    load_df["Total System Load"]
    .rolling(14)
    .mean()
)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=load_df["Date"],
        y=load_df["Total System Load"],
        mode="lines",
        name="Daily System Load"
    )
)


fig.add_trace(
    go.Scatter(
        x=load_df["Date"],
        y=load_df["7-Day Average"],
        mode="lines",
        name="7-Day Average",
        line=dict(
            dash="dash"
        )
    )
)


fig.add_trace(
    go.Scatter(
        x=load_df["Date"],
        y=load_df["14-Day Average"],
        mode="lines",
        name="14-Day Average",
        line=dict(
            dash="dot"
        )
    )
)


fig.update_layout(
    height=450,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="System Load",
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
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
# INTAKE PRESSURE + LOAD CHANGE
# ============================================================

left_col, right_col = st.columns(2)


# ============================================================
# INTAKE PRESSURE
# ============================================================

with left_col:

    st.subheader(
        "Net Intake Pressure"
    )

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Net Intake Pressure"],
            mode="lines",
            fill="tozeroy",
            name="Net Intake Pressure"
        )
    )


    fig.update_layout(
        height=360,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Net Intake Pressure",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
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
# DAILY LOAD CHANGE
# ============================================================

with right_col:

    st.subheader(
        "Daily Load Change"
    )


    fig = go.Figure()


    fig.add_trace(
        go.Bar(
            x=data["Date"],
            y=data["Daily Load Change"],
            name="Daily Load Change"
        )
    )


    fig.update_layout(
        height=360,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Change in System Load",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
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
# VOLATILITY + 7-DAY NET INTAKE
# ============================================================

left_col, right_col = st.columns(2)


# ============================================================
# LOAD VOLATILITY
# ============================================================

with left_col:

    st.subheader(
        "7-Day Load Volatility"
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Load Volatility 7D"],
            mode="lines",
            name="7-Day Volatility"
        )
    )


    fig.update_layout(
        height=350,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Volatility",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
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
# 7-DAY NET INTAKE
# ============================================================

with right_col:

    st.subheader(
        "7-Day Net Intake"
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["7-Day Net Intake"],
            mode="lines",
            name="7-Day Net Intake"
        )
    )


    fig.add_hline(
        y=0,
        line_dash="dash"
    )


    fig.update_layout(
        height=350,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="7-Day Net Intake",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
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
# OPERATIONAL CAPACITY SNAPSHOT
# ============================================================

st.divider()

st.subheader(
    "Current Capacity Snapshot"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "CBP Custody",
        f"{latest['Children in CBP custody']:,.0f}"
    )


with c2:

    st.metric(
        "HHS Care",
        f"{latest['Children in HHS Care']:,.0f}"
    )


with c3:

    st.metric(
        "7-Day Net Intake",
        f"{latest['7-Day Net Intake']:,.0f}"
    )


with c4:

    st.metric(
        "Load Volatility",
        f"{latest['Load Volatility 7D']:,.1f}"
    )


# ============================================================
# HIGHEST LOAD DAYS
# ============================================================

st.divider()

st.subheader(
    "Highest System Load Days"
)


top_days = data[
    [
        "Date",
        "Total System Load",
        "Net Intake Pressure",
        "Children in CBP custody",
        "Children in HHS Care"
    ]
].copy()


top_days = (
    top_days
    .sort_values(
        "Total System Load",
        ascending=False
    )
    .head(10)
)


top_days["Date"] = (
    top_days["Date"]
    .dt.strftime("%d %b %Y")
)


top_days = top_days.rename(
    columns={
        "Date": "Date",
        "Total System Load": "System Load",
        "Net Intake Pressure": "Net Intake Pressure",
        "Children in CBP custody": "CBP Custody",
        "Children in HHS Care": "HHS Care"
    }
)


st.dataframe(
    top_days,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CAPACITY INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "Capacity Interpretation"
)


with st.container(border=True):

    st.write(
        """
        **System Load** represents the combined operational
        population reflected in CBP custody and HHS care.
        """
    )

    st.write(
        """
        **Net Intake Pressure** compares incoming operational
        pressure with transfers and discharges, helping identify
        periods where system load may be increasing.
        """
    )

    st.write(
        """
        **Load Volatility** measures short-term variation in
        system load and can help identify periods of unstable
        operational conditions.
        """
    )

    st.write(
        """
        **Highest System Load Days** highlight historical
        periods where capacity demand reached its greatest
        observed levels.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UAC-CareAI • Capacity & Load Analytics"
)

st.caption(
    "Detailed operational capacity and system-load analysis"
)