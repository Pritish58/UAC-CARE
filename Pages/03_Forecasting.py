import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Forecasting | UAC-CareAI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "Models"

SARIMA_RESULTS_FILE = REPORT_DIR / "forecast_results.csv"
SARIMA_FORECAST_FILE = REPORT_DIR / "sarima_forecast.csv"

SARIMA_MODEL_FILE = (
    MODEL_DIR / "sarima_forecasting_model.pkl"
)


# ============================================================
# THEME
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode


if dark:

    BG = "#07111F"
    CARD = "#0F1B2D"
    CARD_ALT = "#13243A"
    TEXT = "#F5F9FC"
    MUTED = "#91A5B8"
    BORDER = "#243A53"

    BLUE = "#60A5FA"
    CYAN = "#38BDF8"
    GREEN = "#34D399"
    ORANGE = "#FBBF24"
    RED = "#FB7185"

else:

    BG = "#F4F7FB"
    CARD = "#FFFFFF"
    CARD_ALT = "#F8FAFD"
    TEXT = "#14283D"
    MUTED = "#60758A"
    BORDER = "#DCE5EE"

    BLUE = "#1769E0"
    CYAN = "#087EA4"
    GREEN = "#159957"
    ORANGE = "#C77700"
    RED = "#D83A3A"


# ============================================================
# GLOBAL CSS
# IMPORTANT:
# No HTML content is rendered through st.markdown.
# CSS only.
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}

    /* --------------------------------------------------------
       MAIN TITLES
    -------------------------------------------------------- */

    h1, h2, h3, h4 {{
        color: {TEXT} !important;
    }}

    p, label {{
        color: {MUTED};
    }}

    /* --------------------------------------------------------
       METRIC CARDS
    -------------------------------------------------------- */

    [data-testid="stMetric"] {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.035);
    }}

    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}

    [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
        font-weight: 850 !important;
    }}

    /* --------------------------------------------------------
       CONTAINERS
    -------------------------------------------------------- */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {CARD};
        border-color: {BORDER} !important;
        border-radius: 14px;
    }}

    /* --------------------------------------------------------
       DATAFRAME
    -------------------------------------------------------- */

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    /* --------------------------------------------------------
       ALERTS
    -------------------------------------------------------- */

    [data-testid="stAlert"] {{
        border-radius: 12px;
    }}

    /* --------------------------------------------------------
       DIVIDERS
    -------------------------------------------------------- */

    hr {{
        border-color: {BORDER};
    }}

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    [data-testid="stSidebar"] {{
        background-color: {CARD};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path):

    if not path.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except Exception as error:

        st.error(
            f"Unable to read {path.name}: {error}"
        )

        return pd.DataFrame()


def find_column(df, candidates):

    if df.empty:
        return None

    # Exact match
    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    # Case-insensitive match
    column_map = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = str(candidate).strip().lower()

        if key in column_map:
            return column_map[key]

    return None


def numeric_value(df, column):

    if (
        df.empty
        or column is None
        or column not in df.columns
    ):
        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return None

    return float(values.iloc[0])


def style_chart(fig, height=430):

    fig.update_layout(
        template=(
            "plotly_dark"
            if dark
            else "plotly_white"
        ),
        height=height,
        margin=dict(
            l=25,
            r=25,
            t=65,
            b=30
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    return fig


# ============================================================
# LOAD REPORT FILES
# ============================================================

sarima_results = load_csv(
    SARIMA_RESULTS_FILE
)

sarima_forecast = load_csv(
    SARIMA_FORECAST_FILE
)


# ============================================================
# PAGE HEADER
# ============================================================

st.caption(
    "UAC-CAREAI  /  FORWARD PLANNING ENGINE"
)

st.title(
    "🔮 System Load Forecasting"
)

st.write(
    "Time-series forecasting for understanding system-load "
    "behaviour, validating SARIMA predictions and supporting "
    "forward-looking operational planning."
)


# ============================================================
# FORECASTING ENGINE
# NATIVE STREAMLIT — NO HTML
# ============================================================

with st.container(border=True):

    st.caption(
        "FORECASTING ENGINE"
    )

    st.subheader(
        "SARIMA Time-Series Analysis"
    )

    st.write(
        "The forecasting engine uses historical system-load "
        "behaviour to evaluate SARIMA predictions and, when "
        "the trained forecasting model is available, generate "
        "a forward 30-day system-load projection."
    )


# ============================================================
# FILE STATUS
# ============================================================

st.divider()

st.subheader(
    "Forecasting Data Status"
)

status1, status2, status3 = st.columns(3)


with status1:

    if not sarima_forecast.empty:

        st.success(
            "✓ sarima_forecast.csv available"
        )

    else:

        st.error(
            "✗ sarima_forecast.csv missing"
        )


with status2:

    if not sarima_results.empty:

        st.success(
            "✓ forecast_results.csv available"
        )

    else:

        st.warning(
            "⚠ forecast_results.csv missing"
        )


with status3:

    if SARIMA_MODEL_FILE.exists():

        st.success(
            "✓ SARIMA model available"
        )

    else:

        st.warning(
            "⚠ SARIMA model unavailable"
        )


# ============================================================
# STOP ONLY IF MAIN FORECAST FILE IS MISSING
# ============================================================

if sarima_forecast.empty:

    st.error(
        "The SARIMA forecast output could not be loaded."
    )

    st.info(
        "Please make sure this file exists:\n\n"
        "reports/sarima_forecast.csv"
    )

    st.stop()


# ============================================================
# EXACT COLUMNS FROM YOUR ACTUAL CSV
# ============================================================

DATE_COL = find_column(
    sarima_forecast,
    [
        "Date"
    ]
)

ACTUAL_COL = find_column(
    sarima_forecast,
    [
        "Actual Load"
    ]
)

FORECAST_COL = find_column(
    sarima_forecast,
    [
        "Forecast Load"
    ]
)

ERROR_COL = find_column(
    sarima_forecast,
    [
        "Forecast Error"
    ]
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required = {
    "Date": DATE_COL,
    "Actual Load": ACTUAL_COL,
    "Forecast Load": FORECAST_COL,
    "Forecast Error": ERROR_COL
}


missing_columns = [
    name
    for name, actual
    in required.items()
    if actual is None
]


if missing_columns:

    st.error(
        "The SARIMA forecast file does not contain the "
        "expected columns."
    )

    st.write(
        "Missing columns:"
    )

    st.write(
        missing_columns
    )

    st.write(
        "Columns found in your file:"
    )

    st.write(
        list(sarima_forecast.columns)
    )

    st.stop()


# ============================================================
# PREPARE FORECAST DATA
# ============================================================

forecast_df = sarima_forecast.copy()


forecast_df[DATE_COL] = pd.to_datetime(
    forecast_df[DATE_COL],
    errors="coerce"
)


forecast_df[ACTUAL_COL] = pd.to_numeric(
    forecast_df[ACTUAL_COL],
    errors="coerce"
)


forecast_df[FORECAST_COL] = pd.to_numeric(
    forecast_df[FORECAST_COL],
    errors="coerce"
)


forecast_df[ERROR_COL] = pd.to_numeric(
    forecast_df[ERROR_COL],
    errors="coerce"
)


forecast_df = forecast_df.dropna(
    subset=[
        DATE_COL,
        FORECAST_COL
    ]
)


forecast_df = forecast_df.sort_values(
    DATE_COL
).reset_index(drop=True)


# ============================================================
# SECTION 01
# FORECAST SNAPSHOT
# ============================================================

st.divider()

st.caption(
    "01 / FORECAST SNAPSHOT"
)

st.subheader(
    "SARIMA Forecast Overview"
)

st.write(
    "Summary statistics calculated directly from the "
    "available SARIMA forecast output."
)


average_forecast = forecast_df[
    FORECAST_COL
].mean()

maximum_forecast = forecast_df[
    FORECAST_COL
].max()

minimum_forecast = forecast_df[
    FORECAST_COL
].min()

record_count = len(
    forecast_df
)


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.metric(
        "Average Forecast",
        f"{average_forecast:,.0f}"
    )


with k2:

    st.metric(
        "Peak Forecast",
        f"{maximum_forecast:,.0f}"
    )


with k3:

    st.metric(
        "Minimum Forecast",
        f"{minimum_forecast:,.0f}"
    )


with k4:

    st.metric(
        "Forecast Records",
        f"{record_count:,}"
    )


# ============================================================
# SECTION 02
# ACTUAL VS SARIMA
# ============================================================

st.divider()

st.caption(
    "02 / FORECAST VALIDATION"
)

st.subheader(
    "Actual Load vs SARIMA Forecast"
)

st.write(
    "This comparison evaluates how the SARIMA forecast "
    "tracks the observed system-load series."
)


validation_df = forecast_df.dropna(
    subset=[
        ACTUAL_COL,
        FORECAST_COL
    ]
).copy()


if not validation_df.empty:

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=validation_df[DATE_COL],
            y=validation_df[ACTUAL_COL],
            mode="lines",
            name="Actual Load",
            line=dict(
                width=2.5
            )
        )
    )


    fig.add_trace(
        go.Scatter(
            x=validation_df[DATE_COL],
            y=validation_df[FORECAST_COL],
            mode="lines",
            name="SARIMA Forecast",
            line=dict(
                width=2.5,
                dash="dash"
            )
        )
    )


    fig.update_layout(
        title="Observed vs SARIMA Forecast",
        xaxis_title="Date",
        yaxis_title="System Load"
    )


    style_chart(
        fig,
        470
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )

else:

    st.info(
        "Actual Load values are not available for the "
        "forecast-validation comparison."
    )


# ============================================================
# SECTION 03
# FORECAST ERROR
# ============================================================

st.divider()

st.caption(
    "03 / FORECAST ACCURACY"
)

st.subheader(
    "SARIMA Forecast Error"
)

st.write(
    "Forecast error measures the difference between the "
    "observed system load and SARIMA's predicted load."
)


error_df = forecast_df.dropna(
    subset=[
        ERROR_COL
    ]
).copy()


if not error_df.empty:

    mean_error = error_df[
        ERROR_COL
    ].mean()

    mean_absolute_error = error_df[
        ERROR_COL
    ].abs().mean()

    maximum_error = error_df[
        ERROR_COL
    ].abs().max()


    e1, e2, e3 = st.columns(3)


    with e1:

        st.metric(
            "Mean Forecast Error",
            f"{mean_error:,.2f}"
        )


    with e2:

        st.metric(
            "Mean Absolute Error",
            f"{mean_absolute_error:,.2f}"
        )


    with e3:

        st.metric(
            "Maximum Absolute Error",
            f"{maximum_error:,.2f}"
        )


    error_fig = go.Figure()


    error_fig.add_trace(
        go.Bar(
            x=error_df[DATE_COL],
            y=error_df[ERROR_COL],
            name="Forecast Error"
        )
    )


    error_fig.add_hline(
        y=0,
        line_dash="dash"
    )


    error_fig.update_layout(
        title="SARIMA Forecast Error Over Time",
        xaxis_title="Date",
        yaxis_title="Forecast Error"
    )


    style_chart(
        error_fig,
        380
    )


    st.plotly_chart(
        error_fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )

else:

    st.info(
        "Forecast error values are not available."
    )


# ============================================================
# SECTION 04
# SARIMA PERFORMANCE METRICS
# ============================================================

st.divider()

st.caption(
    "04 / SARIMA MODEL PERFORMANCE"
)

st.subheader(
    "Time-Series Model Evaluation"
)

st.write(
    "These metrics evaluate the SARIMA forecasting model. "
    "General machine-learning model benchmarking is handled "
    "on the Model Performance page."
)


if not sarima_results.empty:

    st.dataframe(
        sarima_results,
        use_container_width=True,
        hide_index=True
    )


    sarima_mae_col = find_column(
        sarima_results,
        [
            "MAE",
            "mae",
            "Mean Absolute Error"
        ]
    )


    sarima_rmse_col = find_column(
        sarima_results,
        [
            "RMSE",
            "rmse",
            "Root Mean Squared Error"
        ]
    )


    metric_columns = []

    if sarima_mae_col:
        metric_columns.append(
            ("SARIMA MAE", sarima_mae_col)
        )

    if sarima_rmse_col:
        metric_columns.append(
            ("SARIMA RMSE", sarima_rmse_col)
        )


    if metric_columns:

        metric_cols = st.columns(
            len(metric_columns)
        )


        for ui_col, (
            label,
            source_col
        ) in zip(
            metric_cols,
            metric_columns
        ):

            value = numeric_value(
                sarima_results,
                source_col
            )

            with ui_col:

                if value is not None:

                    st.metric(
                        label,
                        f"{value:,.2f}"
                    )

                else:

                    st.metric(
                        label,
                        "N/A"
                    )

else:

    st.warning(
        "No SARIMA evaluation report was found."
    )


# ============================================================
# SECTION 05
# FUTURE 30-DAY FORECAST
# ============================================================

st.divider()

st.caption(
    "05 / FUTURE FORECAST"
)

st.subheader(
    "30-Day Forward System Load Projection"
)

st.write(
    "A genuine future projection is shown here only when "
    "the saved SARIMA forecasting model can generate it. "
    "Validation observations are not mislabeled as future data."
)


future_df = pd.DataFrame()


# ============================================================
# TRY TO LOAD SAVED SARIMA MODEL
# ============================================================

if SARIMA_MODEL_FILE.exists():

    try:

        sarima_model = joblib.load(
            SARIMA_MODEL_FILE
        )


        # ----------------------------------------------------
        # Statsmodels SARIMAX results
        # ----------------------------------------------------

        if hasattr(
            sarima_model,
            "get_forecast"
        ):

            future_result = sarima_model.get_forecast(
                steps=30
            )


            summary_frame = (
                future_result
                .summary_frame()
                .reset_index()
            )


            if not summary_frame.empty:

                date_candidate = summary_frame.columns[0]

                mean_candidate = find_column(
                    summary_frame,
                    [
                        "mean",
                        "Mean",
                        "predicted_mean"
                    ]
                )


                if mean_candidate is not None:

                    future_df = pd.DataFrame({

                        "Date":
                            pd.to_datetime(
                                summary_frame[
                                    date_candidate
                                ],
                                errors="coerce"
                            ),

                        "Forecast Load":
                            pd.to_numeric(
                                summary_frame[
                                    mean_candidate
                                ],
                                errors="coerce"
                            )
                    })


                    lower_candidate = find_column(
                        summary_frame,
                        [
                            "mean_ci_lower",
                            "lower",
                            "Lower"
                        ]
                    )


                    upper_candidate = find_column(
                        summary_frame,
                        [
                            "mean_ci_upper",
                            "upper",
                            "Upper"
                        ]
                    )


                    if lower_candidate:

                        future_df[
                            "Lower Bound"
                        ] = pd.to_numeric(
                            summary_frame[
                                lower_candidate
                            ],
                            errors="coerce"
                        )


                    if upper_candidate:

                        future_df[
                            "Upper Bound"
                        ] = pd.to_numeric(
                            summary_frame[
                                upper_candidate
                            ],
                            errors="coerce"
                        )


                    future_df = future_df.dropna(
                        subset=[
                            "Date",
                            "Forecast Load"
                        ]
                    )


        # ----------------------------------------------------
        # Generic forecast method fallback
        # ----------------------------------------------------

        elif hasattr(
            sarima_model,
            "forecast"
        ):

            values = sarima_model.forecast(
                steps=30
            )


            if hasattr(
                values,
                "to_numpy"
            ):

                values = values.to_numpy()


            values = np.asarray(
                values,
                dtype=float
            ).reshape(-1)


            if len(values) > 0:

                last_date = forecast_df[
                    DATE_COL
                ].max()


                future_dates = pd.date_range(
                    start=last_date
                    + pd.Timedelta(days=1),
                    periods=len(values),
                    freq="D"
                )


                future_df = pd.DataFrame({

                    "Date":
                        future_dates,

                    "Forecast Load":
                        values

                })


    except Exception as error:

        st.info(
            "The saved SARIMA model was found, but a new "
            "30-day projection could not be generated from "
            "the saved model. The validated forecast output "
            "above remains available."
        )


# ============================================================
# DISPLAY FUTURE FORECAST
# ============================================================

if not future_df.empty:

    future_df = future_df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )


    f1, f2, f3, f4 = st.columns(4)


    future_average = future_df[
        "Forecast Load"
    ].mean()

    future_max = future_df[
        "Forecast Load"
    ].max()

    future_min = future_df[
        "Forecast Load"
    ].min()

    peak_index = future_df[
        "Forecast Load"
    ].idxmax()

    peak_date = future_df.loc[
        peak_index,
        "Date"
    ]


    with f1:

        st.metric(
            "Average Future Load",
            f"{future_average:,.0f}"
        )


    with f2:

        st.metric(
            "Projected Peak",
            f"{future_max:,.0f}"
        )


    with f3:

        st.metric(
            "Projected Minimum",
            f"{future_min:,.0f}"
        )


    with f4:

        st.metric(
            "Peak Date",
            peak_date.strftime(
                "%d %b %Y"
            )
        )


    # --------------------------------------------------------
    # FUTURE CHART
    # --------------------------------------------------------

    future_fig = go.Figure()


    future_fig.add_trace(
        go.Scatter(
            x=future_df["Date"],
            y=future_df["Forecast Load"],
            mode="lines+markers",
            name="Future Forecast",
            line=dict(
                width=3
            ),
            marker=dict(
                size=5
            )
        )
    )


    # --------------------------------------------------------
    # CONFIDENCE INTERVAL
    # --------------------------------------------------------

    if (
        "Lower Bound" in future_df.columns
        and
        "Upper Bound" in future_df.columns
    ):

        future_fig.add_trace(
            go.Scatter(
                x=future_df["Date"],
                y=future_df["Upper Bound"],
                mode="lines",
                line=dict(
                    width=0
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )


        future_fig.add_trace(
            go.Scatter(
                x=future_df["Date"],
                y=future_df["Lower Bound"],
                mode="lines",
                fill="tonexty",
                line=dict(
                    width=0
                ),
                name="Forecast Interval",
                hoverinfo="skip"
            )
        )


    future_fig.update_layout(
        title="30-Day Future System Load Projection",
        xaxis_title="Date",
        yaxis_title="Forecasted System Load"
    )


    style_chart(
        future_fig,
        460
    )


    st.plotly_chart(
        future_fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    # --------------------------------------------------------
    # FUTURE FORECAST TABLE
    # --------------------------------------------------------

    st.subheader(
        "Future Forecast Register"
    )


    future_table = future_df.copy()


    future_table["Date"] = future_table[
        "Date"
    ].dt.strftime(
        "%d %b %Y"
    )


    future_table["Forecast Load"] = (
        future_table[
            "Forecast Load"
        ].round(0)
    )


    if "Lower Bound" in future_table.columns:

        future_table[
            "Lower Bound"
        ] = future_table[
            "Lower Bound"
        ].round(0)


    if "Upper Bound" in future_table.columns:

        future_table[
            "Upper Bound"
        ] = future_table[
            "Upper Bound"
        ].round(0)


    st.dataframe(
        future_table,
        use_container_width=True,
        hide_index=True
    )


else:

    st.info(
        "A separate future 30-day projection could not be "
        "generated from the saved SARIMA model. The available "
        "SARIMA validation forecast above is still displayed."
    )


# ============================================================
# SECTION 06
# FORECAST DIRECTION
# ============================================================

st.divider()

st.caption(
    "06 / OPERATIONAL INTERPRETATION"
)

st.subheader(
    "Forecast Signal"
)


# Use future forecast when available.
# Otherwise use the available forecast series.

if not future_df.empty:

    signal_series = future_df[
        "Forecast Load"
    ]

    signal_dates = future_df[
        "Date"
    ]

else:

    signal_series = forecast_df[
        FORECAST_COL
    ]

    signal_dates = forecast_df[
        DATE_COL
    ]


if len(signal_series) >= 2:

    start_value = float(
        signal_series.iloc[0]
    )

    end_value = float(
        signal_series.iloc[-1]
    )

    absolute_change = (
        end_value
        -
        start_value
    )


    if start_value != 0:

        percentage_change = (
            absolute_change
            /
            abs(start_value)
        ) * 100

    else:

        percentage_change = 0


    if percentage_change > 5:

        direction = "Increasing"

    elif percentage_change < -5:

        direction = "Decreasing"

    else:

        direction = "Relatively Stable"


    s1, s2, s3 = st.columns(3)


    with s1:

        st.metric(
            "Forecast Direction",
            direction
        )


    with s2:

        st.metric(
            "Beginning Load",
            f"{start_value:,.0f}"
        )


    with s3:

        st.metric(
            "Ending Load",
            f"{end_value:,.0f}",
            f"{percentage_change:+.1f}%"
        )


    if direction == "Increasing":

        st.warning(
            "The forecast indicates an increasing system-load "
            "trajectory. This may warrant closer monitoring of "
            "capacity and operational pressure."
        )

    elif direction == "Decreasing":

        st.success(
            "The forecast indicates a decreasing system-load "
            "trajectory, which may indicate easing operational "
            "pressure."
        )

    else:

        st.info(
            "The forecast remains relatively stable across "
            "the available horizon."
        )


# ============================================================
# SECTION 07
# FORECAST INTERPRETATION
# ============================================================

st.divider()

st.caption(
    "07 / DECISION SUPPORT"
)

st.subheader(
    "How to Use the Forecast"
)


with st.container(border=True):

    st.write(
        "**Time-series forecasting**  \n"
        "SARIMA uses historical temporal patterns in system "
        "load to estimate future behaviour."
    )

    st.write(
        "**Forecast validation**  \n"
        "Actual Load, Forecast Load and Forecast Error are "
        "used to evaluate the quality of the forecasting output."
    )

    st.write(
        "**Capacity planning**  \n"
        "Forecast trends can support forward-looking capacity "
        "assessment and resource planning."
    )

    st.write(
        "**Operational caution**  \n"
        "Forecast values are estimates and should be interpreted "
        "together with current operational conditions and "
        "data-quality information."
    )


# ============================================================
# SECTION 08
# FORECAST DATA REGISTER
# ============================================================

st.divider()

st.caption(
    "08 / DATA LINEAGE"
)

st.subheader(
    "Forecast Data Sources"
)

st.write(
    "The dashboard uses project-generated outputs. "
    "Local computer paths are intentionally hidden from "
    "the user-facing interface."
)


source1, source2, source3 = st.columns(3)


with source1:

    st.info(
        "**Forecast Output**\n\n"
        "`sarima_forecast.csv`\n\n"
        "Actual load, SARIMA forecast and forecast error."
    )


with source2:

    st.info(
        "**Model Evaluation**\n\n"
        "`forecast_results.csv`\n\n"
        "SARIMA forecasting performance metrics."
    )


with source3:

    if SARIMA_MODEL_FILE.exists():

        st.success(
            "**Forecasting Model**\n\n"
            "`sarima_forecasting_model.pkl`\n\n"
            "Saved SARIMA model available."
        )

    else:

        st.warning(
            "**Forecasting Model**\n\n"
            "`sarima_forecasting_model.pkl`\n\n"
            "Saved model not available."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UAC-CareAI • Forward Planning Engine • "
    "SARIMA Time-Series Forecasting • Capacity Planning"
)