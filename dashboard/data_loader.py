import streamlit as st
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "Data" / "Processed"
REPORT_DIR = BASE_DIR / "reports"


@st.cache_data
def load_main_data():
    return pd.read_csv(
        PROCESSED_DIR / "uac_features.csv",
        parse_dates=["Date"]
    )


@st.cache_data
def load_anomaly_data():
    return pd.read_csv(
        PROCESSED_DIR / "uac_anomaly_results.csv",
        parse_dates=["Date"]
    )


@st.cache_data
def load_cluster_data():
    return pd.read_csv(
        PROCESSED_DIR / "uac_operational_clusters.csv",
        parse_dates=["Date"]
    )


@st.cache_data
def load_stress_data():
    return pd.read_csv(
        PROCESSED_DIR / "uac_stress_labels.csv",
        parse_dates=["Date"]
    )