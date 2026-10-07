# import streamlit as st
# import pandas as pd
# from pathlib import Path


# BASE_DIR = Path(__file__).resolve().parent.parent

# PROCESSED_DIR = BASE_DIR / "Data" / "Processed"
# REPORT_DIR = BASE_DIR / "reports"


# @st.cache_data
# def load_main_data():
#     return pd.read_csv(
#         PROCESSED_DIR / "uac_features.csv",
#         parse_dates=["Date"]
#     )


# @st.cache_data
# def load_anomaly_data():
#     return pd.read_csv(
#         PROCESSED_DIR / "uac_anomaly_results.csv",
#         parse_dates=["Date"]
#     )


# @st.cache_data
# def load_cluster_data():
#     return pd.read_csv(
#         PROCESSED_DIR / "uac_operational_clusters.csv",
#         parse_dates=["Date"]
#     )


# @st.cache_data
# def load_stress_data():
#     return pd.read_csv(
#         PROCESSED_DIR / "uac_stress_labels.csv",
#         parse_dates=["Date"]
#     )


import streamlit as st
import pandas as pd
from pathlib import Path


# data_loader.py lives in dashboard/, so parents[1] is the repository root.
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "Data" / "Processed"
REPORT_DIR = BASE_DIR / "reports"


def _find_file(*names):
    """Find a processed file using case-insensitive filename matching."""
    if not PROCESSED_DIR.exists():
        return None

    files = {
        p.name.lower(): p
        for p in PROCESSED_DIR.iterdir()
        if p.is_file()
    }

    for name in names:
        path = files.get(name.lower())
        if path is not None:
            return path

    return None


@st.cache_data(show_spinner=False)
def _read_csv(filename, parse_dates=True):
    path = _find_file(filename)

    if path is None:
        raise FileNotFoundError(
            f"File not found: {PROCESSED_DIR / filename}"
        )

    last_error = None

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding)

            if parse_dates and "Date" in df.columns:
                df["Date"] = pd.to_datetime(
                    df["Date"],
                    errors="coerce"
                )

            return df

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not read {path}: "
        f"{type(last_error).__name__}: {last_error}"
    )


@st.cache_data(show_spinner=False)
def load_main_data():
    # Prefer the engineered dataset; fall back to cleaned data.
    if _find_file("uac_features.csv") is not None:
        return _read_csv("uac_features.csv")

    return _read_csv("uac_cleaned.csv")


@st.cache_data(show_spinner=False)
def load_anomaly_data():
    return _read_csv("uac_anomaly_results.csv")


@st.cache_data(show_spinner=False)
def load_cluster_data():
    return _read_csv("uac_operational_clusters.csv")


@st.cache_data(show_spinner=False)
def load_stress_data():
    return _read_csv("uac_stress_labels.csv")
