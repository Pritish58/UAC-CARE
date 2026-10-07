import streamlit as st
import pandas as pd
from pathlib import Path


# This file lives in dashboard/, so parents[1] is the repository root.
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "Data" / "Processed"
REPORT_DIR = BASE_DIR / "reports"


def _find_file(*names):
    """Find a file case-insensitively inside Data/Processed."""
    if not PROCESSED_DIR.exists():
        return None

    lookup = {
        p.name.lower(): p
        for p in PROCESSED_DIR.iterdir()
        if p.is_file()
    }

    for name in names:
        path = lookup.get(name.lower())
        if path is not None:
            return path

    return None


def _read_csv(filename, parse_dates=True):
    """Read a project CSV with encoding/parser fallbacks."""
    path = _find_file(filename)

    if path is None:
        raise FileNotFoundError(
            f"Missing file: {PROCESSED_DIR / filename}"
        )

    errors = []

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
            errors.append(
                f"{encoding}: {type(exc).__name__}: {exc}"
            )

    # Tolerate irregular CSV rows as a final fallback.
    try:
        df = pd.read_csv(
            path,
            encoding="latin-1",
            engine="python",
            on_bad_lines="skip"
        )

        if parse_dates and "Date" in df.columns:
            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

        return df

    except Exception as exc:
        errors.append(
            f"python-engine: {type(exc).__name__}: {exc}"
        )

    raise RuntimeError(
        f"Could not read {path}. " + " | ".join(errors)
    )


@st.cache_data(show_spinner=False)
def load_main_data():
    """Prefer engineered features; fall back to cleaned data."""
    for filename in ("uac_features.csv", "uac_cleaned.csv"):
        path = _find_file(filename)
        if path is None:
            continue

        df = _read_csv(filename)

        if not df.empty:
            return df

    raise RuntimeError(
        "Neither uac_features.csv nor uac_cleaned.csv "
        "could be loaded as a non-empty dataframe."
    )


@st.cache_data(show_spinner=False)
def load_anomaly_data():
    return _read_csv("uac_anomaly_results.csv")


@st.cache_data(show_spinner=False)
def load_cluster_data():
    return _read_csv("uac_operational_clusters.csv")


@st.cache_data(show_spinner=False)
def load_stress_data():
    return _read_csv("uac_stress_labels.csv")
