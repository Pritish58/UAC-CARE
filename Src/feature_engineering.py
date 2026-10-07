import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# UAC CARE ANALYTICS
# ADVANCED FEATURE ENGINEERING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "Data"
    / "Processed"
    / "uac_cleaned.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "Data"
    / "Processed"
    / "uac_features.csv"
)


def create_features():

    print("\n" + "=" * 70)
    print("UAC CARE ANALYTICS - FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load cleaned data
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    print("\nCleaned dataset loaded.")
    print("Rows:", len(df))

    # ========================================================
    # 2. CORE SYSTEM METRICS
    # ========================================================

    # Total children under CBP + HHS responsibility
    df["Total System Load"] = (
        df["Children in CBP custody"]
        + df["Children in HHS Care"]
    )

    # Difference between HHS inflow and outflow
    df["Net Intake Pressure"] = (
        df["Children transferred out of CBP custody"]
        - df["Children discharged from HHS Care"]
    )

    # ========================================================
    # 3. FLOW RATIOS
    # ========================================================

    # Avoid division by zero
    transfers = (
        df["Children transferred out of CBP custody"]
        .replace(0, np.nan)
    )

    df["Discharge Offset Ratio"] = (
        df["Children discharged from HHS Care"]
        / transfers
    )

    df["Transfer to CBP Ratio"] = (
        df["Children transferred out of CBP custody"]
        /
        df["Children in CBP custody"].replace(0, np.nan)
    )

    # ========================================================
    # 4. LOAD CHANGE
    # ========================================================

    df["Daily Load Change"] = (
        df["Total System Load"].diff()
    )

    df["Daily Load Growth %"] = (
        df["Total System Load"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    # ========================================================
    # 5. LAG FEATURES
    # ========================================================

    lag_periods = [1, 3, 7, 14]

    for lag in lag_periods:

        df[f"Load Lag {lag}"] = (
            df["Total System Load"].shift(lag)
        )

        df[f"Net Intake Lag {lag}"] = (
            df["Net Intake Pressure"].shift(lag)
        )

        df[f"HHS Load Lag {lag}"] = (
            df["Children in HHS Care"].shift(lag)
        )

    # ========================================================
    # 6. ROLLING FEATURES
    # ========================================================

    windows = [7, 14, 30]

    for window in windows:

        df[f"Load Rolling Mean {window}"] = (
            df["Total System Load"]
            .rolling(window)
            .mean()
        )

        df[f"Load Rolling Std {window}"] = (
            df["Total System Load"]
            .rolling(window)
            .std()
        )

        df[f"Net Intake Rolling Mean {window}"] = (
            df["Net Intake Pressure"]
            .rolling(window)
            .mean()
        )

    # ========================================================
    # 7. BACKLOG FEATURES
    # ========================================================

    df["7-Day Net Intake"] = (
        df["Net Intake Pressure"]
        .rolling(7)
        .sum()
    )

    df["14-Day Net Intake"] = (
        df["Net Intake Pressure"]
        .rolling(14)
        .sum()
    )

    # Positive sustained intake = accumulation pressure
    df["Backlog Indicator"] = (
        df["7-Day Net Intake"] > 0
    ).astype(int)

    # ========================================================
    # 8. VOLATILITY
    # ========================================================

    df["Load Volatility 7D"] = (
        df["Total System Load"]
        .rolling(7)
        .std()
    )

    df["Load Volatility 14D"] = (
        df["Total System Load"]
        .rolling(14)
        .std()
    )

    # ========================================================
    # 9. CALENDAR FEATURES
    # ========================================================

    df["Day"] = df["Date"].dt.day

    df["Day of Week"] = (
        df["Date"].dt.dayofweek
    )

    df["Week of Year"] = (
        df["Date"].dt.isocalendar()
        .week
        .astype(int)
    )

    df["Month"] = (
        df["Date"].dt.month
    )

    df["Quarter"] = (
        df["Date"].dt.quarter
    )

    df["Year"] = (
        df["Date"].dt.year
    )

    # Weekend indicator
    df["Is Weekend"] = (
        df["Day of Week"] >= 5
    ).astype(int)

    # ========================================================
    # 10. FUTURE TARGET
    # ========================================================

    # Predict next observed day's total system load
    df["Next Day System Load"] = (
        df["Total System Load"].shift(-1)
    )

    # ========================================================
    # 11. REMOVE EXTREME INFINITE VALUES
    # ========================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ========================================================
    # 12. SAVE FEATURE DATASET
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature engineering completed.")

    print("\nFinal shape:")
    print(df.shape)

    print("\nNew features created:")

    original_columns = [
        "Date",
        "Children apprehended and placed in CBP custody*",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care"
    ]

    new_features = [
        column
        for column in df.columns
        if column not in original_columns
    ]

    for feature in new_features:
        print(" -", feature)

    print("\nSaved to:")
    print(OUTPUT_FILE)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_features()