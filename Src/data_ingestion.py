import pandas as pd
from pathlib import Path


# ============================================================
# UAC CARE ANALYTICS
# DATA INGESTION
# ============================================================

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Company-provided dataset
RAW_FILE = (
    BASE_DIR
    / "Data"
    / "raw"
    / "HHS_Unaccompanied_Alien_Children_Program.csv"
)


def load_data():

    # Check whether dataset exists
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{RAW_FILE}"
        )

    # Load dataset
    df = pd.read_csv(RAW_FILE)

    print("\n" + "=" * 60)
    print("UAC CARE ANALYTICS - DATA INGESTION")
    print("=" * 60)

    print("\nDataset loaded successfully!")

    print("\nDataset Shape:")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names:")

    for column in df.columns:
        print(f" - {column}")

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()