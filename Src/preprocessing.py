import pandas as pd
from pathlib import Path


# ============================================================
# UAC CARE ANALYTICS
# DATA CLEANING & PREPROCESSING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    BASE_DIR
    / "Data"
    / "raw"
    / "HHS_Unaccompanied_Alien_Children_Program.csv"
)

PROCESSED_DIR = (
    BASE_DIR
    / "Data"
    / "Processed"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "uac_cleaned.csv"
)


# ============================================================
# Columns
# ============================================================

NUMERIC_COLUMNS = [
    "Children apprehended and placed in CBP custody*",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children in HHS Care",
    "Children discharged from HHS Care"
]


def clean_data():

    print("\n" + "=" * 70)
    print("UAC CARE ANALYTICS - DATA PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load original dataset
    # --------------------------------------------------------

    df = pd.read_csv(RAW_FILE)

    print("\nOriginal dataset:")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # --------------------------------------------------------
    # 2. Remove completely empty rows
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        how="all"
    ).copy()

    removed_empty = before - len(df)

    print("\nCompletely empty rows removed:", removed_empty)

    # --------------------------------------------------------
    # 3. Convert Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    invalid_dates = df["Date"].isna().sum()

    print("Invalid dates:", invalid_dates)

    # Remove rows with invalid dates
    df = df.dropna(
        subset=["Date"]
    ).copy()

    # --------------------------------------------------------
    # 4. Convert numeric columns
    # --------------------------------------------------------

    print("\nConverting numeric columns...")

    for column in NUMERIC_COLUMNS:

        # Remove commas
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        # Convert to numeric
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # 5. Missing numeric values
    # --------------------------------------------------------

    print("\nMissing values after conversion:")

    print(
        df[NUMERIC_COLUMNS]
        .isna()
        .sum()
    )

    # --------------------------------------------------------
    # 6. Remove duplicate dates
    # --------------------------------------------------------

    duplicate_dates = df["Date"].duplicated().sum()

    print(
        "\nDuplicate dates found:",
        duplicate_dates
    )

    df = df.drop_duplicates(
        subset=["Date"],
        keep="first"
    )

    # --------------------------------------------------------
    # 7. Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 8. Data validation flags
    # --------------------------------------------------------

    df["Transfer Validation Flag"] = (
        df["Children transferred out of CBP custody"]
        >
        df["Children in CBP custody"]
    )

    df["Discharge Validation Flag"] = (
        df["Children discharged from HHS Care"]
        >
        df["Children in HHS Care"]
    )

    # --------------------------------------------------------
    # 9. Negative-value validation
    # --------------------------------------------------------

    df["Negative Value Flag"] = (
        df[NUMERIC_COLUMNS]
        .lt(0)
        .any(axis=1)
    )

    # --------------------------------------------------------
    # 10. Create processed folder
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 11. Save cleaned dataset
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 12. Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    print("\nFinal dataset:")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nFinal data types:")
    print(df.dtypes)

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nValidation summary:")

    print(
        "Transfer validation flags:",
        df["Transfer Validation Flag"].sum()
    )

    print(
        "Discharge validation flags:",
        df["Discharge Validation Flag"].sum()
    )

    print(
        "Negative value flags:",
        df["Negative Value Flag"].sum()
    )

    print("\n" + "=" * 70)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    clean_data()