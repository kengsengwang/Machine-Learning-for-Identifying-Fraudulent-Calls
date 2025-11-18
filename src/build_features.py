# src/build_features.py
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .utils import get_logger, ensure_dir

# default paths (relative to repo root)
DATA_DB = Path("data") / "calls.db"
OUTPUT_CSV = Path("data") / "processed_calls.csv"


def _detect_table_name(conn: sqlite3.Connection) -> str:
    """
    Safely detect the first table name in the SQLite DB.
    This avoids hard-coding the table name.
    Uses plain sqlite3 (no pandas SQL helpers).
    """
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = cursor.fetchall()
    if not rows:
        raise RuntimeError("No tables found in SQLite database.")
    # first column of first row = table name
    return rows[0][0]


def _load_table_as_dataframe(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """
    Load the entire table into a pandas DataFrame using sqlite3 only.
    """
    # get column names
    col_cursor = conn.execute(f"PRAGMA table_info({table_name});")
    cols_info = col_cursor.fetchall()
    if not cols_info:
        raise RuntimeError(f"No columns found for table {table_name}.")
    column_names = [c[1] for c in cols_info]  # second field is column name

    # get all rows
    data_cursor = conn.execute(f"SELECT * FROM {table_name};")
    rows = data_cursor.fetchall()

    df = pd.DataFrame.from_records(rows, columns=column_names)
    return df


def load_raw_calls(db_path: Path = DATA_DB) -> pd.DataFrame:
    """
    Load the raw calls table from the SQLite database.

    Expects the SQLite file to be at data/calls.db (as required by AIAP).
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {db_path}. "
            "Make sure you created the 'data' folder and placed calls.db inside."
        )

    conn = sqlite3.connect(db_path)
    try:
        table_name = _detect_table_name(conn)
        df = _load_table_as_dataframe(conn, table_name)
    finally:
        conn.close()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic feature engineering for the fraudulent calls dataset.

    - Extract hour, day of week, and weekend flag from Timestamp
    - Create a numeric target column 'scam_label' from 'Scam Call'
    """
    df = df.copy()

    # Timestamp-based features
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["call_hour"] = ts.dt.hour
        df["call_dayofweek"] = ts.dt.dayofweek
        df["call_is_weekend"] = ts.dt.dayofweek >= 5

    # Binary target: 1 for Scam, 0 for Not Scam
    if "Scam Call" not in df.columns:
        raise KeyError(
            "Expected a 'Scam Call' column in the dataset. "
            "Please check the column names in calls.db."
        )

    # Handle typical labels "Scam" / "Not Scam"; if already 0/1, keep as is
    if df["Scam Call"].dtype == "O":
        df["scam_label"] = df["Scam Call"].map(
            {"Scam": 1, "Not Scam": 0}
        )
    else:
        df["scam_label"] = df["Scam Call"]

    return df


def build_and_save_dataset(
    db_path: Path = DATA_DB, out_csv: Path = OUTPUT_CSV
) -> None:
    """Full step: load from SQLite, engineer features, write to CSV."""
    logger = get_logger("build_features")

    logger.info(f"Loading raw data from {db_path} ...")
    raw_df = load_raw_calls(db_path)
    logger.info(f"Loaded {len(raw_df)} rows with {raw_df.shape[1]} columns.")

    logger.info("Applying feature engineering ...")
    feat_df = engineer_features(raw_df)
    ensure_dir(out_csv.parent)
    feat_df.to_csv(out_csv, index=False)

    logger.info(
        f"Saved processed dataset to {out_csv} "
        f"with shape {feat_df.shape[0]} rows x {feat_df.shape[1]} columns."
    )


def main() -> None:
    build_and_save_dataset()


if __name__ == "__main__":
    main()
