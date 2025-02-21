# file_loading.py

import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_and_parse_file(file_bytes: bytes) -> pd.DataFrame:
    """
    Reads raw bytes from a file, decodes them, and parses numeric data.
    Returns a DataFrame with columns for Ping, Depth, and other values.
    """
    try:
        file_str = file_bytes.decode("utf-8", errors="ignore")
        lines = file_str.splitlines()
        pattern = re.compile(r"[-+]?\d*\.\d+|\d+")
        all_data = [pattern.findall(line) for line in lines if len(pattern.findall(line)) == 16]
        if not all_data:
            logger.error("No valid data found in file.")
            return pd.DataFrame()  # Return an empty DataFrame if no data is found
        columns = [
            "Ping", "Depth", "Col2", "Col3", "Col4", "Col5",
            "Col6", "Col7", "Col8", "Col9", "TS", "Col11",
            "Col12", "Col13", "Col14", "Col15"
        ]
        df = pd.DataFrame(all_data, columns=columns)
        df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_values(by=["Ping"]).reset_index(drop=True)
        logger.info("Data loaded successfully with %d rows.", len(df))
        return df
    except Exception as e:
        logger.exception("Error in load_and_parse_file: %s", e)
        return pd.DataFrame()

def analyze_target_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the target strength (TS) of fish data.
    Calculates TS change and a rolling average.
    """
    try:
        df = df.sort_values(by="Ping").reset_index(drop=True)
        df["TS Change (dB)"] = df["TS"].diff()
        df["TS Rolling Avg (dB)"] = df["TS"].rolling(window=5, min_periods=1).mean()
        logger.info("Target strength analysis completed.")
        return df
    except Exception as e:
        logger.exception("Error in analyze_target_strength: %s", e)
        return df

def compute_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-track features like the number of points, average TS, and average speed.
    Only processes rows where 'IsFish' equals 1.
    """
    try:
        fish_df = df[df["IsFish"] == 1].copy()
        if fish_df.empty:
            return pd.DataFrame(columns=["TrackID", "NumPoints", "AvgTS", "AvgSpeed"])
        if "DepthRateKF" not in fish_df.columns:
            fish_df["DepthRateKF"] = 0.0
        grouped = fish_df.groupby("TrackID")
        summary = grouped.agg(
            NumPoints=("Ping", "count"),
            AvgTS=("TS", "mean"),
            AvgSpeed=("DepthRateKF", lambda x: x.abs().mean())
        ).reset_index()
        logger.info("Track features computed successfully.")
        return summary
    except Exception as e:
        logger.exception("Error in compute_track_features: %s", e)
        return pd.DataFrame()
