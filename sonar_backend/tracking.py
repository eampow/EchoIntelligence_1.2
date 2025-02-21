# sonar_app/tracking.py

import pandas as pd
import numpy as np
import logging
from sklearn.cluster import DBSCAN
from config import Config

logger = logging.getLogger(__name__)

class DBSCANFishTracker:
    """
    Tracks fish echoes using DBSCAN clustering over [Ping, Depth, TS].
    Echoes belonging to clusters with fewer than min_track_length points are ignored.
    """
    def __init__(self,
                 eps: float = Config.DBSCAN_EPS,
                 min_samples: int = Config.DBSCAN_MIN_SAMPLES,
                 min_track_length: int = Config.DBSCAN_MIN_TRACK_LENGTH):
        self.eps = eps
        self.min_samples = min_samples
        self.min_track_length = min_track_length

    def track(self, df: pd.DataFrame) -> pd.DataFrame:
        if not {"Ping", "Depth", "TS"}.issubset(df.columns):
            logger.error("Required columns for tracking missing.")
            return df
        if df.empty:
            return df

        X = df[["Ping", "Depth", "TS"]].to_numpy()
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        df["TrackID"] = clustering.labels_
        # Mark clusters that have too few points as non-fish (label = -1)
        track_counts = df["TrackID"].value_counts().to_dict()
        df["IsFish"] = df["TrackID"].apply(
            lambda tid: 1 if tid != -1 and track_counts.get(tid, 0) >= self.min_track_length else 0
        )
        logger.info("DBSCAN tracking completed. Found %d clusters.", len(track_counts))
        return df
