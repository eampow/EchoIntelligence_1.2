# sonar_app/filters.py

import numpy as np
import pandas as pd
import logging
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

logger = logging.getLogger(__name__)

def fx(x, dt):
    """State transition function: x = [depth, depth_rate]."""
    return np.array([x[0] + dt*x[1], x[1]])

def hx(x):
    """Nonlinear measurement function: measurement = sqrt(depth)."""
    return np.array([np.sqrt(max(x[0], 1e-3))])

class EnhancedUKFFilter:
    """
    Enhanced filter using the Unscented Kalman Filter (UKF) for depth data.
    """
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2., kappa=0)
        self.ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=dt,
                                         fx=fx, hx=hx, points=self.points)
        self.ukf.x = np.array([0.0, 0.0])
        self.ukf.P *= 1.0
        self.ukf.Q = np.eye(2) * 0.1
        self.ukf.R = np.array([[0.5]])

    def filter(self, df: pd.DataFrame, depth_col: str = "Depth") -> pd.DataFrame:
        if df.empty:
            df["DepthUKF"] = []
            df["DepthRateUKF"] = []
            return df

        depth_est, depth_rate_est = [], []
        for i in range(len(df)):
            # Could adapt Q/R based on innovation here
            self.ukf.predict()
            z = np.array([np.sqrt(max(df.iloc[i][depth_col], 1e-3))])
            self.ukf.update(z)
            depth_est.append(self.ukf.x[0])
            depth_rate_est.append(self.ukf.x[1])
        df["DepthUKF"] = depth_est
        df["DepthRateUKF"] = depth_rate_est
        # Create legacy columns for compatibility
        df["DepthKF"] = df["DepthUKF"]
        df["DepthRateKF"] = df["DepthRateUKF"]
        logger.info("UKF filtering completed on %d rows.", len(df))
        return df
