# sonar_app/config.py

import os
from typing import List, Dict, Optional
import numpy as np

class Config:
    """
    Global configuration and constants.
    """

    # Random seeds
    SEED = 42
    RNG = np.random.default_rng(SEED)

    # Classifier and filter selections
    SELECTED_CLASSIFIER: str = "blended"  # Options: "random_forest", "deep", "blended"
    SELECTED_FILTER: str = "ukf"

    # DBSCAN parameters
    DBSCAN_EPS = 3.0
    DBSCAN_MIN_SAMPLES = 3
    DBSCAN_MIN_TRACK_LENGTH = 2

    # PyQt Dark Theme
    DARK_QSS: str = """
    QMainWindow { background-color: #121212; }
    QMenuBar { background-color: #1f1f1f; color: #ffffff; }
    QMenuBar::item:selected { background-color: #333333; }
    """

    # Zoom instructions for plotting (set via user queries)
    AI_INSTRUCTIONS: Dict[str, Optional[List[float]]] = {
        "zoom_ping": None,
        "zoom_depth": None,
        "zoom_x": None,
        "zoom_y": None,
        "zoom_z": None
    }

    # Training / Tuning
    RF_PARAM_GRID = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 4]
    }
    CNN_EPOCHS = 20
    CNN_BATCH_SIZE = 64
    CNN_LEARNING_RATE = 1e-3

    # For advanced hyperparameter tuning, set a flag here:
    USE_ADVANCED_TUNING = False  # If True, run e.g. Optuna or Hyperopt.

    # GPU usage
    USE_GPU = os.environ.get("USE_GPU", "False").lower() == "true"
