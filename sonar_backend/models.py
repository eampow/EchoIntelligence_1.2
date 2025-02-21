# sonar_app/models.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from typing import List
from config import Config

logger = logging.getLogger(__name__)

# Improved 1D CNN with additional layers, batch normalization, and dropout
class Improved1DCNN(nn.Module):
    """
    An improved 1D CNN that treats (speed, target_strength) as a 1D signal.
    """
    def __init__(self, num_classes: int = 3):
        super(Improved1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.3)
        # After conv1 (kernel_size=2) on a 2-element input, shape = (8, 1)
        # conv2 (kernel_size=1) -> shape = (16, 1)
        # So final flattened size = 16
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class EchogramCNN(nn.Module):
    """
    A CNN for echogram images.
    """
    def __init__(self, num_classes: int = 3):
        super(EchogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def load_random_forest_model(path: str):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Random Forest model loaded from %s", path)
        return model
    except Exception as e:
        logger.exception("Failed to load Random Forest model from %s: %s", path, e)
        raise

def load_1d_cnn_model(path: str) -> Improved1DCNN:
    try:
        model = Improved1DCNN(num_classes=3)
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        model.eval()
        logger.info("Improved 1D CNN model loaded from %s", path)
        return model
    except Exception as e:
        logger.exception("Failed to load Improved 1D CNN model from %s: %s", path, e)
        raise

def load_deep_learning_model(path: str) -> nn.Module:
    try:
        model = EchogramCNN(num_classes=3)
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        model.eval()
        logger.info("Deep learning model loaded from %s", path)
        return model
    except Exception as e:
        logger.exception("Failed to load deep model from %s: %s", path, e)
        raise

def predict_species_rf(rf_model, track_feats: pd.DataFrame) -> List:
    if track_feats.empty:
        return []
    feature_cols = ["AvgTS", "AvgSpeed"]
    for col in feature_cols:
        track_feats[col] = pd.to_numeric(track_feats[col], errors="coerce")
    X = track_feats[feature_cols]
    preds = rf_model.predict(X)
    logger.info("Random Forest predictions computed.")
    return preds

def predict_species_blended(track_feats: pd.DataFrame) -> List:
    if track_feats.empty:
        return []
    feature_cols = ["AvgTS", "AvgSpeed"]
    X_rf = track_feats[feature_cols].to_numpy()
    X_cnn = track_feats[feature_cols].to_numpy()
    rf_model = load_random_forest_model("rf_fish_model.pkl")
    cnn_model = load_1d_cnn_model("improved1d_cnn.pt")

    rf_probs = rf_model.predict_proba(X_rf)
    X_tensor = torch.tensor(X_cnn, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        logits = cnn_model(X_tensor)
        cnn_probs = F.softmax(logits, dim=1).cpu().numpy()
    alpha = 0.5
    blended_probs = alpha * rf_probs + (1 - alpha) * cnn_probs
    preds = np.argmax(blended_probs, axis=1)
    logger.info("Blended species predictions computed.")
    return preds
