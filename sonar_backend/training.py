# sonar_app/training.py

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
# If you want advanced hyperparameter tuning, e.g., with Optuna or Hyperopt, you can import here
# import optuna  # Example

from data_generation import generate_all_fish_data
from models import Improved1DCNN
from config import Config

logger = logging.getLogger(__name__)

def data_augmentation_speed_ts(X, y, augmentation_factor=0.2):
    """
    Simple data augmentation that randomly perturbs speed and target_strength
    by small amounts.
    """
    n = len(X)
    augmented_size = int(n * augmentation_factor)
    indices = np.random.choice(n, augmented_size, replace=False)
    X_aug = X[indices].copy()
    y_aug = y[indices].copy()

    # Add small random noise
    noise_speed = np.random.normal(0, 0.05, size=augmented_size)
    noise_ts = np.random.normal(0, 1.0, size=augmented_size)
    X_aug[:, 0] += noise_speed
    X_aug[:, 1] += noise_ts
    # Clip to realistic ranges
    X_aug[:, 0] = np.clip(X_aug[:, 0], 0.0, 2.0)
    X_aug[:, 1] = np.clip(X_aug[:, 1], -50, -15)

    X_new = np.vstack([X, X_aug])
    y_new = np.hstack([y, y_aug])
    return X_new, y_new

def train_ml_models_with_visualization():
    """
    Train an optimized Random Forest classifier and an improved 1D CNN on synthetic fish data,
    with optional data augmentation, then generate visualizations.
    """
    print("Training ML models on synthetic fish data...")
    logger.info("Generating synthetic data.")
    df = generate_all_fish_data()
    X = df[['speed', 'target_strength']].values
    y = df['species'].values

    # Data augmentation
    X, y = data_augmentation_speed_ts(X, y, augmentation_factor=0.2)

    # Encode labels
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=Config.SEED, stratify=y_enc
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # --- Random Forest with GridSearch ---
    rf = RandomForestClassifier(random_state=Config.SEED)
    grid_search = GridSearchCV(rf, Config.RF_PARAM_GRID, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_sc, y_train)
    best_rf = grid_search.best_estimator_
    rf_preds = best_rf.predict(X_test_sc)
    print("\nOptimized Random Forest Classification Report:")
    print(classification_report(y_test, rf_preds, target_names=encoder.classes_))
    with open("rf_fish_model.pkl", "wb") as f:
        pickle.dump(best_rf, f)
    logger.info("Optimized Random Forest model saved.")

    # Feature importances
    importances = best_rf.feature_importances_
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['AvgSpeed', 'AvgTS'], y=importances)
    plt.title("Random Forest Feature Importances")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("rf_feature_importances.png")
    plt.close()
    logger.info("Random Forest feature importance plot saved.")

    # --- Train Improved 1D CNN ---
    device = torch.device("cuda" if (Config.USE_GPU and torch.cuda.is_available()) else "cpu")
    model_cnn = Improved1DCNN(num_classes=len(encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_cnn.parameters(), lr=Config.CNN_LEARNING_RATE)

    X_train_t = torch.tensor(X_train_sc, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test_sc, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=Config.CNN_BATCH_SIZE, shuffle=True)

    epochs = Config.CNN_EPOCHS
    train_losses, test_accuracies = [], []
    for epoch in range(epochs):
        model_cnn.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model_cnn(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluate
        model_cnn.eval()
        with torch.no_grad():
            logits_test = model_cnn(X_test_t)
            _, preds_cnn = torch.max(logits_test, 1)
            accuracy = (preds_cnn.cpu().numpy() == y_test).mean()
        test_accuracies.append(accuracy)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    print("\nImproved 1D CNN Classification Report:")
    print(classification_report(y_test, preds_cnn.cpu().numpy(), target_names=encoder.classes_))

    # Plot training progress
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.title("Improved CNN Training Loss and Test Accuracy")
    plt.plot(epochs_range, train_losses, color='red', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss', color='red')
    plt.twinx()
    plt.plot(epochs_range, test_accuracies, color='blue', label='Test Accuracy')
    plt.ylabel('Accuracy', color='blue')
    plt.tight_layout()
    plt.savefig("cnn_training_progress.png")
    plt.close()
    logger.info("Improved CNN training progress plot saved.")

    torch.save(model_cnn.state_dict(), "improved1d_cnn.pt")
    logger.info("Improved 1D CNN model saved.")
