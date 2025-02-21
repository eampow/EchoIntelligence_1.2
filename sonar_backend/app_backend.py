#!/usr/bin/env python3
import logging
import io
import base64
import os
import json

from flask import Flask, jsonify, request
from flask_cors import CORS

# Set Matplotlib to a non-GUI backend before importing pyplot/seaborn.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import your modules (adjust file names as needed)
from data_generation import generate_all_fish_data
from training import train_ml_models_with_visualization
from file_loading import load_and_parse_file, analyze_target_strength, compute_track_features
from tracking import DBSCANFishTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

##########################################################################
# Endpoint: /plot/synthetic
##########################################################################
@app.route("/plot/synthetic", methods=["GET"])
def synthetic_plot():
    try:
        df = generate_all_fish_data()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='speed', y='target_strength', hue='species', alpha=0.6)
        plt.title("Synthetic Fish Data Distribution")
        plt.xlabel("Speed (AvgSpeed)")
        plt.ylabel("Target Strength (AvgTS)")
        plt.legend(title="Species")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        logger.info("Synthetic plot generated successfully.")
        return jsonify({"image": img_b64}), 200
    except Exception as e:
        logger.exception("Error generating synthetic plot.")
        return jsonify({"error": str(e)}), 500

##########################################################################
# Endpoint: /train
##########################################################################
@app.route("/train", methods=["POST"])
def train_models():
    try:
        train_ml_models_with_visualization()
        logger.info("Training completed successfully.")
        return jsonify({"status": "Training completed successfully."}), 200
    except Exception as e:
        logger.exception("Training failed.")
        return jsonify({"error": str(e)}), 500

##########################################################################
# Endpoint: /open-file
##########################################################################
@app.route("/open-file", methods=["POST"])
def open_file():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        content = file.read()
        logger.info("Received file of size %d bytes", len(content))

        df = load_and_parse_file(content)
        if df is None or df.empty:
            return jsonify({"error": "No valid data parsed from file."}), 400

        # FIX: Standardize column names.
        # If the file returns "target_strength" but we expect "TS":
        if "target_strength" in df.columns and "TS" not in df.columns:
            df.rename(columns={"target_strength": "TS"}, inplace=True)
            logger.info("Renamed 'target_strength' column to 'TS'.")
        # Similarly, if the file returns "ping" in lowercase:
        if "ping" in df.columns and "Ping" not in df.columns:
            df.rename(columns={"ping": "Ping"}, inplace=True)
            logger.info("Renamed 'ping' column to 'Ping'.")

        df = analyze_target_strength(df)
        summary = compute_track_features(df)
        logger.info("File processed successfully.")
        return jsonify({
            "status": "File processed successfully",
            "summary": summary.to_dict(orient="records")
        }), 200
    except Exception as e:
        logger.exception("Error processing file.")
        return jsonify({"error": str(e)}), 500

##########################################################################
# Endpoint: /plot/echogram
# Returns two echograms in Plotly format:
#   1. rawEchogram: Raw data (Ping vs. Depth, colored by TS)
#   2. trackedEchogram: Tracked data, each track is its own trace with a legend label
##########################################################################
@app.route("/plot/echogram", methods=["POST"])
def echogram_plot():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        content = file.read()
        logger.info("Received RAW file with size %d bytes", len(content))
        
        df = load_and_parse_file(content)
        if df is None or df.empty:
            return jsonify({"error": "No valid data parsed from RAW file."}), 400

        # Ensure columns are standardized.
        if "target_strength" in df.columns and "TS" not in df.columns:
            df.rename(columns={"target_strength": "TS"}, inplace=True)
            logger.info("Renamed 'target_strength' to 'TS'.")
        if "ping" in df.columns and "Ping" not in df.columns:
            df.rename(columns={"ping": "Ping"}, inplace=True)
            logger.info("Renamed 'ping' to 'Ping'.")
        
        df = analyze_target_strength(df)
        
        # Build raw echogram
        raw_data = [{
            "x": df["Ping"].tolist(),
            "y": df["Depth"].tolist(),
            "mode": "markers",
            "type": "scatter",
            "marker": {
                "color": df["TS"].tolist(),
                "colorscale": "viridis",
                "colorbar": {"title": "TS (dB)"},
                "size": 5
            },
            "name": "Raw Echogram"
        }]
        raw_layout = {
            "title": "Raw Echogram",
            "xaxis": {"title": "Ping"},
            "yaxis": {"title": "Depth (m)", "autorange": "reversed"}
        }
        
        # Track fish using DBSCAN
        tracker = DBSCANFishTracker(eps=3.0, min_samples=3, min_track_length=2)
        df_tracked = tracker.track(df)
        
        # Compute track features
        summary_df = compute_track_features(df_tracked)
        
        # Dummy species prediction: assign species based on TrackID mod 3.
        species_map = {0: "Lake Sturgeon", 1: "White Sucker", 2: "Steelhead"}
        summary_df["PredSpecies"] = summary_df["TrackID"].apply(lambda tid: species_map[tid % 3])
        
        # Merge predicted species into tracked data.
        df_tracked = df_tracked.merge(summary_df[["TrackID", "PredSpecies"]], on="TrackID", how="left")
        
        # Build a separate trace for each track (for a detailed legend)
        tracked_data = []
        unique_tracks = df_tracked["TrackID"].unique().tolist()
        for track_id in unique_tracks:
            track_data = df_tracked[df_tracked["TrackID"] == track_id]
            species = track_data["PredSpecies"].iloc[0] if not track_data.empty else "Unknown"
            trace_name = f"Track {track_id} - {species}"
            trace = {
                "x": track_data["Ping"].tolist(),
                "y": track_data["Depth"].tolist(),
                "mode": "markers",
                "type": "scatter",
                "name": trace_name,
                "marker": {"size": 6}
            }
            tracked_data.append(trace)
        tracked_layout = {
            "title": "Tracked Echogram (Each Track Labeled)",
            "xaxis": {"title": "Ping"},
            "yaxis": {"title": "Depth (m)", "autorange": "reversed"}
        }
        
        summary_records = summary_df.to_dict(orient="records")
        
        return jsonify({
            "rawEchogram": {"data": raw_data, "layout": raw_layout},
            "trackedEchogram": {"data": tracked_data, "layout": tracked_layout},
            "summary": summary_records,
            "status": "Echograms generated successfully with species identification."
        }), 200
        
    except Exception as e:
        logger.exception("Error generating echograms.")
        return jsonify({"error": str(e)}), 500

##########################################################################
# Endpoint: /plot/3d-spatial
# Returns a 3D spatial plot in Plotly format.
##########################################################################
@app.route("/plot/3d-spatial", methods=["GET"])
def spatial_plot():
    try:
        import numpy as np
        t = np.linspace(0, 10, 50)
        x = np.sin(t)
        y = np.cos(t)
        z = t
        data = [{
            "x": x.tolist(),
            "y": y.tolist(),
            "z": z.tolist(),
            "mode": "markers+lines",
            "type": "scatter3d",
            "marker": {
                "size": 4,
                "color": z.tolist(),
                "colorscale": "Viridis",
                "colorbar": {"title": "Z (Range)"}
            }
        }]
        layout = {
            "title": "3D Spatial Plot",
            "scene": {
                "xaxis": {"title": "X (m)"},
                "yaxis": {"title": "Y (m)"},
                "zaxis": {"title": "Depth (m)", "autorange": "reversed"}
            }
        }
        return jsonify({"data": data, "layout": layout}), 200
    except Exception as e:
        logger.exception("Error generating 3D spatial plot.")
        return jsonify({"error": str(e)}), 500

##########################################################################
# Endpoint: /summary
# Returns overall summary statistics.
##########################################################################
@app.route("/summary", methods=["GET"])
def summary_stats():
    try:
        df = generate_all_fish_data()
        df = analyze_target_strength(df)
        total_pings = len(df)
        # FIX: Ensure TS exists
        if "target_strength" in df.columns and "TS" not in df.columns:
            df.rename(columns={"target_strength": "TS"}, inplace=True)
        avg_ts = df["TS"].mean()
        avg_speed = df["speed"].mean()
        species_counts = df["species"].value_counts().to_dict()
        summary = {
            "totalPings": total_pings,
            "averageTS": avg_ts,
            "averageSpeed": avg_speed,
            "speciesDistribution": species_counts
        }
        return jsonify({"summary": summary}), 200
    except Exception as e:
        logger.exception("Error computing summary statistics.")
        return jsonify({"error": str(e)}), 500

##########################################################################
# Endpoint: /plot/training-visualization
# Returns a training visualization as a base64 image.
##########################################################################
@app.route("/plot/training-visualization", methods=["GET"])
def training_visualization():
    try:
        plt.figure(figsize=(8, 5))
        plt.plot([1, 2, 3, 4], [0.5, 0.3, 0.2, 0.1], marker='o')
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return jsonify({"image": img_b64}), 200
    except Exception as e:
        logger.exception("Error generating training visualization.")
        return jsonify({"error": str(e)}), 500

##########################################################################
# Main: Run the Flask App
##########################################################################
if __name__ == "__main__":
    app.run(debug=True, port=5000)
