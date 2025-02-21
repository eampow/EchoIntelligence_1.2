# sonar_app/data_generation.py

import numpy as np
import pandas as pd
from typing import List, Tuple
from config import Config

def generate_species_data_multicluster(
    species_name: str,
    n_samples: int,
    cluster_means: List[Tuple[float, float]],
    cluster_std_speed: float = 0.05,
    cluster_std_ts: float = 1.0,
    speed_minmax: Tuple[float, float] = (0.0, 2.0),
    ts_minmax: Tuple[float, float] = (-50.0, -15.0),
) -> pd.DataFrame:
    """
    Generate synthetic data for a species using multiple Gaussian clusters.
    """
    n_clusters = len(cluster_means)
    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples % n_clusters
    dfs = []
    for i, (mean_speed, mean_ts) in enumerate(cluster_means):
        data = Config.RNG.normal(
            loc=[mean_speed, mean_ts],
            scale=[cluster_std_speed, cluster_std_ts],
            size=(samples_per_cluster, 2)
        )
        df_cluster = pd.DataFrame(data, columns=["speed", "target_strength"])
        df_cluster["speed"] = np.clip(df_cluster["speed"], speed_minmax[0], speed_minmax[1])
        df_cluster["target_strength"] = np.clip(df_cluster["target_strength"], ts_minmax[0], ts_minmax[1])
        df_cluster["species"] = species_name
        dfs.append(df_cluster)
    if remainder > 0:
        mean_speed, mean_ts = cluster_means[-1]
        data_left = Config.RNG.normal(
            loc=[mean_speed, mean_ts],
            scale=[cluster_std_speed, cluster_std_ts],
            size=(remainder, 2)
        )
        df_left = pd.DataFrame(data_left, columns=["speed", "target_strength"])
        df_left["speed"] = np.clip(df_left["speed"], speed_minmax[0], speed_minmax[1])
        df_left["target_strength"] = np.clip(df_left["target_strength"], ts_minmax[0], ts_minmax[1])
        df_left["species"] = species_name
        dfs.append(df_left)
    return pd.concat(dfs, ignore_index=True)

def generate_all_fish_data() -> pd.DataFrame:
    """
    Generate synthetic data for three fish species.
    """
    # Lake Sturgeon
    sturgeon_df = generate_species_data_multicluster(
        species_name="Lake Sturgeon",
        n_samples=3000,
        cluster_means=[(0.25, -22.0), (0.35, -24.5)],
        cluster_std_speed=0.07,
        cluster_std_ts=1.2,
        speed_minmax=(0.0, 1.0),
        ts_minmax=(-30.0, -18.0)
    )
    # Steelhead
    steelhead_df = generate_species_data_multicluster(
        species_name="Steelhead",
        n_samples=5000,
        cluster_means=[(0.9, -32.0), (1.3, -36.0), (1.6, -39.0)],
        cluster_std_speed=0.2,
        cluster_std_ts=1.5,
        speed_minmax=(0.0, 2.0),
        ts_minmax=(-45.0, -25.0)
    )
    # White Sucker
    white_sucker_df = generate_species_data_multicluster(
        species_name="White Sucker",
        n_samples=4000,
        cluster_means=[(0.45, -28.0), (0.6, -31.0)],
        cluster_std_speed=0.1,
        cluster_std_ts=1.3,
        speed_minmax=(0.0, 2.0),
        ts_minmax=(-38.0, -20.0)
    )
    df = pd.concat([sturgeon_df, steelhead_df, white_sucker_df], ignore_index=True)
    df = df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
    return df
