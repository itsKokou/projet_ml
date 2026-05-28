"""Fonctions utilitaires de chargement des donnees."""

from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("data/raw")


def load_fraud_data(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Charge le dataset de fraude bancaire."""
    return pd.read_csv(data_dir / "detection_fraude.csv", sep=";")


def load_cluster_data(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Charge le dataset de segmentation client."""
    return pd.read_csv(data_dir / "data_cluster.csv", sep=";")
