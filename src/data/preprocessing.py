"""Preprocessing de base pour fraude et segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les doublons et remet les noms de colonnes en forme."""
    cleaned = df.copy()
    cleaned.columns = [col.strip() for col in cleaned.columns]
    cleaned = cleaned.drop_duplicates()
    return cleaned


def preprocess_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing initial pour la fraude."""
    fraud_df = basic_cleaning(df)

    # Evite les divisions par zero pour les futures features.
    if "oldbalanceOrg" in fraud_df.columns:
        fraud_df["oldbalanceOrg"] = fraud_df["oldbalanceOrg"].replace({0: np.nan})

    return fraud_df


def preprocess_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing initial pour le clustering client."""
    cluster_df = basic_cleaning(df)

    if "Income" in cluster_df.columns:
        cluster_df["Income"] = cluster_df["Income"].fillna(cluster_df["Income"].median())

    # Colonnes constantes connues.
    for col in ("Z_CostContact", "Z_Revenue"):
        if col in cluster_df.columns:
            cluster_df = cluster_df.drop(columns=col)

    return cluster_df
