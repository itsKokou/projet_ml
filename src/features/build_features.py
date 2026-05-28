"""Feature engineering pour les deux cas d'usage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit des variables derivees utiles pour detecter la fraude."""
    features = df.copy()

    if {"oldbalanceOrg", "newbalanceOrig", "amount"}.issubset(features.columns):
        features["origin_balance_diff"] = features["oldbalanceOrg"] - features["newbalanceOrig"]
        features["origin_error"] = features["origin_balance_diff"] - features["amount"]

    if {"oldbalanceDest", "newbalanceDest", "amount"}.issubset(features.columns):
        features["dest_balance_diff"] = features["newbalanceDest"] - features["oldbalanceDest"]
        features["dest_error"] = features["dest_balance_diff"] - features["amount"]

    if {"type"}.issubset(features.columns):
        features["is_transfer_or_cashout"] = features["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

    if {"newbalanceOrig"}.issubset(features.columns):
        features["is_zero_newbalance_origin"] = (features["newbalanceOrig"] == 0).astype(int)

    if {"oldbalanceDest"}.issubset(features.columns):
        features["is_zero_oldbalance_dest"] = (features["oldbalanceDest"] == 0).astype(int)

    if {"amount", "oldbalanceOrg"}.issubset(features.columns):
        denom = features["oldbalanceOrg"].replace({0: np.nan})
        features["amount_to_oldbalance_ratio"] = features["amount"] / denom
        features["amount_to_oldbalance_ratio"] = features["amount_to_oldbalance_ratio"].replace(
            [np.inf, -np.inf], np.nan
        )

    if {"step"}.issubset(features.columns):
        # Encodage temporel simple sans supposer une periodicite metier forte.
        features["step_bucket"] = (features["step"] // 24).astype(int)

    return features


def build_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit des variables metier pour la segmentation."""
    features = df.copy()

    if "Year_Birth" in features.columns:
        features["Age"] = 2026 - features["Year_Birth"]

    if "Dt_Customer" in features.columns:
        dt = pd.to_datetime(features["Dt_Customer"], format="%d/%m/%Y", errors="coerce")
        if dt.notna().any():
            reference_date = dt.max()
            features["Customer_Tenure_days"] = (reference_date - dt).dt.days

    spending_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    existing_spending = [col for col in spending_cols if col in features.columns]
    if existing_spending:
        features["Total_Spending"] = features[existing_spending].sum(axis=1)

    purchase_cols = [
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]
    existing_purchase = [col for col in purchase_cols if col in features.columns]
    if existing_purchase:
        features["Total_Purchases"] = features[existing_purchase].sum(axis=1)
        denom = features["Total_Purchases"].replace({0: np.nan})
        for col in existing_purchase:
            ratio_name = col.replace("Num", "").replace("Purchases", "_Ratio")
            features[ratio_name] = features[col] / denom

    if {"Kidhome", "Teenhome"}.issubset(features.columns):
        features["Children"] = features["Kidhome"] + features["Teenhome"]

    campaign_cols = [f"AcceptedCmp{i}" for i in range(1, 6)]
    existing_campaign = [col for col in campaign_cols if col in features.columns]
    if "Response" in features.columns:
        existing_campaign.append("Response")
    if existing_campaign:
        features["Campaign_Acceptance_Total"] = features[existing_campaign].sum(axis=1)

    return features
