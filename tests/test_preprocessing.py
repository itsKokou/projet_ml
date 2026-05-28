"""Tests de base du preprocessing."""

import pandas as pd

from src.data.preprocessing import preprocess_cluster, preprocess_fraud


def test_preprocess_cluster_removes_constant_columns():
    df = pd.DataFrame(
        {
            "Income": [1000, None, 2000],
            "Z_CostContact": [3, 3, 3],
            "Z_Revenue": [11, 11, 11],
        }
    )
    out = preprocess_cluster(df)

    assert "Z_CostContact" not in out.columns
    assert "Z_Revenue" not in out.columns
    assert out["Income"].isna().sum() == 0


def test_preprocess_fraud_replaces_zero_oldbalance():
    df = pd.DataFrame(
        {
            "oldbalanceOrg": [0.0, 100.0],
            "newbalanceOrig": [0.0, 20.0],
            "amount": [10.0, 80.0],
        }
    )
    out = preprocess_fraud(df)

    assert pd.isna(out.loc[0, "oldbalanceOrg"])
    assert out.loc[1, "oldbalanceOrg"] == 100.0
