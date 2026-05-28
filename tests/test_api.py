"""Tests API de base."""

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert "fraud_model" in body
    assert "cluster_model" in body


def test_predict_fraud_payload_valid():
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 1000.0,
        "nameOrig": "C123",
        "oldbalanceOrg": 2000.0,
        "newbalanceOrig": 1000.0,
        "nameDest": "C456",
        "oldbalanceDest": 500.0,
        "newbalanceDest": 1500.0,
    }
    response = client.post("/predict/fraud", json={"payload": payload})
    assert response.status_code in (200, 503)


def test_predict_segment_payload_valid():
    payload = {
        "Year_Birth": 1985,
        "Education": "Graduation",
        "Marital_Status": "Single",
        "Income": 50000,
        "Kidhome": 1,
        "Teenhome": 0,
        "Dt_Customer": "01/01/2014",
        "Recency": 20,
        "MntWines": 200,
        "MntFruits": 20,
        "MntMeatProducts": 120,
        "MntFishProducts": 30,
        "MntSweetProducts": 15,
        "MntGoldProds": 40,
        "NumDealsPurchases": 2,
        "NumWebPurchases": 5,
        "NumCatalogPurchases": 2,
        "NumStorePurchases": 4,
        "NumWebVisitsMonth": 6,
        "AcceptedCmp1": 0,
        "AcceptedCmp2": 0,
        "AcceptedCmp3": 0,
        "AcceptedCmp4": 0,
        "AcceptedCmp5": 0,
        "Complain": 0,
        "Response": 0,
    }
    response = client.post("/predict/segment", json={"payload": payload})
    assert response.status_code in (200, 503)
