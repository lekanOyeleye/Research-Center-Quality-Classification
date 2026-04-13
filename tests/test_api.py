# pytest is the testing framework — finds and runs all functions starting with test_
import pytest

# patch — temporarily replaces a real thing with a fake during a test
# MagicMock — creates a fake object that pretends to be anything we need
from unittest.mock import patch, MagicMock

# FastAPI's built in test client — creates a fake version of the API, no real server needed
from fastapi.testclient import TestClient

# numpy is needed because our fake model returns a numpy array — same as the real model
import numpy as np


# Create a fake saved_process dictionary before the app loads
# This mirrors the exact structure of the real pickle file
# selected_features — the list of column names the model expects
# pipeline — a fake model whose predict() returns cluster 1
fake_saved_process = {
    "selected_features": [
        "internalFacilitiesCount",
        "hospitals_10km",
        "pharmacies_10km",
        "facilityDiversity_10km",
        "facilityDensity_10km"
    ],
    # MagicMock creates a fake pipeline object
    # predict is set to return numpy array [1] — meaning cluster 1 — Standard
    "pipeline": MagicMock(predict=MagicMock(return_value=np.array([1])))
}

# Patch joblib.load BEFORE importing the app
# Without this, importing app.py would immediately try to load the real pickle file
# which may not exist in the CI environment
# We intercept joblib.load and return our fake_saved_process instead
with patch("joblib.load", return_value=fake_saved_process):
    # Also patch get_connection to stop any database connection attempts during import
    with patch("src.database.db.get_connection"):
        # Now it is safe to import the app — no real model or database needed
        from src.serving.app import app

# Create one test client shared by all tests
# This is a fake HTTP client — no real server running
client = TestClient(app)


# @patch decorators replace real functions with fakes for the duration of this test only
# create_table — stops it trying to connect to PostgreSQL at startup
# save_prediction — stops it trying to save to PostgreSQL after prediction
@patch("src.serving.app.create_table")
@patch("src.serving.app.save_prediction")
def test_health_endpoint(mock_save, mock_create):
    # Send a GET request to the health endpoint
    response = client.get("/health")
    # Check the response status code is 200 — meaning success
    assert response.status_code == 200
    # Check the response body contains status: healthy
    assert response.json()["status"] == "healthy"


@patch("src.serving.app.create_table")
@patch("src.serving.app.save_prediction")
def test_predict_returns_200(mock_save, mock_create):
    # Send a POST request to predict with valid input data
    response = client.post(
        "/predict",
        json={
            "internalFacilitiesCount": 9,
            "hospitals_10km": 3,
            "pharmacies_10km": 2,
            "facilityDiversity_10km": 0.82,
            "facilityDensity_10km": 0.45
        }
    )
    # Check the response is 200 — meaning the prediction was successful
    assert response.status_code == 200


@patch("src.serving.app.create_table")
@patch("src.serving.app.save_prediction")
def test_predict_returns_correct_fields(mock_save, mock_create):
    # Send a valid prediction request
    response = client.post(
        "/predict",
        json={
            "internalFacilitiesCount": 9,
            "hospitals_10km": 3,
            "pharmacies_10km": 2,
            "facilityDiversity_10km": 0.82,
            "facilityDensity_10km": 0.45
        }
    )
    # Convert response to dictionary
    data = response.json()
    # Check both expected fields are present in the response
    assert "PredictedCluster" in data
    assert "PredictedCategory" in data


@patch("src.serving.app.create_table")
@patch("src.serving.app.save_prediction")
def test_predict_rejects_missing_fields(mock_save, mock_create):
    # Send an empty request body — missing all required fields
    response = client.post("/predict", json={})
    # FastAPI returns 422 for validation errors — not 400 like Flask
    # 422 means Unprocessable Entity — the request was received but data is invalid
    assert response.status_code == 422