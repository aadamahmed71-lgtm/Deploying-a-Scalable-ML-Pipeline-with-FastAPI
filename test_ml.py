import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture
def sample_processed_data():
    """Process a small sample of the census data for use across tests."""
    sample = data.sample(n=200, random_state=42)
    X, y, encoder, lb = process_data(
        sample,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    return X, y


def test_train_model(sample_processed_data):
    """
    Test that train_model returns a trained RandomForestClassifier.
    """
    X, y = sample_processed_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference(sample_processed_data):
    """
    Test that inference returns predictions of the expected type and shape.
    """
    X, y = sample_processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns the expected precision, recall,
    and F1 values for a known set of labels and predictions.
    """
    y = np.array([1, 1, 1, 0, 0])
    preds = np.array([1, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)