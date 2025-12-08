import mlflow
import pandas as pd

def test_model_prediction():
    """
    Vérifie que le modèle MLflow charge correctement.
    """
    model_uri = "model"  # dossier local après download dans GitHub Actions
    model = mlflow.pyfunc.load_model(model_uri)

    # Exemple : un échantillon minimal
    sample = pd.DataFrame([{
        "cc_num": 1234567890123456,
        "merchant": "fraud_Test",
        "category": "shopping",
        "amt": 50.0,
        "first": "John",
        "last": "Doe",
        "gender": "M",
        "street": "1 Test Street",
        "city": "Paris",
        "state": "FR",
        "zip": 75000,
        "lat": 48.8566,
        "long": 2.3522,
        "city_pop": 2148000,
        "job": "Engineer",
        "dob": "1980-01-01",
        "trans_num": "test123",
        "merch_lat": 48.85,
        "merch_long": 2.34,
        "current_time": "2025-01-01 00:00:00"
    }])

    pred = model.predict(sample)

    assert pred is not None
    assert len(pred) == 1
