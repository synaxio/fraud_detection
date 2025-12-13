from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from .model_loader import load_model
from io import StringIO
from typing import List

app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude",
    version="1.0.0"
)

# Charger le modèle au démarrage avec le chemin absolu
model = load_model("/app/model")

class DataRow(BaseModel):
    cc_num: float
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: float
    lat: float
    long: float
    city_pop: float
    job: str
    dob: str
    trans_num: str
    merch_lat: float
    merch_long: float
    is_fraud: float
    current_time: str

class Payload(BaseModel):
    data: List[DataRow]

@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "message": "Fraud Detection API is running", 
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_type": str(type(model).__name__)
    }

@app.post("/predict")
def predict(payload: List[DataRow]):
    """Endpoint de prédiction."""
    try:
        df = pd.DataFrame([row.dict() for row in payload])
        preds = model.predict(df)
        #return {"predictions": preds.to_dict(orient='records')}
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))