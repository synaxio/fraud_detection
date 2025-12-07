from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from .model_loader import load_model

app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude",
    version="1.0.0"
)

# Charger le modèle au démarrage avec le chemin absolu
model = load_model("/app/model")

class Payload(BaseModel):
    data: list

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
def predict(payload: Payload):
    """Endpoint de prédiction."""
    try:
        df = pd.DataFrame(payload.data)
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))