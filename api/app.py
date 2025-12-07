from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from .model_loader import load_model

app = FastAPI()

model = load_model()

class Payload(BaseModel):
    data: list

@app.get("/")
async def root():
    """Endpoint racine."""
    return {"message": "Fraud Detection API is running", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: Payload):
    try:
        df = pd.DataFrame(payload.data)
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
