"""FastAPI inference server for Research Center Quality Classification"""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app = FastAPI()

# Load your saved pipeline
saved_process = joblib.load('./model/artifacts/kmeans_pipeline_model.pkl')


class RecearchCenterData(BaseModel):
    internalFacilitiesCount: float
    hospitals_10km: float
    pharmacies_10km: float
    facilityDiversity_10km: float
    facilityDensity_10km: float


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: RecearchCenterData):
    """Predicts quality tier for a new research center."""
    try:
        selected_features = saved_process["selected_features"]
        model = saved_process['pipeline']
        
        # Dump the ResearchCenterData in a dataframe and use the seleceted features as column names
        df = pd.DataFrame([data.model_dump()], columns=selected_features)
        
        prediction = model.predict(df)
        
        # M
        qualityTier = ['Premium' if i == 0 else 'Standard' if i == 1 else 'Basic' if i == 2 else 'other' for i in prediction]
        
        return {
        "PredictedCluster": int(prediction[0]),
        "PredictedCategory": qualityTier[0]
        }
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
