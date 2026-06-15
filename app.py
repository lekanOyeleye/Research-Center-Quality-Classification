from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import os




app = FastAPI(title="Research Center Quality API")

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://[::1]:8081")
MODEL_NAME = "FEM"
MODEL_VERSION = "latest"

# Global variables
model = None
selected_features = None


def load_model_from_mlflow():
    """Load model and selected features from MLflow"""
    global model, selected_features
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        model = mlflow.sklearn.load_model(model_uri)
        
        client = mlflow.tracking.MlflowClient()
        model_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
        
        if not model_versions:
            raise Exception(f"No versions found for model {MODEL_NAME}")
        
        run_id = model_versions[0].run_id
        run = client.get_run(run_id)
        selected_features_str = run.data.params.get('selected_features', '')
        
        import ast
        selected_features = ast.literal_eval(selected_features_str)
        
        print(f"✅ Model loaded: {MODEL_NAME}/{MODEL_VERSION}")
        print(f"   Selected features: {selected_features}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


@app.on_event("startup")
def startup_event():
    """Load model on startup"""
    print("🚀 Starting up...")
    load_model_from_mlflow()


# Make all fields optional
class ResearchCenterData(BaseModel):
    internalFacilitiesCount: Optional[float] = None
    hospitals_10km: Optional[float] = None
    pharmacies_10km: Optional[float] = None
    facilityDiversity_10km: Optional[float] = None
    facilityDensity_10km: Optional[float] = None


@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "selected_features": selected_features
    }


@app.post("/predict")
def predict(data: ResearchCenterData):
    """Make prediction"""
    if model is None or selected_features is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get non-null values
        input_dict = data.model_dump(exclude_none=True)
        
        # Check if we have all required features
        missing_features = [f for f in selected_features if f not in input_dict]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required features",
                    "missing": missing_features,
                    "required": selected_features,
                    "received": list(input_dict.keys())
                }
            )
        
        # Filter to selected features in the correct order
        input_filtered = {k: input_dict[k] for k in selected_features}
        df = pd.DataFrame([input_filtered])
        
        # Predict
        prediction = model.predict(df)
        
        # Map cluster to category
        cluster_mapping = {0: 'Premium', 1: 'Standard', 2: 'Basic'}
        quality_tier = cluster_mapping.get(int(prediction[0]), 'Other')
        
        return {
            "predicted_cluster": int(prediction[0]),
            "predicted_category": quality_tier,
            "features_used": selected_features
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting the server", flush=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)