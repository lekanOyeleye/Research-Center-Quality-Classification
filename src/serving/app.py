# FastAPI is the web framework — faster and async compared to Flask
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Import database functions
from src.database.db import create_table, save_prediction

# Import settings — no hardcoded paths
from src.config.settings import settings

# Instantiate the FastAPI app
app = FastAPI()

# Load the saved pipeline once at startup — not on every request
saved_process = joblib.load(settings.MODEL_PATH)

# Map cluster numbers to quality tier labels
TIER_MAP = {0: "Premium", 1: "Standard", 2: "Basic"}


# Pydantic schema — validates every incoming request automatically
class ResearchCenterData(BaseModel):
    internalFacilitiesCount: float
    hospitals_10km: float
    pharmacies_10km: float
    facilityDiversity_10km: float
    facilityDensity_10km: float


@app.on_event("startup")
async def startup():
    # Create the predictions table when the app starts — only if it does not exist
    create_table()


@app.get("/health")
def health():
    # Simple check to confirm the API is alive
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: ResearchCenterData):
    # Get selected features and pipeline from saved model
    selected_features = saved_process["selected_features"]
    model = saved_process["pipeline"]

    # Convert incoming request to DataFrame using selected feature names
    df = pd.DataFrame([data.model_dump()], columns=selected_features)

    # Get prediction from model
    prediction = model.predict(df)
    predicted_cluster = int(prediction[0])

    # Map cluster to quality tier — clean dictionary lookup
    predicted_category = TIER_MAP.get(predicted_cluster, "Unknown")

    # Save input data and prediction to PostgreSQL
    save_prediction(
        input_data=data.model_dump(),
        predicted_cluster=predicted_cluster,
        predicted_category=predicted_category
    )

    return {
        "PredictedCluster": predicted_cluster,
        "PredictedCategory": predicted_category
    }


@app.get("/history")
def history():
    # Returns the last 10 predictions from the database
    from src.database.db import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, internal_facilities_count, hospitals_10km, pharmacies_10km,
               facility_diversity_10km, facility_density_10km,
               predicted_cluster, predicted_category, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 10
    """)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()

    results = []
    for row in rows:
        row_dict = dict(zip(columns, row))
        row_dict["created_at"] = str(row_dict["created_at"])
        results.append(row_dict)

    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)