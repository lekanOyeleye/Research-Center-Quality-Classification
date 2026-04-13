import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "db")
    POSTGRES_PORT: int = os.getenv("POSTGRES_PORT", 5432)
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "research")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./model/artifacts/kmeans_pipeline_model.pkl")

settings = Settings()