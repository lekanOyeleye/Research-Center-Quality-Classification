import os
import time
import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)


def get_connection():
    max_retries = 5
    retry_delay = 3

    for attempt in range(max_retries):
        try:
            return psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "db"),
                port=os.getenv("POSTGRES_PORT", 5432),
                database=os.getenv("POSTGRES_DB", "research"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "postgres")
            )
        except psycopg2.OperationalError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Database not ready, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)


def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            internal_facilities_count FLOAT,
            hospitals_10km FLOAT,
            pharmacies_10km FLOAT,
            facility_diversity_10km FLOAT,
            facility_density_10km FLOAT,
            predicted_cluster INTEGER,
            predicted_category VARCHAR,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()


def save_prediction(input_data: dict, predicted_cluster: int, predicted_category: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            internal_facilities_count,
            hospitals_10km,
            pharmacies_10km,
            facility_diversity_10km,
            facility_density_10km,
            predicted_cluster,
            predicted_category
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """,
    (
        input_data["internalFacilitiesCount"],
        input_data["hospitals_10km"],
        input_data["pharmacies_10km"],
        input_data["facilityDiversity_10km"],
        input_data["facilityDensity_10km"],
        predicted_cluster,
        predicted_category
    ))
    conn.commit()
    cursor.close()
    conn.close()