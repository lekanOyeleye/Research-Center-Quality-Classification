# Research Center Quality Classification — Production ML Pipeline with CI/CD

A production-grade machine learning system that classifies research centers into quality tiers — Premium, Standard, and Basic — using KMeans clustering. Built to demonstrate end-to-end ML engineering from model training through containerised deployment with a fully automated CI/CD pipeline.

This project was built as part of a technical assessment and subsequently refactored into a production-grade system with proper separation of concerns, persistent storage, automated testing, and continuous integration.

---

## Architecture

```
src/
├── schemas/        # Pydantic input validation schemas
├── serving/        # FastAPI REST API (serving layer)
├── config/         # Settings and environment configuration
└── database/       # PostgreSQL connection, table creation, and queries
tests/              # Pytest unit tests with mocking
.github/
└── workflows/
    └── ci.yml      # GitHub Actions CI/CD pipeline
Dockerfile          # Container definition with non-root user
docker-compose.yml  # Multi-container orchestration with healthcheck
model/
└── artifacts/
    └── kmeans_pipeline_model.pkl   # Trained model — mounted as volume
```

Three layers are kept strictly separate:
- **Serving layer** — knows only about HTTP requests and responses
- **Database layer** — knows only about PostgreSQL connections and queries
- **Infrastructure** — Docker, environment configuration, CI/CD

---

## Model

A KMeans clustering model trained on research center facility data. Centers are grouped into three quality tiers based on internal facilities, proximity to hospitals and pharmacies, facility diversity, and facility density.

**Model validation:**
- Silhouette score: ~0.7 (range -1 to 1, higher is better)
- Features selected using VarianceThreshold to remove low-information variables
- Optimal number of clusters determined through elbow method and silhouette analysis

**Quality tiers:**

| Cluster | Category |
|---------|----------|
| 0 | Premium |
| 1 | Standard |
| 2 | Basic |

---

## CI/CD Pipeline

Every push to the `cicd` branch automatically triggers the pipeline:

```
Push to cicd branch
        ↓
GitHub Actions triggered
        ↓
Fresh Ubuntu machine created
        ↓
Python 3.12 installed
        ↓
Dependencies installed from requirements.txt
        ↓
pytest runs — 4 tests
        ↓
If any test fails — STOP, nothing builds
        ↓
If tests pass — Docker image built
        ↓
Green checkmark on GitHub
```

The pipeline ensures no broken code ever reaches the Docker build stage.

---

## Features

- **KMeans Classification** — clusters research centers into Premium, Standard, and Basic quality tiers
- **Pydantic Validation** — every incoming request validated against a strict schema before reaching the model
- **PostgreSQL Logging** — every prediction saved permanently with input features, predicted cluster, category, and timestamp
- **Volume Mount** — model pickle file mounted separately from the container image — retraining only replaces the file, no image rebuild needed
- **REST API** — FastAPI serving layer with `/health`, `/predict`, and `/history` endpoints
- **Containerised Deployment** — fully Dockerised with Docker Compose, non-root user, and healthcheck-based startup ordering
- **Automated Tests** — 4 Pytest tests covering API behaviour with full mocking of external dependencies
- **CI/CD** — GitHub Actions pipeline runs tests and builds Docker image on every push

---

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.12 (for local development)

### Setup

1. Clone the repository and switch to the cicd branch:

```bash
git clone https://github.com/lekanOyeleye/Research-Center-Quality-Classification.git
cd Research-Center-Quality-Classification
git checkout cicd
```

2. Create a `.env` file in the project root:

```
POSTGRES_HOST=db
POSTGRES_PORT=5432
POSTGRES_DB=research
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
MODEL_PATH=/app/model/artifacts/kmeans_pipeline_model.pkl
```

3. Train the model locally first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

4. Build and start the containers:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

---

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "healthy"}
```

### Predict Quality Tier

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "internalFacilitiesCount": 9,
    "hospitals_10km": 3,
    "pharmacies_10km": 2,
    "facilityDiversity_10km": 0.82,
    "facilityDensity_10km": 0.45
  }'
```

Response:
```json
{
  "PredictedCluster": 1,
  "PredictedCategory": "Standard"
}
```

### View Prediction History

```bash
curl http://localhost:8000/history
```

Response:
```json
[
  {
    "id": 1,
    "internal_facilities_count": 9.0,
    "hospitals_10km": 3.0,
    "pharmacies_10km": 2.0,
    "facility_diversity_10km": 0.82,
    "facility_density_10km": 0.45,
    "predicted_cluster": 1,
    "predicted_category": "Standard",
    "created_at": "2026-04-13 15:36:21.337964"
  }
]
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

4 tests covering:
- Health endpoint returns healthy status
- Predict endpoint returns 200 with valid input
- Predict endpoint returns correct fields
- Predict endpoint rejects missing fields with 422

All external dependencies (PostgreSQL, model) are mocked so tests run offline with no database needed.

---

## Retraining

The model is mounted as a volume — not baked into the Docker image. This means retraining does not require rebuilding the image:

```bash
# Step 1 — Retrain the model with new data
python train.py

# Step 2 — Restart the API to load the new model
docker compose restart api
```

The new pickle file is picked up automatically from the mounted volume.

---

## Scaling to Production

### Kubernetes with KServe

For production scale the Docker image is deployed on Kubernetes using KServe:

- **Autoscaling** — KServe automatically scales containers up with traffic and down to zero during quiet periods, eliminating idle compute costs
- **Canary deployments** — new models are routed 10% of traffic first. If the silhouette score holds, traffic switches fully. If not, rollback is instant
- **Model versioning** — integrates with MLflow so a promoted model triggers automatic deployment without manual intervention

### Continuous Retraining

```
New prediction data accumulates in PostgreSQL
        ↓
Retraining triggers automatically
        ↓
train.py runs on fresh data
        ↓
MLflow logs new silhouette score
        ↓
If score improves — new model promoted to production
        ↓
KServe deploys via canary — 10% traffic first
        ↓
Full rollout or instant rollback
```

---

## Design Decisions

**Why volume mount instead of baking the model into the image?**
Separating model from code means retraining only replaces one file. No image rebuild, no reinstalling dependencies — the container restarts in seconds with the new model.

**Why FastAPI over Flask?**
FastAPI is asynchronous — while one worker waits for a database response, it handles other requests. It also provides automatic API documentation at `/docs` and native Pydantic integration.

**Why PostgreSQL over a CSV log?**
Every prediction saved to PostgreSQL becomes training data for future retraining. A CSV file cannot be queried efficiently at scale. PostgreSQL supports complex queries — for example, fetching all predictions from the last 30 days where the model was uncertain.

**Why a non-root user in Docker?**
Running containers as root means an attacker who exploits the app gets root access to the container. `appuser` limits the blast radius of any security incident.

---

## Tech Stack

- **Model** — KMeans clustering via Scikit-learn
- **API** — FastAPI with Uvicorn (4 workers)
- **Validation** — Pydantic v2
- **Database** — PostgreSQL 15 with psycopg2
- **Containerisation** — Docker + Docker Compose
- **CI/CD** — GitHub Actions
- **Testing** — Pytest with unittest.mock and httpx
