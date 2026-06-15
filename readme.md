# Advanced MLOps: Production-Grade ML System for Research Centre Quality Classification

A production-grade Machine Learning system that classifies research centres into quality tiers (Premium, Standard, Basic) using KMeans clustering. The system is built with a full MLOps stack including MLflow for experiment tracking, Apache Airflow for pipeline orchestration, Kubernetes (via Kind) for container orchestration, and Prometheus + Grafana for monitoring.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Step-by-Step Setup](#step-by-step-setup)
   - [Step 1 – Environment & Cluster Setup](#step-1--environment--cluster-setup)
   - [Step 2 – MLflow Installation & Configuration](#step-2--mlflow-installation--configuration)
   - [Step 3 – Model Training](#step-3--model-training)
   - [Step 4 – FastAPI Inference Server](#step-4--fastapi-inference-server)
   - [Step 5 – Containerisation & Kubernetes Deployment](#step-5--containerisation--kubernetes-deployment)
   - [Step 6 – Apache Airflow Pipeline Orchestration](#step-6--apache-airflow-pipeline-orchestration)
   - [Step 7 – Monitoring with Prometheus & Grafana](#step-7--monitoring-with-prometheus--grafana)
   - [Step 8 – CI/CD Verification](#step-8--cicd-verification)
7. [API Reference](#api-reference)
8. [ML Pipeline Details](#ml-pipeline-details)
9. [Data Description](#data-description)
10. [Configuration Reference](#configuration-reference)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements an end-to-end MLOps workflow for classifying research centres based on their facility quality and accessibility. It trains KMeans clustering models across all combinations of available features, tracks every experiment in MLflow, serves the best model via a FastAPI REST API deployed on Kubernetes, and orchestrates retraining pipelines through Apache Airflow.

**Key capabilities:**
- Automated feature combination search (31 combinations from 5 features)
- Silhouette score evaluation for unsupervised cluster quality
- Full experiment tracking and model registry via MLflow
- REST API for real-time inference
- Kubernetes-native deployment with Kind (local cluster)
- Airflow DAG for scheduled/triggered retraining
- Prometheus + Grafana for cluster and application monitoring

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Kind Kubernetes Cluster                   │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │   MLflow    │   │  FastAPI App │   │   Apache Airflow     │   │
│  │  (mlflow ns)│   │ (default ns) │   │    (airflow ns)      │   │
│  │  Port 8081  │   │  Port 5002   │   │   KubernetesPodOp    │   │
│  └──────┬──────┘   └──────┬───────┘   └──────────┬───────────┘   │
│         │                 │                      │               
│  ┌──────┴──────┐          │              ┌───────┴────────── ┐   │
│  │  PostgreSQL │          │              │  MLOps Train Pod  │   │
│  │  (backend)  │          │              │  (research-center-│   │
│  └─────────────┘          │              │   mlops:latest)   │   │
│                           │              └───────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                  Monitoring (monitoring ns)              │    │
│  │         Prometheus                   Grafana             │    │
│  │         (metrics)                  Port 3000             │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
         ▲                ▲
         │                │
   Local Port       Local Port
   8081 (MLflow)    5002 (API)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Framework | scikit-learn (KMeans, StandardScaler, VarianceThreshold) |
| Experiment Tracking | MLflow + PostgreSQL backend |
| Inference API | FastAPI + Uvicorn |
| Containerisation | Docker |
| Local Kubernetes | Kind (Kubernetes in Docker) |
| Package Management | Helm |
| Pipeline Orchestration | Apache Airflow (KubernetesPodOperator) |
| Monitoring | Prometheus + Grafana (kube-prometheus-stack) |
| Dependency Management | uv (Python) |

---

## Project Structure

```
advanced-mlops/
├── train.py                  # ML training script – feature combinations + MLflow logging
├── app.py                    # FastAPI inference server
├── requirements.txt          # Python dependencies
├── research_centers.csv      # Dataset (50 research centres, 10 columns)
├── kind-config.yaml          # Kind cluster config (1 control-plane, 2 workers)
├── k8s-manifest.yaml         # Kubernetes Deployment + Service for the API
├── mlflow-values.yaml        # Helm values for MLflow (PostgreSQL, artifact storage)
├── mlflow-setup.sh           # Script to install and port-forward MLflow
├── research_center_dag.py    # Airflow DAG for Kubernetes-based retraining
├── run_pipeline.sh           # Simple local pipeline runner
├── monitoring-setup.sh       # Script to install kube-prometheus-stack
├── observe.sh                # CI/CD health check script
└── readme.md                 # Original quick-reference notes
```

---

## Prerequisites

Ensure the following are installed on your machine before beginning:

- **Docker Desktop** (running) – required by Kind
- **Kind** – Kubernetes in Docker: https://kind.sigs.k8s.io/
- **kubectl** – Kubernetes CLI
- **Helm** – Kubernetes package manager
- **uv** – Fast Python package manager: https://github.com/astral-sh/uv
- **Python 3.10+**

---

## Step-by-Step Setup

### Step 1 – Environment & Cluster Setup

**Create the project directory:**

```bash
mkdir advanced-mlops
cd advanced-mlops
```

**Add the community Helm chart repository** (used for MLflow):

```bash
helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update
```

**Create the Kind cluster** using the provided configuration file. The `kind-config.yaml` defines 1 control-plane node and 2 worker nodes, mapping host ports `8080` and `8081` to NodePort services `30000` and `30001`:

```bash
kind create cluster --name advanced-mlops --config kind-config.yaml
```

This step may take a few minutes as Kind pulls the node images.

**Verify the nodes are ready:**

```bash
kubectl get nodes
```

You should see three nodes: one control-plane and two workers, all in `Ready` status.

---

### Step 2 – MLflow Installation & Configuration

MLflow is deployed on Kubernetes using Helm with a PostgreSQL backend store and local artifact storage.

**Kill any existing MLflow port-forward processes** to avoid conflicts:

```bash
pkill -f "kubectl port-forward svc/mlflow" || true
```

**Run the MLflow setup script:**

```bash
chmod +x ./mlflow-setup.sh
./mlflow-setup.sh
```

The setup script installs MLflow via Helm into the `mlflow` namespace with:
- PostgreSQL as the backend metadata store
- Artifact serving enabled (proxied storage at `/tmp/mlartifacts`)
- A port-forward from local port `8081` to the MLflow service

**Verify MLflow is running:**

```bash
curl http://localhost:8081/health
```

A successful response returns `OK`. You can also open the MLflow UI at `http://localhost:8081` in your browser.

---

### Step 3 – Model Training

**Set up the Python virtual environment** using `uv`:

```bash
uv init
uv sync
source .venv/bin/activate
uv add mlflow scikit-learn pandas
```

> **Note:** For background on the Exploratory Data Analysis and model selection, see the EDA notebook at:  
> https://github.com/lekanOyeleye/Research-Center-Quality-Classification/blob/main/EDA_and_Model.ipynb

**Run model training:**

```bash
uv run train.py
```

The training script (`train.py`) does the following:

1. Loads `research_centers.csv`
2. Defines 5 candidate features: `internalFacilitiesCount`, `hospitals_10km`, `pharmacies_10km`, `facilityDiversity_10km`, `facilityDensity_10km`
3. Generates **all 31 combinations** of these features (single features up to all 5 together)
4. For each combination, trains a scikit-learn pipeline:
   - `StandardScaler` → `VarianceThreshold` → `KMeans` (3 clusters, n_init=20, random_state=42)
5. Evaluates each model using the **Silhouette Score**
6. Logs parameters, metrics, selected features, and the trained model to MLflow under the experiment `Research_Centre_v1`
7. Registers each model in the MLflow Model Registry under the name `FEM`

**Expected output per run:**
```
✅ Model trained and logged successfully
   - Silhouette Score: 0.XXXX
   - Features selected: X/X
   - Selected features: [...]
```

---

### Step 4 – FastAPI Inference Server

The `app.py` FastAPI server loads the latest registered `FEM` model from MLflow at startup and exposes REST endpoints for health checking and prediction.

**Start the server locally:**

```bash
uv run app.py
```

The server runs on port `5001`.

**Test the health endpoint:**

```bash
curl http://localhost:5001/health
```

**Test a prediction:**

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "internalFacilitiesCount": 9,
    "hospitals_10km": 3,
    "pharmacies_10km": 2,
    "facilityDiversity_10km": 0.82,
    "facilityDensity_10km": 0.45
  }'
```

The server automatically determines which features are required based on the features selected during model training (stored as a run parameter in MLflow). Only those features need to be provided in the request body.

**Export dependencies to requirements.txt** (needed for Docker build):

```bash
uv export --format requirements-txt --no-hashes > requirements.txt
```

---

### Step 5 – Containerisation & Kubernetes Deployment

**Build the Docker image for the FastAPI server:**

```bash
docker build -t research-center-server:latest .
```

**Load the image into the Kind cluster** (Kind does not pull from Docker Hub by default):

```bash
kind load docker-image research-center-server:latest --name advanced-mlops
```

**Apply the Kubernetes manifest:**

```bash
kubectl apply -f k8s-manifest.yaml
```

The manifest (`k8s-manifest.yaml`) creates:
- A `Deployment` with 1 replica of the `research-center-server` container (port 5001), configured with the internal MLflow service URL
- A `ClusterIP` `Service` exposing port 80, targeting container port 5001

**Wait for the deployment to roll out:**

```bash
kubectl rollout status deployment/research-center-server --timeout=120s
```

**Kill any existing port-forwards and start a new one:**

```bash
pkill -f "kubectl port-forward svc/research-center-service" || true
kubectl port-forward svc/research-center-service 5002:80 &
```

**Verify the deployed API:**

```bash
# Health check (note: port 5002 from the port-forward, not 5001)
curl http://localhost:5002/health

# Make a prediction via the Kubernetes-deployed service
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "internalFacilitiesCount": 9,
    "hospitals_10km": 3,
    "pharmacies_10km": 2,
    "facilityDiversity_10km": 0.82,
    "facilityDensity_10km": 0.45
  }'
```

---

### Step 6 – Apache Airflow Pipeline Orchestration

Airflow is used to orchestrate retraining via a `KubernetesPodOperator`, which spawns a dedicated training pod inside the cluster on demand.

**Create the required files:**

```bash
touch research_center_dag.py
touch airflow-values.yaml
touch run_pipeline.sh
touch Dockerfile.mlops
```

**Build the API server image** (if not already done) and the dedicated MLOps training image:

```bash
# API server image
docker build -t research-center-server:latest .
kind load docker-image research-center-server:latest --name advanced-mlops

# MLOps training image (used by the Airflow KubernetesPodOperator)
docker build -t research-center-mlops:latest -f Dockerfile.mlops .
kind load docker-image research-center-mlops:latest --name advanced-mlops
```

**Run the Airflow setup script** (creates the Airflow namespace, RBAC, and Helm install):

```bash
chmod +x ./airflow-setup.sh
./airflow-setup.sh
```

**The Airflow DAG** (`research_center_dag.py`) defines a single task `train_model_in_k8s` using `KubernetesPodOperator`:
- Runs in the `airflow` namespace
- Uses the `research-center-mlops:latest` image (pulled from Kind cache)
- Executes `python3 train.py` inside the pod
- Sets `MLFLOW_TRACKING_URI` to the in-cluster MLflow service URL
- Deletes the pod after completion (`is_delete_operator_pod=True`)
- Schedule: `None` (manual trigger only)

Trigger a retraining run from the Airflow UI or CLI after setup is complete.

---

### Step 7 – Monitoring with Prometheus & Grafana

The `kube-prometheus-stack` Helm chart installs Prometheus and Grafana into the `monitoring` namespace.

**Run the monitoring setup script:**

```bash
chmod +x ./monitoring-setup.sh
./monitoring-setup.sh
```

The script:
1. Adds the `prometheus-community` Helm repo
2. Kills any existing Grafana port-forwards
3. Installs `kube-prometheus-stack` with Grafana admin password set to `admin`
4. Configures Prometheus to discover `PodMonitor` and `ServiceMonitor` resources across all namespaces
5. Waits for the Grafana pod to be ready (up to 10 minutes)
6. Port-forwards Grafana to local port `3000`

**Access Grafana:**

Open `http://localhost:3000` in your browser.
- Username: `admin`
- Password: `admin`

Prometheus is available internally at `http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090`.

---

### Step 8 – CI/CD Verification

The `observe.sh` script performs a full system health check across all components:

```bash
chmod +x observe.sh
./observe.sh
```

The script verifies:
1. **MLflow** – Queries the registered model registry for the latest `FEM` model version
2. **Airflow** – Checks that the `airflow-dags` ConfigMap exists in the `airflow` namespace
3. **API health** – Calls `http://localhost:5002/health` and confirms the server is responding
4. **Monitoring** – Checks that the Grafana pod is running in the `monitoring` namespace

A successful run ends with:
```
SYSTEM is READY you can trigger the airflow
```

---

## API Reference

### `GET /health`

Returns the current health status of the server and model.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "FEM",
  "selected_features": ["internalFacilitiesCount", "hospitals_10km"]
}
```

---

### `POST /predict`

Classifies a research centre into a quality tier.

**Request body** – all fields are optional at the API level; required fields are determined dynamically from the loaded model's selected features:

```json
{
  "internalFacilitiesCount": 20,
  "hospitals_10km": 5,
  "pharmacies_10km": 12,
  "facilityDiversity_10km": 0.75,
  "facilityDensity_10km": 0.45
}
```

**Success response:**
```json
{
  "predicted_cluster": 0,
  "predicted_category": "Premium",
  "features_used": ["internalFacilitiesCount", "hospitals_10km"]
}
```

**Cluster-to-category mapping:**

| Cluster | Category |
|---|---|
| 0 | Premium |
| 1 | Standard |
| 2 | Basic |

**Error – missing required features (400):**
```json
{
  "error": "Missing required features",
  "missing": ["hospitals_10km"],
  "required": ["internalFacilitiesCount", "hospitals_10km"],
  "received": ["internalFacilitiesCount"]
}
```

**Error – model not loaded (503):**
```json
{
  "detail": "Model not loaded"
}
```

---

## ML Pipeline Details

The training pipeline (`train.py`) is a scikit-learn `Pipeline` with three steps:

```
Raw features → StandardScaler → VarianceThreshold → KMeans
```

| Step | Purpose |
|---|---|
| `StandardScaler` | Normalises features to zero mean and unit variance |
| `VarianceThreshold` | Removes any features with near-zero variance (prevents degenerate clusters) |
| `KMeans` | Clusters centres into 3 groups using 20 random initialisations |

**Hyperparameters (fixed):**

| Parameter | Value |
|---|---|
| `n_clusters` | 3 |
| `n_init` | 20 |
| `random_state` | 42 |

**Evaluation metric:** Silhouette Score (range –1 to 1; higher is better).

**Feature combinations explored:** All 31 non-empty subsets of the 5 candidate features, from single-feature models up to the full 5-feature model.

**MLflow logging per run:**
- Parameters: `random_state`, `n_init`, `n_clusters`, `selected_features`, `n_features_selected`
- Metrics: `silhouette_score`
- Artifact: full serialised scikit-learn pipeline
- Model registry: registered as `FEM` (Facility Environment Model)

---

## Data Description

The dataset (`research_centers.csv`) contains 50 research centres across multiple UK cities.

| Column | Type | Description |
|---|---|---|
| `researchCenterId` | String | Unique identifier |
| `researchCenterName` | String | Name of the research centre |
| `city` | String | City of the research centre |
| `latitude` | Float | Geographic latitude |
| `longitude` | Float | Geographic longitude |
| `internalFacilitiesCount` | Integer | Number of facilities within the centre |
| `hospitals_10km` | Integer | Number of hospitals within 10 km |
| `pharmacies_10km` | Integer | Number of pharmacies within 10 km |
| `facilityDiversity_10km` | Float | Diversity index of facility types within 10 km |
| `facilityDensity_10km` | Float | Density of facilities within 10 km |

---

## Configuration Reference

### `kind-config.yaml`

Defines a 3-node Kind cluster (1 control-plane, 2 workers) using Kubernetes v1.31.0. Host ports `8080` and `8081` are mapped to NodePort services `30000` and `30001` on the control-plane node.

### `mlflow-values.yaml`

Helm values for the MLflow deployment:
- PostgreSQL enabled with credentials `mlflow/password` on database `mlflow`
- Artifact root: `/tmp/mlartifacts`
- Artifact serving enabled via `serve-artifacts` flag

### `k8s-manifest.yaml`

Kubernetes resources for the inference API:
- **Deployment:** 1 replica, `imagePullPolicy: Never` (uses Kind local cache), `MLFLOW_TRACKING_URI` set to in-cluster MLflow service
- **Service:** `ClusterIP` on port 80 targeting pod port 5001

> **Note:** The manifest currently defines `type: NodePort` and `type: ClusterIP` on the same Service. The effective type is `ClusterIP` (last definition wins). Access is via `kubectl port-forward`.

---

## Troubleshooting

**MLflow health check fails (`curl http://localhost:8081/health`):**
- Confirm the port-forward is running: `ps aux | grep port-forward`
- Re-run: `kubectl port-forward svc/mlflow -n mlflow 8081:80 &`
- Check pod status: `kubectl get pods -n mlflow`

**`kind load` succeeds but pod uses old image:**
- Ensure `imagePullPolicy: Never` is set in `k8s-manifest.yaml`
- Delete and re-create the deployment: `kubectl rollout restart deployment/research-center-server`

**Model not loading in FastAPI (`❌ Error loading model`):**
- Verify training completed successfully and model is registered: `curl -s http://localhost:8081/api/2.0/mlflow/registered-models/get-latest-versions -d '{"name": "FEM"}'`
- Check that `MLFLOW_TRACKING_URI` inside the pod matches the in-cluster MLflow service URL

**Airflow pod stuck or failing:**
- Check pod logs: `kubectl logs -n airflow -l app=research-center-training`
- Verify the `research-center-mlops:latest` image is loaded in Kind: `docker exec advanced-mlops-control-plane crictl images | grep mlops`

**Grafana not accessible at port 3000:**
- Re-run the port-forward: `kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80 &`
- Check Grafana pod status: `kubectl get pods -n monitoring | grep grafana`
