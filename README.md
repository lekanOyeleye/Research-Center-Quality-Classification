# Research Center Quality Classification

A machine learning system that classifies research center quality using KMeans clustering, served via a FastAPI endpoint and containerised with Docker.

---

## Project Structure

```
├── EDA_and_Model.ipynb          # Full exploratory data analysis and model training walkthrough
├── research_centers.csv
├── train.py                     # Summarised training script
├── app.py                       # FastAPI application
├── docker-compose.yml           # Docker Compose configuration
├── Dockerfile                   # Docker build instructions
├── requirements.txt             # Python dependencies
├── .gitignore
├── .dockerignore
└── model/
    └── artifacts/
        └── kmeans_pipeline_model.pkl   # Trained model file
```

---

## Quickstart — Run with Docker Compose

This is the recommended way to run the project. You only need `docker-compose.yml` and the model file.

**Step 1 — Start the API**

```bash
docker-compose up
```

Docker will automatically pull the image from Docker Hub and start the API on port 8000.

**Step 2 — Test a prediction**

Visit the interactive API docs in your browser:

```
http://localhost:8000/docs
```

Or send a prediction directly using curl:

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

Expected response:

```json
{"PredictedCluster": 1, "PredictedCategory": "Standard"}
```

**Step 3 — Stop the API**

```bash
docker-compose down
```

---

## Local Development Setup

Follow these steps if you want to run the project locally without Docker.

**Step 1 — Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Step 2 — Install dependencies**

```bash
python3 -m pip install -r requirements.txt
```

**Step 3 — Train the model**

```bash
python3 train.py
```

This saves the trained model to `model/artifacts/kmeans_pipeline_model.pkl`.

> For the full exploratory data analysis and model training walkthrough, open `EDA_and_Model.ipynb`. When the notebook is run it saves it in the main folder but using the train.py saves it into the model/artifact folder for further use.


**Step 4 — Start the API**

The process in the app.py explained. 

>The fastapi is instantiated, then the saved model(pipeline and the column names of the selected features) is loaded to the file. This saved model
 defines the expected input schema which is the *RecearchCenterData* class to match our selected features, sets up a health check endpoint, processes incoming prediction requests by converting input data into a DataFrame, uses the model to predict a cluster, maps it to a quality category, returns the result as JSON, and runs the API server using Uvicorn.

```bash
python3 app.py
```

**Step 5 — Test a prediction**

Visit `http://localhost:8000/docs` or run:

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

---

## Docker — Build and Push

Follow these steps if you want to build and publish your own Docker image.


**Step 1 — Build the Dockerfile**
>The Dockerfile builds a lightweight Python container, sets environment variables, installs dependencies, copies the app code, trains and saves the model, exposes port 8000, and runs the FastAPI application using Uvicorn.

**Step 2 — Build the image**

```bash
docker build -t research-center-quality-classification .
```

**Step 3 — Verify the image was created**

```bash
docker images | grep research-center-quality-classification
```

**Step 4 — Run the image locally**

```bash
docker run -d -p 8000:8000 research-center-quality-classification
```

**Step 5 — Test the running container**

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

---

## Docker Hub — Publishing Your Image

**Step 1 — Create a Docker Hub account and Personal Access Token**

If you do not have a Docker Hub account, create one at [hub.docker.com](https://hub.docker.com). Then generate a Personal Access Token (PAT) for secure authentication:

1. Log in to hub.docker.com
2. Go to **Account Settings → Security → Personal Access Tokens**
3. Click **New Access Token**
4. Give it a name, set an expiry, and choose **Read/Write** permissions
5. Copy the token — you will only see it once

**Step 2 — Log in from the terminal**

```bash
docker login
```

Enter your Docker Hub username and paste your Personal Access Token as the password.

**Step 3 — Tag the image**

Replace the image ID with your own from `docker images`:

```bash
docker tag 4090883bb3f6 consolejay/research-center-quality-classification:v1
```

**Step 4 — Push the image**

```bash
docker push consolejay/research-center-quality-classification:v1
```

**Step 5 — Verify it is available**

```bash
docker pull consolejay/research-center-quality-classification:v1
```

---

## Docker Image

The pre-built image is available on Docker Hub and used as image in the docker-compose.yml:

```
consolejay/research-center-quality-classification:v1
```

## How Can the Endpoint Be Extended for Continuous Retraining?

The FastAPI endpoint has already been containerised and pushed to Docker Hub. Extending it for continuous retraining requires five additions.
First, Amazon RDS for PostgreSQL is introduced to persist every inference request and prediction, giving the system a permanent memory it currently lacks.
Second, DVC is introduced alongside PostgreSQL to version every training data snapshot. Each time retraining triggers, the data fetched from PostgreSQL is saved as a versioned snapshot, pushed to Amazon S3, and linked to the corresponding Git commit. This means any model from any point in time can be reproduced exactly by checking out the Git commit and running a single DVC pull command — solving the reproducibility problem that PostgreSQL alone cannot address.
Third, a retraining endpoint is added to the API that accepts a manual or automated trigger and runs the retraining pipeline in the background without interrupting live predictions. Fourth, MLflow is introduced to version every retrained model, track silhouette scores across runs, and manage model stages. Every MLflow run is linked directly to its DVC data snapshot, creating a complete lineage from raw data through to deployed model. This ensures only a better performing model ever reaches production and any version can be rolled back instantly. Fifth, retraining is automated by triggering every a giben number of predictions, replacing the need for manual intervention entirely. Together these additions transform the static endpoint into a fully reproducible, self improving system where real world data continuously shapes model performance and every decision can be traced and recreated.

## How to Commercialise and Scale the Solution
### Commercialisation
The most natural commercial fit is an API as a Service model targeting organisations that assess research facility quality at scale — government funding bodies, academic institutions, and infrastructure consultancies. Each customer receives a unique API key, usage is metered per prediction, and billing is automated through Stripe or the AWS Marketplace Metering Service for enterprise customers. A Software as a Service layer wrapping the API in a dashboard would extend the customer base to non-technical decision makers who need cluster visualisations and downloadable reports without interacting with the endpoint directly.

### Scaling — Two Paths Depending on Infrastructure Preference
At scale the single Docker container is insufficient and the solution can be scaled through two complementary approaches.
The first is Kubernetes with KServe. The Docker image stored in Amazon ECR is deployed onto Amazon EKS where KServe replaces the FastAPI serving layer entirely. KServe automatically generates a standardised prediction endpoint, scales containers up and down with traffic, and scales to zero during off-peak periods eliminating unnecessary compute costs. When a retrained model is promoted to production in MLflow, KServe picks it up automatically through a canary deployment  routing a small percentage of traffic to the new model before fully switching over. If the new model underperforms, rollback is instant with a single command.

The second is AWS SageMaker. For teams preferring fully managed infrastructure, Amazon SageMaker Endpoints replace KServe for model serving, SageMaker Pipelines replace Kubeflow for pipeline orchestration, and SageMaker Model Monitor replaces Evidently AI for drift detection. SageMaker connects natively to the MLflow model registry meaning a promoted model triggers an automatic endpoint update without manual intervention. Amazon CloudWatch handles infrastructure monitoring while SageMaker Model Monitor compares live prediction distributions against the training baseline and raises alerts when drift is detected.
Both approaches support canary deployments, instant rollback, auto scaling, and continuous retraining. The KServe path offers more control and avoids vendor lock-in. The SageMaker path reduces operational overhead for teams who prefer managed services over self hosted infrastructure.
The Result
Whether deployed through KServe on EKS or through SageMaker managed services, the research center quality classification system evolves from a static Docker container into a fully automated commercial platform. Models improve continuously from real world data, deployments are safe through canary releases and instant rollback, infrastructure costs are minimised through scale to zero, and the system is capable of serving thousands of customers simultaneously with minimal manual intervention.
























































