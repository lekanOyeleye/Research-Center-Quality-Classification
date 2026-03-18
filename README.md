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

> For the full exploratory data analysis and model training walkthrough, open `EDA_and_Model.ipynb`.

**Step 4 — Start the API**

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

**Step 1 — Build the image**

```bash
docker build -t research-center-quality-classification .
```

**Step 2 — Verify the image was created**

```bash
docker images | grep research-center-quality-classification
```

**Step 3 — Run the image locally**

```bash
docker run -d -p 8000:8000 research-center-quality-classification
```

**Step 4 — Test the running container**

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

## Extending the Endpoints

The current endpoint accepts inference data, returns a predicted cluster and category, and discards the request data immediately after. To support continuous retraining, the endpoint must be extended across four layers, each building on the previous one.

Layer 1 — Persistent Data Collection
The first extension is giving the system a memory. Every prediction request that passes through the endpoint contains real world data that reflects current patterns. Rather than discarding this data after each response, the endpoint is extended to log every request and its corresponding prediction to a PostgreSQL database. Over time this builds a dataset of real world inference data that can be used to retrain the model. Without this layer, continuous retraining is impossible because there is no data to train on.

Layer 2 — A Dedicated Retraining Endpoint
The second extension is adding a separate retraining endpoint to the API. This endpoint accepts a trigger — either from a human operator or an automated system — and initiates a retraining job in the background. Running retraining in the background is important because it allows the API to continue serving predictions without interruption while the new model is being trained. This endpoint connects directly to the logged data in PostgreSQL, fetches the most recent records, and retrains the model on that data.

Layer 3 — Hot Model Swapping
The third extension addresses how the newly trained model is loaded into production. In the current setup, updating the model requires rebuilding and redeploying the entire container. Instead, a model reload endpoint is added that instructs the running API to load the newly trained model from disk without any restart or redeployment. This means the transition from an old model to a new one is seamless and invisible to the end user.

Layer 4 — Automated Retraining Triggers
The fourth extension removes the need for manual intervention entirely. Rather than requiring someone to call the retraining endpoint manually, the system monitors its own state and triggers retraining automatically. There are three approaches to this. The simplest is schedule based retraining, where retraining fires on a fixed schedule such as every week regardless of data changes. A more intelligent approach is volume based retraining, where retraining triggers automatically once a certain number of new predictions have been collected — for example every one thousand predictions. The most sophisticated approach is drift based retraining, where the system monitors the statistical distribution of incoming data and triggers retraining only when that distribution shifts significantly away from what the model was originally trained on. This final approach ensures the model is only retrained when it genuinely needs to be updated.

The Result
When all four layers are combined, the endpoint evolves from a static prediction service into a self improving system. Real world data is continuously collected, the model is periodically retrained on the most recent patterns, the best performing model is automatically promoted to production, and the entire cycle runs without manual intervention. The model therefore improves over time as more real world data flows through the system.

## How to Commercialize and Scale the Solution 
Commercialization
The natural buyers are organisations that need to evaluate or compare research facilities at scale — such as government bodies, academic institutions, and research consultancies. Three commercial models apply: API as a Service where customers are charged per prediction or on a subscription, Software as a Service where a user interface removes the technical barrier for non-technical buyers, and a vertical product model where the solution is packaged for a specific industry commanding higher prices. Regardless of the model chosen, the commercial foundation requires API key management, usage metering, and automated billing through a tool such as Stripe.

Scaling With KServe
The current single container setup is insufficient for production. KServe is a purpose built model serving platform that runs on Kubernetes and replaces the manual FastAPI serving layer entirely. Registering the model with KServe automatically generates a standardised endpoint, handles scaling, manages model versions, and monitors predictions without additional code.
At the infrastructure layer KServe scales containers automatically based on traffic and can scale to zero when no requests are arriving, meaning compute costs are only incurred when predictions are being served. At the model management layer KServe handles multiple versions simultaneously using canary deployments, routing a small percentage of traffic to a new model before fully switching over. If the new model underperforms, KServe rolls back instantly. At the retraining layer KServe integrates directly with MLflow for model versioning and Kubeflow for pipeline orchestration, meaning a retrained model is picked up and served automatically without manual intervention. For monitoring, prediction metrics feed into Prometheus and Grafana while Evidently AI handles data drift detection and retraining alerts.

The Result
KServe transforms the solution into a fully automated, commercially viable platform. Models improve continuously, deployments are safe through canary releases and instant rollback, infrastructure costs are minimised through scale to zero, and the system is capable of serving thousands of customers simultaneously with minimal manual intervention.

























































