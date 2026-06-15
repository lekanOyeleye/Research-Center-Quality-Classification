#!/bin/bash


# check mlflow is working
curl -s -H "Content-Type: application/json" http://localhost:8081/api/2.0/mlflow/registered-models/get-latest-versions \
  -d '{"name": "FEM"}' | python3 -c "import sys, json; print(json.load(sys.stdin))"

# check airflow
kubectl get configmap airflow-dags -n airflow


# check the health 
curl -s http://localhost:5002/health || echo "API is not working"

# check monitoring is working
kubectl get pods -n monitoring | grep prometheus-grafana
echo "SYSTEM is READY you can trigger the airflow"