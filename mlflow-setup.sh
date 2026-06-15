#!/bin/bash


echo "1. Installing MLflow..."

helm upgrade --install mlflow community-charts/mlflow \
  --namespace mlflow --create-namespace \
  --set backendStore.databaseMigration=true \
  --set postgresql.enabled=true \
  --set postgresql.auth.username=mlflow \
  --set postgresql.auth.password=password \
  --set postgresql.auth.database=mlflow \
  --set artifactRoot.proxiedArtifactStorage=true \
  --set artifactRoot.defaultArtifactRoot="mlflow-artifacts:/mlartifacts" \
  --set extraEnvVars.MLFLOW_SERVER_ALLOWED_HOSTS="*" \
  --set extraArgs.artifacts-destination="/tmp/mlartifacts" \
  --set extraFlags[0]=serveArtifacts \
  --wait --timeout=10m

kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=mlflow --timeout=300s

kubectl port-forward svc/mlflow -n mlflow 8081:80 &

echo "MLFLOW is readdy at :: http://localhost:8081"
