#!/bin/bash

# create airflow namespace and RBAC
kubectl create namespace airflow --dry-run=client -o yaml | kubectl apply -f -

cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: airflow
  name: pod-launcher
rules:
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch", "create", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
    namespace: airflow
    name: pod-launcher-binding
subjects:
- kind: ServiceAccount
  name: airflow-worker # Default SA for worker (or webserver in LocalExecutor)
  namespace: airflow
- kind: ServiceAccount
  name: airflow-scheduler
  namespace: airflow
- kind: ServiceAccount
  name: airflow-webserver
  namespace: airflow
roleRef:
  kind: Role
  name: pod-launcher
  apiGroup: rbac.authorization.k8s.io
EOF

kubectl create configmap airflow-dags --from-file=research_center_dag.py -n airflow --dry-run=client -o yaml | kubectl apply -f -
helm repo add apache-airflow https://airflow.apache.org
helm repo update

helm upgrade --install airflow apache-airflow/airflow \
  --namespace airflow \
  -f airflow-values.yaml \
  --version 1.16.0 \
  --timeout 15m

kubectl wait --for=condition=ready pod -l component=webserver -n airflow --timeout=600s

pkill -f "kubectl port-forward svc/airflow-webserver" || true
kubectl port-forward svc/airflow-webserver -n airflow 8080:8080 &

echo "Airflow UI at -> http://localhost:8080 (admin/admin)"