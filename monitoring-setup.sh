helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

pkill -f "kubectl port-forward svc/prometheus-grafana" || true

helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
 --namespace monitoring --create-namespace \
 --set grafana.adminPassword=admin \
 --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
 --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

 kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=grafana -n monitoring --timeout=600s

 kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80 &