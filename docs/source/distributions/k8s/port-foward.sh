kubectl port-forward svc/llama-stack-ui-service 8322:8322 &
kubectl port-forward svc/llama-stack-service 8321:8321 & 
kubectl port-forward svc/jaeger-dev-query 16686:16686 -n observability & 
kubectl port-forward svc/kube-prometheus-stack-1754270486-grafana 3000:3000 -n prometheus
