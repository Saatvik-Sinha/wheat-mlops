minikube kubectl -- create secret generic wheat-api-secrets \
  -n wheat-mlops --from-env-file=.env


minikube kubectl -- apply -f k8s/
minikube kubectl -- rollout restart deployment/wheat-api -n wheat-mlops





# docker build --no-cache -f docker/Dockerfile -t wheat-mlops:0.1.0 .
docker build -f docker/Dockerfile -t wheat-mlops:0.2.0 .


minikube config set memory 4096
# minikube stop
minikube start --driver=docker --cpus=6 --memory=4096
minikube image load wheat-mlops:0.2.0
minikube kubectl -- apply --dry-run=client -f k8s/
minikube kubectl -- apply -f k8s/
minikube kubectl -- apply -f monitoring/
minikube kubectl -- rollout status deployment/wheat-api -n wheat-mlops
minikube kubectl -- get pods,svc -n wheat-mlops


# API (Terminal A):
minikube kubectl -- port-forward svc/wheat-api 8000:80 -n wheat-mlops &
# Prometheus (Terminal B):
minikube kubectl -- port-forward svc/prometheus 9090:9090 -n wheat-mlops &
# Grafana (Terminal C):
minikube kubectl -- port-forward svc/grafana 3000:3000 -n wheat-mlops &

ngrok http 8000 --url=https://oxychloric-carma-subgelatinously.ngrok-free.dev


# Can open http://localhost:8080
# Which will show a .yaml and .test file
# Then ctrl+c on terminal because kubectl -n wheat-mlops port-forward svc/wheat-api 8080:80 command never closes on its own

# After that
kubectl apply -f k8s/jobs/smoke-test.yaml
kubectl -n wheat-mlops wait --for=condition=complete job/wheat-smoke-test --timeout=180s
kubectl -n wheat-mlops logs job/wheat-smoke-test

# Jobs can’t be edited, so before re-running it, delete it:
# kubectl -n wheat-mlops delete job wheat-smoke-test --ignore-not-found.

# Run
# minikube stop
