# Part 3 - Docker and Kubernetes (initial setup)

Image: `wheat-mlops:0.1.0` (CPU only, no model or data inside).
Cluster: minikube, namespace `wheat-mlops`, no registry (image is loaded into minikube).

```
docker build -f docker/Dockerfile -t wheat-mlops:0.1.0 --build-arg PYTHON_VERSION=<x.y> .
docker run --rm wheat-mlops:0.1.0                      # smoke test, expect SMOKE TEST PASSED

minikube start --driver=docker --cpus=2 --memory=3072
minikube image load wheat-mlops:0.1.0
kubectl apply -f k8s/
kubectl -n wheat-mlops rollout status deployment/wheat-api
kubectl -n wheat-mlops port-forward svc/wheat-api 8080:80

kubectl -n wheat-mlops delete job wheat-smoke-test --ignore-not-found
kubectl apply -f k8s/jobs/smoke-test.yaml
kubectl -n wheat-mlops wait --for=condition=complete job/wheat-smoke-test --timeout=180s
kubectl -n wheat-mlops logs job/wheat-smoke-test
```

The Deployment is a PLACEHOLDER (`python -m http.server`); Part 8 replaces its command
and probe paths with the real API.
