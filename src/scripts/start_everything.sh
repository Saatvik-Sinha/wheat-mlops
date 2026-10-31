#!/bin/sh
set -eu

# Run from any directory; manifests, Dockerfile, model, and .env live at repo root.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

IMAGE='wheat-mlops:0.2.0'
NAMESPACE='wheat-mlops'
PUBLIC_HEALTH_URL='https://oxychloric-carma-subgelatinously.ngrok-free.dev/health'
FORWARD_PIDS=''

cleanup() {
    for pid in $FORWARD_PIDS; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

for command in docker minikube curl ngrok; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        exit 1
    fi
done

if [ ! -f .env ]; then
    echo 'Missing .env in the project root.' >&2
    exit 1
fi
if [ ! -s data/working/trained/best.pt ]; then
    echo 'Missing trained checkpoint: data/working/trained/best.pt' >&2
    exit 1
fi
if ! grep -q '^NGROK_AUTHTOKEN=.' .env || ! grep -q '^NGROK_DOMAIN=.' .env; then
    echo '.env must define NGROK_AUTHTOKEN and NGROK_DOMAIN for the public health check.' >&2
    exit 1
fi

echo 'Starting minikube (Docker driver)...'
minikube config set memory 4096
minikube start --driver=docker --cpus=6 --memory=4096
minikube kubectl -- apply -f k8s/00-namespace.yaml

echo 'Removing old API containers and image copies...'
docker ps -aq --filter "ancestor=$IMAGE" | while IFS= read -r container_id; do
    [ -z "$container_id" ] || docker rm -f "$container_id" >/dev/null
done
minikube kubectl -- delete deployment wheat-api -n "$NAMESPACE" --ignore-not-found
minikube image rm "$IMAGE" >/dev/null 2>&1 || true
docker image rm -f "$IMAGE" >/dev/null 2>&1 || true

echo 'Building and loading the fresh API image...'
docker build -f docker/Dockerfile --target serve -t "$IMAGE" .
minikube image load "$IMAGE"

echo 'Applying application and monitoring manifests...'
minikube kubectl -- create secret generic wheat-api-secrets \
    -n "$NAMESPACE" --from-env-file=.env --dry-run=client -o yaml |
    minikube kubectl -- apply -f -
minikube kubectl -- apply -f k8s/
minikube kubectl -- apply -f monitoring/
minikube kubectl -- rollout status deployment/wheat-api -n "$NAMESPACE" --timeout=300s
minikube kubectl -- rollout status deployment/prometheus -n "$NAMESPACE" --timeout=180s
minikube kubectl -- rollout status deployment/grafana -n "$NAMESPACE" --timeout=180s

echo 'Starting local port forwards...'
minikube kubectl -- port-forward -n "$NAMESPACE" svc/wheat-api 8000:80 >/tmp/wheat-api-port-forward.log 2>&1 &
FORWARD_PIDS="$FORWARD_PIDS $!"
minikube kubectl -- port-forward -n "$NAMESPACE" svc/prometheus 9090:9090 >/tmp/wheat-prometheus-port-forward.log 2>&1 &
FORWARD_PIDS="$FORWARD_PIDS $!"
minikube kubectl -- port-forward -n "$NAMESPACE" svc/grafana 3000:3000 >/tmp/wheat-grafana-port-forward.log 2>&1 &
FORWARD_PIDS="$FORWARD_PIDS $!"
ngrok http 8000 --url=https://oxychloric-carma-subgelatinously.ngrok-free.dev \
    >/tmp/wheat-ngrok.log 2>&1 &
NGROK_PID=$!
FORWARD_PIDS="$FORWARD_PIDS $NGROK_PID"

echo 'Waiting for the public backend health endpoint...'
attempt=0
until curl --fail --silent --show-error "$PUBLIC_HEALTH_URL"; do
    attempt=$((attempt + 1))
    if ! kill -0 "$NGROK_PID" 2>/dev/null; then
        echo 'ngrok exited before the public health check succeeded. See /tmp/wheat-ngrok.log' >&2
        exit 1
    fi
    if [ "$attempt" -ge 36 ]; then
        echo "Public health check failed: $PUBLIC_HEALTH_URL" >&2
        echo 'Check pod logs with: minikube kubectl -- logs -n wheat-mlops deployment/wheat-api' >&2
        echo 'Check the tunnel with: tail -f /tmp/wheat-ngrok.log' >&2
        exit 1
    fi
    sleep 5
done
printf '\nBackend is live. Local endpoints: API http://localhost:8000/health, Prometheus http://localhost:9090, Grafana http://localhost:3000\n'
echo 'Port forwards and ngrok remain active until Ctrl-C. Logs: /tmp/wheat-*.log'

# Keep the forwarding subprocesses alive for the duration of this script.
wait
