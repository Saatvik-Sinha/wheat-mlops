#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=wheat-mlops
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DB_FILE="$ROOT/mlflow.db"
COPY_DB=0

case "${1:-}" in
  "") ;;
  --copydb) COPY_DB=1 ;;
  -h|--help)
    echo "Usage: $0 [--copydb]"
    echo "  --copydb  Replace the API pod's /tmp/mlflow.db with this repo's mlflow.db"
    exit 0
    ;;
  *) echo "Usage: $0 [--copydb]" >&2; exit 2 ;;
esac

if [[ "$COPY_DB" == 1 && ! -f "$DB_FILE" ]]; then
  echo "Missing local database: $DB_FILE" >&2
  exit 1
fi

if command -v docker >/dev/null; then
  MEMORY_LIMIT="$(docker inspect minikube --format '{{.HostConfig.Memory}}' 2>/dev/null || echo 0)"
  minikube config get memory
  echo MEMORY_LIMIT $MEMORY_LIMIT
  if [[ "$MEMORY_LIMIT" =~ ^[0-9]+$ ]] && (( MEMORY_LIMIT > 0 && MEMORY_LIMIT < 4294967296 )); then
    echo "Minikube has less than 4 GiB allocated. Increase its memory and restart it first." >&2
    exit 1
  fi
fi

kube() { minikube kubectl -- "$@"; }
kube rollout status deployment/wheat-api -n "$NAMESPACE" --timeout=180s
POD="$(kube get pods -n "$NAMESPACE" -l app=wheat-api -o jsonpath='{.items[0].metadata.name}')"
[[ -n "$POD" ]] || { echo "No wheat-api pod found" >&2; exit 1; }

if [[ "$COPY_DB" == 1 ]]; then
  echo "Copying local mlflow.db into $POD:/tmp/mlflow.db (this replaces the API's current DB file)." >&2
  kube cp "$DB_FILE" -n "$NAMESPACE" "$POD:/tmp/mlflow.db"
fi

kube exec -n "$NAMESPACE" "$POD" -- sh -c \
  'test "$MLFLOW_TRACKING_URI" = "sqlite:////tmp/mlflow.db" && test -f /tmp/mlflow.db' \
  || { echo "API pod is not using /tmp/mlflow.db; refusing to start the UI on a different DB." >&2; exit 1; }

UI_LOG="$(mktemp)"
FORWARD_LOG="$(mktemp)"
UI_STARTED=0
FORWARD_PID=
cleanup() {
  [[ -z "$FORWARD_PID" ]] || kill "$FORWARD_PID" 2>/dev/null || true
  if [[ "$UI_STARTED" == 1 ]]; then
    kube exec -n "$NAMESPACE" "$POD" -- sh -c \
      'if [ -f /tmp/mlflow-ui.pid ]; then kill "$(cat /tmp/mlflow-ui.pid)" 2>/dev/null || true; rm -f /tmp/mlflow-ui.pid; fi' \
      >/dev/null 2>&1 || true
  fi
  rm -f "$UI_LOG" "$FORWARD_LOG"
}
trap cleanup EXIT INT TERM

START_RESULT="$(kube exec -n "$NAMESPACE" "$POD" -- sh -c '
  if python -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:5000/\", timeout=2)" >/dev/null 2>&1; then
    echo existing
  else
    if [ -f /tmp/mlflow-ui.pid ]; then kill "$(cat /tmp/mlflow-ui.pid)" 2>/dev/null || true; rm -f /tmp/mlflow-ui.pid; fi
    MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false \
      MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS=false \
      nohup mlflow server --host 0.0.0.0 --port 5000 --workers 1 \
      --backend-store-uri sqlite:////tmp/mlflow.db >/tmp/mlflow-ui.log 2>&1 </dev/null &
    echo $! >/tmp/mlflow-ui.pid
    echo started
  fi
')"
if [[ "$START_RESULT" == *started* ]]; then UI_STARTED=1; fi

READY=0
for _ in {1..30}; do
  if kube --request-timeout=5s exec -n "$NAMESPACE" "$POD" -- python -c \
    'import urllib.request; urllib.request.urlopen("http://127.0.0.1:5000/", timeout=2)' \
    >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "$READY" != 1 ]]; then
  echo "MLflow UI did not start. Server log from the pod:" >&2
  kube exec -n "$NAMESPACE" "$POD" -- sh -c 'cat /tmp/mlflow-ui.log 2>/dev/null || true' >&2 || true
  exit 1
fi

kube port-forward -n "$NAMESPACE" "pod/$POD" 5000:5000 >"$FORWARD_LOG" 2>&1 &
FORWARD_PID=$!
for _ in {1..30}; do
  if ! kill -0 "$FORWARD_PID" 2>/dev/null; then break; fi
  if command -v curl >/dev/null && curl --silent --fail --output /dev/null http://127.0.0.1:5000/; then
    echo "MLflow UI: http://localhost:5000 (Ctrl-C stops the port-forward and UI)"
    wait "$FORWARD_PID"
    exit $?
  fi
  sleep 1
done

echo "Port-forward failed:" >&2
cat "$FORWARD_LOG" >&2
exit 1
