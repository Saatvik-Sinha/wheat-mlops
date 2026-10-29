#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DB_FILE="$ROOT/mlflow.db"
MLFLOW="$ROOT/.venv/bin/mlflow"

[[ -f "$DB_FILE" ]] || { echo "Missing local database: $DB_FILE" >&2; exit 1; }
[[ -x "$MLFLOW" ]] || { echo "Missing project MLflow CLI: $MLFLOW" >&2; exit 1; }
if command -v curl >/dev/null && curl --silent --fail --output /dev/null http://127.0.0.1:5000/; then
  echo "MLflow is already available at http://localhost:5000"
  exit 0
fi

echo "MLflow UI: http://localhost:5000 (Ctrl-C stops it)"
exec env MLFLOW_DISABLE_AGENT_HINT=1 "$MLFLOW" server \
  --host 127.0.0.1 --port 5000 --workers 1 \
  --backend-store-uri "sqlite:///$DB_FILE"
