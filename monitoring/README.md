# Monitoring (Part 4: Prometheus & Grafana)

Plain Kubernetes manifests for running Prometheus and Grafana on the local minikube cluster without heavy operators or Helm charts.

---

## Architecture & Resources

- **Namespace**: `wheat-mlops`
- **Memory & Resource Limits**:
  - **Prometheus**: Requests `100m` CPU / `256Mi` Memory; Limits `500m` CPU / `512Mi` Memory.
  - **Grafana**: Requests `100m` CPU / `128Mi` Memory; Limits `500m` CPU / `256Mi` Memory.
- **Storage**: Uses `emptyDir` volumes (no persistent volume claims required).
- **Retention**: TSDB retention set to 2 days (`--storage.tsdb.retention.time=2d`).
- **Images**: Pinned Docker Hub images:
  - `prom/prometheus:v2.51.2`
  - `grafana/grafana:10.4.2`

---

## File Layout

```
monitoring/
├── README.md                           # Documentation, handover details, and metric contract
├── 10-prometheus-config.yaml           # ConfigMap with prometheus.yml & alert_rules.yml
├── 11-prometheus.yaml                  # Prometheus Deployment & ClusterIP Service
├── 20-grafana-config.yaml              # ConfigMaps: datasources, dashboard provider, dashboards
├── 21-grafana.yaml                     # Grafana Deployment & ClusterIP Service
└── dashboards/
    └── wheat-api-overview.json         # Raw Grafana dashboard JSON (Traffic, Latency, Errors, Quality)
```

---

## Exact `kubectl` Commands (Handover for Saatvik / Part 5)

### 1. Apply Monitoring Stack

```bash
kubectl apply -f monitoring/
```

*(Or apply individually in order:)*
```bash
kubectl apply -f monitoring/10-prometheus-config.yaml
kubectl apply -f monitoring/11-prometheus.yaml
kubectl apply -f monitoring/20-grafana-config.yaml
kubectl apply -f monitoring/21-grafana.yaml
```

### 2. Verify Pod & Service Status

```bash
kubectl get pods,svc -n wheat-mlops -l "app in (prometheus, grafana, wheat-api)"
```

### 3. Access UIs Locally via Port-Forward

- **Prometheus**:
  ```bash
  kubectl port-forward -n wheat-mlops svc/prometheus 9090:9090
  ```
  Open: [http://localhost:9090](http://localhost:9090)
  - Targets page: [http://localhost:9090/targets](http://localhost:9090/targets)
    - `prometheus` target will show as **UP**.
    - `wheat-api` target (`wheat-api.wheat-mlops.svc:80/metrics`) will show as **DOWN** until Part 8 API is implemented.
  - Alerts page: [http://localhost:9090/alerts](http://localhost:9090/alerts)

- **Grafana**:
  ```bash
  kubectl port-forward -n wheat-mlops svc/grafana 3000:3000
  ```
  Open: [http://localhost:3000](http://localhost:3000)
  - Anonymous admin access is enabled by default for convenience, or login with `admin` / `admin`.
  - The **Wheat Detection API - Performance & Quality** dashboard is automatically provisioned under the **Wheat Detection** folder.

---

## Services & Ports Summary

| Service Name | Namespace | Type | Port | Target Port | Description |
|---|---|---|---|---|---|
| `prometheus` | `wheat-mlops` | ClusterIP | `9090` | `9090` | Prometheus server & TSDB |
| `grafana` | `wheat-mlops` | ClusterIP | `3000` | `3000` | Grafana visual analytics & dashboards |

---

## Prometheus Scrape Targets & Alert Rules

### Scrape Targets:
1. `prometheus`: `localhost:9090` (Self-scrape, proves operational status).
2. `wheat-api`: `wheat-api.wheat-mlops.svc:80` with path `/metrics` (Shows down until Part 8 backend API provides `/metrics`).

### Alert Rules:
- `WheatApiHighP95Latency`: Triggers when 95th percentile latency exceeds 500ms over 5 minutes.
- `WheatApiHighErrorRate`: Triggers when HTTP error percentage exceeds 5% over 5 minutes.
- `WheatApiDown`: Triggers when the wheat-api endpoint is unreachable for 2 minutes.

---

## Metric Contract for Backend Writer (Part 8)

The backend service (`wheat-api`) is expected to expose a standard Prometheus text endpoint at `/metrics` with the following metrics:

### 1. `wheat_api_requests_total`
- **Type**: Counter
- **Description**: Total number of HTTP requests handled by the API.
- **Labels**:
  - `endpoint`: Request path (e.g., `/predict`, `/health`).
  - `method`: HTTP method (e.g., `POST`, `GET`).
  - `status`: HTTP status code as string (e.g., `"200"`, `"400"`, `"500"`).

### 2. `wheat_api_errors_total`
- **Type**: Counter
- **Description**: Total count of failed or erroneous requests.
- **Labels**:
  - `endpoint`: Request path (e.g., `/predict`).
  - `error_type`: Error classification (e.g., `"bad_input"`, `"inference_failure"`, `"timeout"`, `"internal_error"`).
  - `status`: HTTP status code (e.g., `"422"`, `"500"`).

### 3. `wheat_api_request_duration_seconds`
- **Type**: Histogram
- **Description**: Latency duration of API request processing in seconds.
- **Labels**:
  - `endpoint`: Request path (e.g., `/predict`).
- **Recommended Buckets**:
  `[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]`

### 4. `wheat_api_detections_per_image`
- **Type**: Histogram (or Summary)
- **Description**: Number of wheat heads detected per image by YOLO.
- **Labels**:
  - `model_version`: Identifier/tag of the model checkpoint (e.g., `"yolov8n-0.1.0"`).
- **Recommended Buckets**:
  `[0, 1, 5, 10, 20, 30, 50, 75, 100, 150]`

### 5. `wheat_api_detection_confidence`
- **Type**: Histogram (or Summary)
- **Description**: Confidence scores of detected bounding boxes (0.0 to 1.0).
- **Labels**:
  - `model_version`: Identifier/tag of the model checkpoint.
- **Recommended Buckets**:
  `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]`
