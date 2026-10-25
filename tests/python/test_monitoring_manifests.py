"""Static validation tests for Prometheus and Grafana monitoring manifests."""

import json
from pathlib import Path

import pytest
import yaml

MONITORING_DIR = Path(__file__).resolve().parents[2] / "monitoring"
NAMESPACE = "wheat-mlops"


@pytest.fixture(scope="module")
def monitoring_docs():
    found = []
    for path in sorted(MONITORING_DIR.glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if doc:
                found.append(doc)
    assert found, f"no manifests found in {MONITORING_DIR}"
    return found


@pytest.fixture(scope="module")
def dashboard_json():
    dash_path = MONITORING_DIR / "dashboards" / "wheat-api-overview.json"
    assert dash_path.is_file(), f"dashboard JSON missing at {dash_path}"
    data = json.loads(dash_path.read_text(encoding="utf-8"))
    return data


def _of_kind(docs, kind):
    return [d for d in docs if d["kind"] == kind]


def _containers(doc):
    return doc["spec"]["template"]["spec"]["containers"]


def test_every_manifest_has_basic_fields(monitoring_docs):
    for doc in monitoring_docs:
        assert doc.get("apiVersion") and doc.get("kind"), f"missing apiVersion/kind in {doc}"
        assert doc["metadata"]["name"], f"missing metadata.name in {doc}"


def test_namespace_is_wheat_mlops(monitoring_docs):
    for doc in monitoring_docs:
        assert doc["metadata"].get("namespace") == NAMESPACE, (
            f"manifest {doc['metadata']['name']} not in {NAMESPACE}"
        )


def test_workload_images_pinned(monitoring_docs):
    deployments = _of_kind(monitoring_docs, "Deployment")
    assert len(deployments) == 2, "expected deployments for prometheus and grafana"
    for d in deployments:
        for c in _containers(d):
            image = c["image"]
            assert ":" in image, f"image not tagged: {image}"
            assert not image.endswith(":latest"), f"latest image tag used: {image}"


def test_workloads_have_probes_and_hardening(monitoring_docs):
    deployments = _of_kind(monitoring_docs, "Deployment")
    for d in deployments:
        for c in _containers(d):
            assert "readinessProbe" in c, f"missing readinessProbe in {d['metadata']['name']}"
            assert "livenessProbe" in c, f"missing livenessProbe in {d['metadata']['name']}"
            assert c["securityContext"]["runAsNonRoot"] is True
            assert c["securityContext"]["allowPrivilegeEscalation"] is False
            for k in ("requests", "limits"):
                assert {"cpu", "memory"} <= set(c["resources"][k])


def test_prometheus_retention_and_storage(monitoring_docs):
    (prom_deploy,) = [
        d for d in _of_kind(monitoring_docs, "Deployment") if d["metadata"]["name"] == "prometheus"
    ]
    container = _containers(prom_deploy)[0]
    args = container.get("args", [])
    assert any("--storage.tsdb.retention.time=" in arg for arg in args)

    # EmptyDir volumes used
    volumes = prom_deploy["spec"]["template"]["spec"]["volumes"]
    storage_vols = [v for v in volumes if v["name"] == "storage"]
    assert len(storage_vols) == 1
    assert "emptyDir" in storage_vols[0]


def test_prometheus_scrape_configs_and_alerts(monitoring_docs):
    cm_docs = [
        d
        for d in _of_kind(monitoring_docs, "ConfigMap")
        if d["metadata"]["name"] == "prometheus-config"
    ]
    assert len(cm_docs) == 1
    prom_cm = cm_docs[0]
    prom_cfg = yaml.safe_load(prom_cm["data"]["prometheus.yml"])
    scrape_jobs = {job["job_name"]: job for job in prom_cfg["scrape_configs"]}

    assert "prometheus" in scrape_jobs
    assert "wheat-api" in scrape_jobs

    wheat_job = scrape_jobs["wheat-api"]
    assert wheat_job.get("metrics_path") == "/metrics"
    targets = [t for sc in wheat_job["static_configs"] for t in sc["targets"]]
    assert "wheat-api.wheat-mlops.svc:80" in targets

    rules_yaml = prom_cm["data"]["alert_rules.yml"]
    alert_rules = yaml.safe_load(rules_yaml)
    all_alerts = [
        r["alert"]
        for group in alert_rules["groups"]
        for r in group.get("rules", [])
        if "alert" in r
    ]
    assert any("Latency" in a for a in all_alerts), f"no latency alert found: {all_alerts}"


def test_grafana_datasource_and_provider_provisioning(monitoring_docs):
    cm_map = {d["metadata"]["name"]: d for d in _of_kind(monitoring_docs, "ConfigMap")}
    assert "grafana-datasources" in cm_map
    assert "grafana-dashboard-provider" in cm_map

    datasources_cfg = yaml.safe_load(cm_map["grafana-datasources"]["data"]["datasources.yaml"])
    ds_list = datasources_cfg["datasources"]
    prom_ds = [ds for ds in ds_list if ds["type"] == "prometheus"]
    assert prom_ds
    assert "prometheus.wheat-mlops.svc:9090" in prom_ds[0]["url"]


def test_grafana_dashboard_json(dashboard_json):
    assert dashboard_json.get("title")
    panels = dashboard_json.get("panels", [])

    # Collect all titles and expressions
    all_titles = []
    all_exprs = []
    for p in panels:
        all_titles.append(p.get("title", ""))
        for t in p.get("targets", []):
            if "expr" in t:
                all_exprs.append(t["expr"])

    joined_exprs = " ".join(all_exprs)
    # Check coverage of required metrics
    assert "wheat_api_requests_total" in joined_exprs
    assert "wheat_api_request_duration_seconds" in joined_exprs
    assert "wheat_api_errors_total" in joined_exprs
    assert "wheat_api_detections_per_image" in joined_exprs
    assert "wheat_api_detection_confidence" in joined_exprs
