"""M3 integration with the API (Prometheus gauges), the Grafana dashboard and the DVC stage."""

import json
from pathlib import Path

import pytest
import yaml

try:
    from fastapi.testclient import TestClient
    from wheat_mlops import api
except (ImportError, RuntimeError) as exc:  # serving extras not installed
    pytest.skip(f"serving dependencies missing: {exc}", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
CSV = (
    "condition,corruption,severity,matched_ratio,mean_iou,center_disp_px,abs_count_diff,"
    "width_change_pct,height_change_pct,conf_diff\n"
    "gaussian_blur_s1,gaussian_blur,1,0.95,0.82,1.5,2.0,4.0,3.0,-0.01\n"
    "gaussian_blur_s5,gaussian_blur,5,0.40,0.61,6.5,30.0,15.0,12.0,\n"
)


class _Detector:
    def detect(self, image, conf, iou):
        return []


@pytest.fixture()
def summary(tmp_path, monkeypatch):
    path = tmp_path / "condition_summary.csv"
    monkeypatch.setattr(api, "STABILITY_SUMMARY_PATH", str(path))
    return path


@pytest.fixture()
def client():
    return TestClient(api.create_app(detector=_Detector()), raise_server_exceptions=False)


def test_gauges_exported_and_refreshed_on_change(client, summary):
    assert "wheat_api_stability_matched_ratio{" not in client.get("/metrics").text  # no file yet
    summary.write_text(CSV, encoding="utf-8")
    text = client.get("/metrics").text
    assert 'wheat_api_stability_matched_ratio{corruption="gaussian_blur",severity="5"} 0.4' in text
    assert 'wheat_api_stability_center_disp_px{corruption="gaussian_blur",severity="1"} 1.5' in text
    # the empty conf_diff cell is skipped instead of exported as NaN
    assert 'wheat_api_stability_conf_diff{corruption="gaussian_blur",severity="5"}' not in text
    summary.write_text(CSV.replace("0.40", "0.35"), encoding="utf-8")
    assert 'severity="5"} 0.35' in client.get("/metrics").text


def test_stability_summary_endpoint(client, summary):
    assert client.get("/stability/summary").status_code == 404
    summary.write_text(CSV, encoding="utf-8")
    body = client.get("/stability/summary").json()
    assert [c["severity"] for c in body["conditions"]] == [1, 5]
    assert body["conditions"][1]["mean_iou"] == 0.61


def test_stability_requests_are_not_counted_as_traffic(client, summary):
    client.get("/stability/summary")
    assert "/stability/summary" not in client.get("/metrics").text


def test_dashboard_is_valid_and_embedded_in_configmap():
    dashboard = json.loads((ROOT / "monitoring/dashboards/wheat-stability.json").read_text())
    assert dashboard["uid"] == "wheat-stability" and dashboard["panels"]
    exprs = [t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])]
    assert exprs and all(
        e.startswith(("wheat_api_stability_", "avg(wheat_api_stability_")) for e in exprs
    )
    known = set(api.STABILITY_METRICS)
    assert all(any(f"wheat_api_stability_{m}" in e for m in known) for e in exprs)
    docs = yaml.safe_load_all((ROOT / "monitoring/20-grafana-config.yaml").read_text())
    config = next(d for d in docs if d and d["metadata"]["name"] == "grafana-dashboards")
    assert json.loads(config["data"]["wheat-stability.json"]) == dashboard


def test_dvc_stage_declared():
    stage = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]["stability"]
    assert "wheat_mlops.stability.collect" in stage["cmd"] and "stability.report" in stage["cmd"]
    assert stage["outs"] == ["data/stability"] and "data/perturbed" in stage["deps"]
