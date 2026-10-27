"""API tests with a fake detector (no model, GPU or ultralytics needed).

They are skipped automatically when the serving packages are not installed. Run them with:
  uv run --with fastapi --with httpx --with prometheus-client --with python-multipart pytest
"""

import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

try:
    from fastapi.testclient import TestClient
    from wheat_mlops import api
except (ImportError, RuntimeError) as exc:  # serving extras not installed
    pytest.skip(f"serving dependencies missing: {exc}", allow_module_level=True)


class FakeDetector:
    def __init__(self, boxes=None, error=None):
        self.boxes = [(1.0, 2.0, 30.0, 40.0, 0.9), (5.0, 6.0, 20.0, 25.0, 0.5)]
        if boxes is not None:
            self.boxes = boxes
        self.error = error
        self.calls = []

    def detect(self, image, conf, iou):
        self.calls.append((image.size, conf, iou))
        if self.error:
            raise self.error
        return self.boxes


def jpeg_bytes(size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 120, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def upload(client, data=None, **params):
    data = jpeg_bytes() if data is None else data
    return client.post("/predict", files={"file": ("a.jpg", data, "image/jpeg")}, params=params)


@pytest.fixture()
def detector():
    return FakeDetector()


@pytest.fixture()
def client(detector):
    return TestClient(api.create_app(detector=detector), raise_server_exceptions=False)


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_health_is_503_while_model_not_loaded():
    client = TestClient(api.create_app(detector=None))
    assert client.get("/health").status_code == 503
    assert upload(client).status_code == 503


def test_predict_returns_detections(client):
    body = upload(client).json()
    assert body["count"] == 2
    assert body["mean_confidence"] == pytest.approx(0.7)
    assert (body["image_width"], body["image_height"]) == (64, 48)
    assert body["detections"][0] == {
        "x1": 1.0, "y1": 2.0, "x2": 30.0, "y2": 40.0, "confidence": 0.9,
    }  # fmt: skip


def test_predict_with_no_detections(client, detector):
    detector.boxes = []
    body = upload(client).json()
    assert body["count"] == 0 and body["mean_confidence"] is None


def test_query_parameters_reach_the_detector(client, detector):
    upload(client, conf=0.5, iou=0.6)
    assert detector.calls[-1] == ((64, 48), 0.5, 0.6)


def test_invalid_image_is_400(client):
    response = upload(client, data=b"definitely not an image")
    assert response.status_code == 400
    assert response.json()["error_type"] == "invalid_image"


def test_missing_file_and_bad_conf_are_422(client):
    assert client.post("/predict").status_code == 422
    assert upload(client, conf=5).status_code == 422


def test_too_large_upload_is_413(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    client = TestClient(api.create_app(detector=FakeDetector()))
    response = upload(client, data=b"x" * (2 * 2**20))
    assert response.status_code == 413
    assert response.json()["error_type"] == "file_too_large"


def test_metrics_follow_the_contract(client):
    upload(client)
    upload(client, data=b"junk")
    text = client.get("/metrics").text
    assert 'wheat_api_requests_total{endpoint="/predict",method="POST",status="200"} 1.0' in text
    assert 'wheat_api_requests_total{endpoint="/predict",method="POST",status="400"} 1.0' in text
    assert (
        'wheat_api_errors_total{endpoint="/predict",error_type="invalid_image",status="400"} 1.0'
        in text
    )
    assert 'wheat_api_request_duration_seconds_bucket{endpoint="/predict",le="0.5"}' in text
    assert 'wheat_api_detections_per_image_count{model_version="unversioned"} 1.0' in text
    assert 'wheat_api_detection_confidence_count{model_version="unversioned"} 2.0' in text
    assert "process_resident_memory_bytes" in text


def test_health_and_metrics_requests_are_not_counted(client):
    client.get("/health")
    client.get("/metrics")
    text = client.get("/metrics").text
    assert 'endpoint="/health"' not in text and 'endpoint="/metrics"' not in text


def test_unhandled_error_is_a_500_and_counted():
    client = TestClient(
        api.create_app(detector=FakeDetector(error=RuntimeError("boom"))),
        raise_server_exceptions=False,
    )
    assert upload(client).status_code == 500
    text = client.get("/metrics").text
    assert 'error_type="internal",status="500"' in text


def test_cors_allows_the_browser_frontend(client):
    response = client.get("/health", headers={"Origin": "https://someone.github.io"})
    assert response.headers["access-control-allow-origin"] == "*"
    preflight = client.options(
        "/predict",
        headers={
            "Origin": "https://someone.github.io",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "ngrok-skip-browser-warning",
        },
    )
    assert preflight.status_code == 200


class _Tensor:  # mimics a torch tensor
    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeYolo:
    def __init__(self):
        self.kwargs = None

    def predict(self, image, **kwargs):
        self.kwargs = kwargs
        boxes = SimpleNamespace(xyxy=_Tensor([[1, 2, 3, 4]]), conf=_Tensor([0.9]))
        return [SimpleNamespace(boxes=boxes)]


def test_yolo_detector_parses_ultralytics_results():
    fake = _FakeYolo()
    detector = api.YoloDetector(fake, imgsz=640)
    boxes = detector.detect(Image.new("RGB", (10, 10)), 0.3, 0.6)
    assert boxes == [(1.0, 2.0, 3.0, 4.0, 0.9)]
    assert fake.kwargs["conf"] == 0.3 and fake.kwargs["iou"] == 0.6
    assert fake.kwargs["imgsz"] == 640 and fake.kwargs["device"] == "cpu"


def test_ngrok_stays_off_without_a_token(monkeypatch):
    monkeypatch.delenv("NGROK_AUTHTOKEN", raising=False)
    assert api.start_ngrok() is None


def test_ngrok_failure_does_not_break_the_api(monkeypatch):
    monkeypatch.setenv("NGROK_AUTHTOKEN", "dummy-token")
    monkeypatch.setitem(sys.modules, "pyngrok", None)  # makes `import pyngrok` fail
    assert api.start_ngrok() is None
