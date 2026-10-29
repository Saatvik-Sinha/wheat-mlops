"""FastAPI inference service for wheat head detection.

Standalone on purpose: it imports nothing from the rest of the package.

Run:  uvicorn wheat_mlops.api:app --host 0.0.0.0 --port 8000
Env:  MODEL_PATH         default /app/models/best.pt
      MODEL_VERSION      label used in /health and the Prometheus metrics
      MLFLOW_TRACKING_URI MLflow tracking server URI (default sqlite:///mlflow.db)
      MLFLOW_EXPERIMENT_NAME experiment used for API prediction traces
      IMGSZ              inference size, default 640 (match training)
      INFERENCE_THREADS  torch CPU threads, default 2
      CORS_ORIGINS       comma-separated origins, default "*"
      MAX_UPLOAD_MB      default 10

Endpoints: POST /predict, GET /health, GET /metrics (Prometheus), GET /docs.
Metric names and labels follow the contract in monitoring/README.md.
"""

from __future__ import annotations

import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    ProcessCollector,
    generate_latest,
)
from pydantic import BaseModel

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
DETECTION_BUCKETS = (0, 1, 5, 10, 20, 30, 50, 75, 100, 150)
CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
UNTRACKED_PATHS = {"/metrics", "/health", "/docs", "/redoc", "/openapi.json"}
MAX_PIXELS = 25_000_000
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "wheat-api")


class ApiError(Exception):
    def __init__(self, status: int, error_type: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.error_type = error_type
        self.detail = detail


class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class PredictResponse(BaseModel):
    count: int
    mean_confidence: float | None
    detections: list[Detection]
    image_width: int
    image_height: int
    inference_ms: float
    model_version: str


class Metrics:
    """All Prometheus metrics, on a private registry (so tests can build many apps)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)  # process_resident_memory_bytes etc.
        self.requests = Counter(
            "wheat_api_requests_total",
            "HTTP requests",
            ["endpoint", "method", "status"],
            registry=self.registry,
        )
        self.errors = Counter(
            "wheat_api_errors_total",
            "Failed HTTP requests",
            ["endpoint", "error_type", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "wheat_api_request_duration_seconds",
            "Request duration",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.detections = Histogram(
            "wheat_api_detections_per_image",
            "Wheat heads detected per image",
            ["model_version"],
            buckets=DETECTION_BUCKETS,
            registry=self.registry,
        )
        self.confidence = Histogram(
            "wheat_api_detection_confidence",
            "Confidence of each detected box",
            ["model_version"],
            buckets=CONFIDENCE_BUCKETS,
            registry=self.registry,
        )


class YoloDetector:
    """Thin wrapper around an Ultralytics model (the only place that knows its API)."""

    def __init__(self, model, imgsz: int = 640) -> None:
        self.model = model
        self.imgsz = imgsz
        self._lock = threading.Lock()  # one prediction at a time; the model is not thread-safe

    @classmethod
    def from_env(cls) -> YoloDetector:
        import torch  # heavy imports only when really loading a model
        from ultralytics import YOLO

        path = Path(os.getenv("MODEL_PATH", "/app/models/best.pt"))
        print("Model Path", path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        torch.set_num_threads(int(os.getenv("INFERENCE_THREADS", "2")))
        imgsz = int(os.getenv("IMGSZ", "640"))
        detector = cls(YOLO(str(path)), imgsz=imgsz)
        detector.detect(Image.new("RGB", (imgsz, imgsz)), 0.25, 0.7)  # warm-up
        return detector

    def detect(self, image: Image.Image, conf: float, iou: float) -> list[tuple]:
        """Return a list of (x1, y1, x2, y2, score) in the coordinates of ``image``."""
        with self._lock:
            print("Predict Called")
            result = self.model.predict(
                image, imgsz=self.imgsz, device="cpu", verbose=False, conf=conf, iou=iou
            )[0]
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [
            (float(x1), float(y1), float(x2), float(y2), float(s))
            for (x1, y1, x2, y2), s in zip(xyxy, scores, strict=True)
        ]


def _decode_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_PIXELS:
            raise ApiError(400, "image_too_large", "image has too many pixels")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ApiError(400, "invalid_image", "the uploaded file is not a readable image") from exc
    # Apply the EXIF rotation, so box coordinates match what a browser shows.
    return ImageOps.exif_transpose(image).convert("RGB")


def start_ngrok() -> tuple[str, object] | None:
    """Open an ngrok tunnel to this server if NGROK_AUTHTOKEN is set. Never fails the API."""
    token = os.getenv("NGROK_AUTHTOKEN")
    if not token:
        return None
    try:
        from pyngrok import ngrok
        from pyngrok.conf import PyngrokConfig

        # /tmp is writable for the non-root container user; pyngrok downloads the binary there.
        config = PyngrokConfig(auth_token=token, config_version="3")
        domain = os.getenv("NGROK_DOMAIN")
        port = int(os.getenv("PORT", "8080"))
        options = {
            "upstream": {"url": f"http://localhost:{port}"},
            "url": f"https://{domain}",
        }
        url = ngrok.connect(pyngrok_config=config, **options).public_url
        print(f"ngrok tunnel open: {url}")
        return url, config
    except Exception as exc:  # noqa: BLE001 - the API must keep running without the tunnel
        print(f"WARNING: ngrok did not start: {exc}")
        return None


def stop_ngrok(tunnel: tuple[str, object] | None) -> None:
    if tunnel:
        from pyngrok import ngrok

        ngrok.kill(pyngrok_config=tunnel[1])


def create_app(detector: YoloDetector | None = None) -> FastAPI:
    # Use the same MLflow backend as the UI. For an API running in another
    # container, set MLFLOW_TRACKING_URI to a reachable MLflow tracking server.
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    metrics = Metrics()
    model_version = os.getenv("MODEL_VERSION", "unversioned")
    max_bytes = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.detector is None:
            app.state.detector = YoloDetector.from_env()  # fails fast if the model is missing
        tunnel = start_ngrok()
        try:
            yield
        finally:
            stop_ngrok(tunnel)

    app = FastAPI(title="Wheat Detection API", version="1.0", lifespan=lifespan)
    app.state.detector = detector

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        request.state.error_type = exc.error_type
        return JSONResponse(
            {"detail": exc.detail, "error_type": exc.error_type}, status_code=exc.status
        )

    @app.middleware("http")
    async def track(request: Request, call_next):
        if request.url.path in UNTRACKED_PATHS:
            return await call_next(request)
        start = time.perf_counter()
        status, error_type = 500, "internal"  # what we record if an exception escapes
        try:
            response = await call_next(request)
            status = response.status_code
            default = "client_error" if status < 500 else "server_error"
            error_type = getattr(request.state, "error_type", default)
            return response
        finally:
            endpoint = getattr(request.scope.get("route"), "path", "unmatched")
            metrics.requests.labels(endpoint, request.method, str(status)).inc()
            metrics.latency.labels(endpoint).observe(time.perf_counter() - start)
            if status >= 400:
                metrics.errors.labels(endpoint, error_type, str(status)).inc()

    # Added after the tracking middleware, so CORS is the outermost layer.
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        if app.state.detector is None:
            return JSONResponse({"status": "loading", "model_loaded": False}, status_code=503)
        return {"status": "ok", "model_loaded": True, "model_version": model_version}

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        request: Request,
        file: Annotated[UploadFile, File()],
        conf: Annotated[float, Query(ge=0.01, le=0.99)] = 0.25,
        iou: Annotated[float, Query(ge=0.1, le=0.95)] = 0.7,
    ):
        # Do not record the uploaded image or request object in MLflow. The trace
        # contains only safe request metadata, inference timing, and result summary.
        with mlflow.start_span("wheat.predict", span_type="CHAIN") as trace:
            trace.set_inputs({"confidence_threshold": conf, "iou_threshold": iou})
            active = request.app.state.detector
            if active is None:
                raise ApiError(503, "model_unavailable", "the model is not loaded yet")
            data = file.file.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ApiError(
                    413, "file_too_large", f"file is larger than {max_bytes // 2**20} MB"
                )

            with mlflow.start_span("decode_image", span_type="PARSER") as decode_span:
                image = _decode_image(data)
                decode_span.set_inputs({"content_length": len(data)})
                decode_span.set_outputs({"image_width": image.width, "image_height": image.height})

            with mlflow.start_span("yolo_inference", span_type="TOOL") as inference_span:
                inference_span.set_inputs(
                    {
                        "model_version": model_version,
                        "confidence_threshold": conf,
                        "iou_threshold": iou,
                    }
                )
                start = time.perf_counter()
                boxes = active.detect(image, conf, iou)
                inference_ms = (time.perf_counter() - start) * 1000
                inference_span.set_outputs(
                    {"detection_count": len(boxes), "inference_ms": round(inference_ms, 1)}
                )

            scores = [b[4] for b in boxes]
            metrics.detections.labels(model_version).observe(len(boxes))
            for score in scores:
                metrics.confidence.labels(model_version).observe(score)
            result = PredictResponse(
                count=len(boxes),
                mean_confidence=sum(scores) / len(scores) if scores else None,
                detections=[
                    Detection(x1=b[0], y1=b[1], x2=b[2], y2=b[3], confidence=b[4]) for b in boxes
                ],
                image_width=image.width,
                image_height=image.height,
                inference_ms=round(inference_ms, 1),
                model_version=model_version,
            )
            trace.set_outputs(
                {
                    "detection_count": result.count,
                    "inference_ms": result.inference_ms,
                    "model_version": result.model_version,
                }
            )
            return result

    return app


app = create_app()
