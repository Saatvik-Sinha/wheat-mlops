# syntax=docker/dockerfile:1
#
# Project image (CPU only - training happens on Kaggle/Colab, never here).
# Build from the repo root:
#   docker build -f docker/Dockerfile -t wheat-mlops:0.1.0 --build-arg PYTHON_VERSION=3.12 .
#
# PYTHON_VERSION must satisfy `requires-python` in pyproject.toml.
# TORCH_VARIANT=cpu  -> CPU-only torch wheels (small image, default)
# TORCH_VARIANT=default -> exactly what uv.lock says (large: pulls CUDA libraries)
ARG PYTHON_VERSION=3.14

# ---------- builder: resolve and install dependencies into /opt/venv ----------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG TORCH_VARIANT=cpu
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock README.md ./

# 1) Third-party dependencies only. This layer is cached until pyproject/uv.lock change.
RUN uv venv --python ${PYTHON_VERSION} /opt/venv \
 && uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt \
 && if [ "$TORCH_VARIANT" = "cpu" ]; then \
        grep -vE '^(nvidia-|triton)' requirements.txt > requirements.cpu.txt \
        && uv pip install --python /opt/venv/bin/python -r requirements.cpu.txt \
             --extra-index-url https://download.pytorch.org/whl/cpu \
             --index-strategy unsafe-best-match ; \
    else \
        uv pip install --python /opt/venv/bin/python -r requirements.txt ; \
    fi
COPY requirements-serve.txt ./
RUN uv pip install --python /opt/venv/bin/python -r requirements-serve.txt

# 2) The project itself (changes often, so it comes last).
COPY src/python ./src/python
RUN uv pip install --python /opt/venv/bin/python --no-deps .
RUN uv pip uninstall --python /opt/venv/bin/python opencv-python && uv pip install --python /opt/venv/bin/python opencv-python-headless

# ---------- runtime: slim image with only what is needed to run ----------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
# OpenCV (pulled in by ultralytics) needs these two system libraries.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY params.yaml ./
COPY docker/smoke_test.py ./
USER app

# No model weights or data are baked in. How the model reaches the container is decided
# in Part 8 (it differs between pipeline A and B).
CMD ["python", "smoke_test.py"]

# ---------- serve: runtime + API + the trained model ----------
FROM runtime AS serve
COPY data/working/trained/best.pt /app/models/best.pt
ENV MODEL_PATH=/app/models/best.pt
EXPOSE 8000
CMD ["uvicorn", "wheat_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
