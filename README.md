# wheat-mlops

MLOps mini-project on the Kaggle *Global Wheat Detection* competition
(small Ultralytics YOLO model; the point is the pipeline, not the leaderboard).

## Pipeline A (MLflow Experimentation & Pipelining)
- **Data Ingestion**: DVC manages `download_data` (`data/raw`).
- **Data Preparation & Model Building**: Orchestrated via MLflow pipeline (`src/python/wheat_mlops/pipeline.py`), logging parameters, metrics, and artifacts into MLflow.
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yml`): ruff lint, pytest, Docker build check, and deployment of `frontend/` to GitHub Pages on every push to `pipeline-a`.

```
params.yaml                         all paths, parameters, and MLflow config
dvc.yaml                            DVC data ingestion stage (download_data)
MLproject                           MLflow project entry points
src/python/wheat_mlops/             installable Python package
 - download_data.py                 Kaggle download -> data/raw (tracked by DVC)
 - prepare_data.py                  data/raw -> data/processed (YOLO format, stratified split)
 - build_model.py                   pretrained YOLO -> checkpoints/base/yolov8n_base.pt
 - pipeline.py                      MLflow pipeline runner (coordinates prepare & build stages)
src/scripts/                        development test and lint scripts
tests/python/                       isolated test suite (no real data/checkpoints needed)
```

## Commands
```sh
# 1. Download raw data (DVC)
uv run dvc repro

# 2. Run full MLflow pipeline (data preparation + model build + metrics/artifacts logging)
uv run python -m wheat_mlops.pipeline

# 3. View experiments in MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# 4. Tests and Linting
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
