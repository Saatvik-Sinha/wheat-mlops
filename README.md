# wheat-mlops

MLOps mini-project on the Kaggle *Global Wheat Detection* competition
(small Ultralytics YOLO model; the point is the pipeline, not the leaderboard).

## Pipeline B (DVC Experimentation & Pipelining)
- **Data Ingestion & Preparation**: Fully managed and versioned via DVC stages (`download_data`, `prepare_data`).
- **Model Building & Experiment Tracking**: Native DVC pipeline stage (`build_model`) tracking parameters from `params.yaml`, outputs (`checkpoints/base/yolov8n_base.pt`), and metrics (`checkpoints/base/model_info.json`).
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yml`) runs ruff, pytest, Docker build checks, and frontend deployment.

```
params.yaml                         all paths and parameters (single source of truth)
dvc.yaml                            DVC pipeline stages (download_data, prepare_data, build_model)
dvc.lock                            locked stage hashes and data hashes
src/python/wheat_mlops/             installable Python package
 - download_data.py                 Kaggle download -> data/raw
 - prepare_data.py                  data/raw -> data/processed (YOLO format, stratified split)
 - build_model.py                   pretrained YOLO -> checkpoints/base/yolov8n_base.pt
src/scripts/                        development test and lint scripts
src/old_scripts/                    setup scripts, not for running, for record.
tests/python/                       isolated test suite (no real data/checkpoints needed)
```

## Commands
```powershell
# 1. Reproduce the complete DVC pipeline
uv run dvc repro

# 2. Run and track DVC experiments
uv run dvc exp run

# 3. View metrics and compare experiments
uv run dvc metrics show
uv run dvc exp show

# 4. Push data and model artifacts to remote storage
uv run dvc push

# 5. Tests and Linting
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
