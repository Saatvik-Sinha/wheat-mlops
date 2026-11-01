# wheat-mlops

An end-to-end MLOps project for **wheat head detection** on the Kaggle
[Global Wheat Detection](https://www.kaggle.com/competitions/global-wheat-detection) dataset,
using an Ultralytics YOLOv8 model. The aim is a reproducible, automated pipeline around the
model: data versioning, experiment tracking, containerised serving on Kubernetes, monitoring,
CI/CD and a web frontend.

The same project is built **two ways**, one per branch, so the MLOps tools can be compared:

| | [Pipeline A](https://github.com/Saatvik-Sinha/wheat-mlops/tree/pipeline-a) | [Pipeline B](https://github.com/Saatvik-Sinha/wheat-mlops/tree/pipeline-b) |
|---|---|---|
| Branch | `pipeline-a` | `pipeline-b` |
| Data versioning | DVC | DVC |
| Pipeline and experiments | **MLflow** (runs, metrics, artifacts, model registry, prediction traces) | **DVC** (`dvc repro`, `dvc exp run`, `dvc exp show`, `dvc metrics`) |
| CI/CD | **GitHub Actions** (lint, test, Docker build, deploy frontend to GitHub Pages) | **Jenkins** (lint and test in parallel, DVC pipeline check) |
| Training | YOLOv8s tracked in MLflow | YOLOv8s results as files |
| Serving and monitoring | Docker, Kubernetes (minikube), Prometheus, Grafana | same |
| Frontend | https://saatvik-sinha.github.io/wheat-mlops/, deployed to GitHub Pages by GitHub Actions | no deployment of its own (Jenkins does not deploy it); the same page works with Pipeline B's API |

## Installation and full workflow

Each pipeline has its own detailed README with prerequisites, installation, every step from data
download to deployment and monitoring, CI/CD, results and troubleshooting:

- **Pipeline A (DVC + MLflow + GitHub Actions):** [README on `pipeline-a`](https://github.com/Saatvik-Sinha/wheat-mlops/blob/pipeline-a/README.md)
- **Pipeline B (DVC + Jenkins):** [README on `pipeline-b`](https://github.com/Saatvik-Sinha/wheat-mlops/blob/pipeline-b/README.md)

```bash
git clone https://github.com/Saatvik-Sinha/wheat-mlops.git
cd wheat-mlops
git checkout pipeline-a     # or: git checkout pipeline-b
```

## This branch (`main`)

`main` holds the shared starting point (Part 1) that both pipelines were branched from: data
download, YOLO-format data preparation, base model creation and tests. It is kept unchanged;
all further work is on `pipeline-a` and `pipeline-b`.


Data download, YOLO-format conversion, model creation and base checkpoint. No training yet.

```
params.yaml                         all paths and parameters (single source of truth)
dvc.yaml                            data pipeline stages
src/python/wheat_mlops/             installable Python package
 - download_data.py                 Kaggle download -> data/raw
 - prepare_data.py                  data/raw -> data/processed (YOLO format, stratified split)
 - build_model.py                   pretrained YOLO -> checkpoints/base/yolov8n_base.pt
src/scripts/                        development test and lint scripts
src/old_scripts/                    setup scripts, not for running, for record.
tests/python/                       tests that need neither real data nor a checkpoint
```

## Team

| Member | Tools |
|---|---|
| Anirudh Sharma | DVC, MLflow (also model training) |
| Saatvik Sinha | Docker, Kubernetes (also repository owner, API and frontend) |
| Ritu Raj Sharma | GitHub Actions, Jenkins |
| Lomesh | Prometheus, Grafana |