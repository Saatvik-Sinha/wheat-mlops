# wheat-mlops

MLOps mini-project on the Kaggle *Global Wheat Detection* competition
(small Ultralytics YOLO model; the point is the pipeline, not the leaderboard).

## Pipeline B (DVC Experimentation & Pipelining)
- **Data Ingestion & Preparation**: Fully managed and versioned via DVC stages (`download_data`, `prepare_data`).
- **Model Building & Experiment Tracking**: Native DVC pipeline stage (`build_model`) tracking parameters from `params.yaml`, outputs (`checkpoints/base/yolov8n_base.pt`), and metrics (`checkpoints/base/model_info.json`).
- **CI/CD**: Jenkins (`Jenkinsfile`, local controller in `jenkins/`): ruff lint and pytest in parallel, then a DVC pipeline check. Setup steps are in `jenkins/README.md`.

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

## Steps to reproduce the demo (Pipeline B)

Pipeline B demonstrates DVC data preparation and experiment tracking, plus Jenkins CI. The DVC
`build_model` stage creates and smoke-checks a COCO-pretrained YOLO checkpoint; it does not train
on wheat data. For the deployed API, use the trained Pipeline-A model as described below.

### 1. Get the code and configure access

1. Clone the repository and check out the `pipeline-b` branch.
2. Obtain the project `.env` file if needed for the API deployment. Do not commit it.
3. Install Python 3.14 and [uv](https://docs.astral.sh/uv/), then install the project:

   ```sh
   uv sync --all-groups
   ```

4. Put your Kaggle API token in the repository-root `.env` as `KAGGLE_API_TOKEN=...` and accept
   the Global Wheat Detection competition rules on Kaggle. `download_data.py` loads `.env` with
   python-dotenv before it launches the Kaggle CLI.
5. Configure the DVC remote from the repository root. The checked-in default remote is a local
   path at `../../dvc-storage`; use the project's shared remote if one has been provided, for
   example `uv run dvc remote modify my_local_remote url <remote-url>`.

### 2. Download and prepare the dataset with DVC

Run the full pipeline:

```sh
uv run dvc repro
```

This runs `download_data` (Kaggle data to `data/raw/`) followed by `prepare_data` (YOLO-format
images, labels, split, dataset YAML, and statistics under `data/processed/`). The processed data
is created by `src/python/wheat_mlops/prepare_data.py`; MLflow is not used. If DVC reports that
an output is already cached, use `uv run dvc pull` first when the configured remote contains the
outputs, or rerun the stage with `uv run dvc repro --force prepare_data` to regenerate processed
data from the local raw dataset.

### 3. Create a base checkpoint and inspect DVC results

```sh
uv run dvc repro build_model
uv run dvc metrics show
uv run dvc exp run
uv run dvc exp show
uv run dvc push
```

`build_model` downloads the configured pretrained weights if needed, runs a blank-image
prediction, saves `checkpoints/base/yolov8n_base.pt`, reloads it, and writes
`checkpoints/base/model_info.json`. It proves a model checkpoint can be created; it is not the
wheat-trained model used by the serving image. DVC experiment commands track parameter changes
and outputs; `dvc push` uploads cached outputs to the configured remote.

### 4. Provide the trained model for the API image

Copy the trained Pipeline-A checkpoint into this checkout at
`data/working/trained/best.pt` (create the parent directories if needed). The Docker `serve` target
copies that exact file into the image. The Pipeline-B DVC `build_model` checkpoint is not a
replacement for this trained model. Training is intentionally left out of this demo flow.

### 5. Build and smoke-check the Docker image

Docker must be installed and running. From the repository root:

```sh
docker build -f docker/Dockerfile --target serve -t wheat-mlops:0.2.0 .
docker run --rm -p 8000:8000 wheat-mlops:0.2.0
```

The first command requires the Pipeline-A checkpoint from step 4. To check the base Docker image
without the API/model, use:

```sh
docker build -f docker/Dockerfile --target runtime -t wheat-mlops:0.2.0 .
docker run --rm wheat-mlops:0.2.0
```

It should print `SMOKE TEST PASSED`.

### 6. Run the API and monitoring stack on minikube

Install Docker, minikube, and kubectl. Start a local cluster, build and load the API image, then
apply the Kubernetes and monitoring manifests:

```sh
minikube start --driver=docker --cpus=6 --memory=4096
docker build -f docker/Dockerfile --target serve -t wheat-mlops:0.2.0 .
minikube image load wheat-mlops:0.2.0
minikube kubectl -- apply -f k8s/00-namespace.yaml
```

If `.env` contains API secrets, create/update the namespace secret before applying the deployment:

```sh
minikube kubectl -- create secret generic wheat-api-secrets -n wheat-mlops --from-env-file=.env --dry-run=client -o yaml | minikube kubectl -- apply -f -
```

Then apply the application and monitoring manifests:

```sh
minikube kubectl -- apply -f k8s/
minikube kubectl -- apply -f monitoring/
minikube kubectl -- rollout status deployment/wheat-api -n wheat-mlops
minikube kubectl -- get pods,svc -n wheat-mlops
```

Keep each port-forward running in a separate terminal:

```sh
minikube kubectl -- port-forward -n wheat-mlops svc/wheat-api 8000:80
minikube kubectl -- port-forward -n wheat-mlops svc/prometheus 9090:9090
minikube kubectl -- port-forward -n wheat-mlops svc/grafana 3000:3000
```

Open `http://localhost:8000/health`, `http://localhost:9090/`, and `http://localhost:3000/`.
Grafana's provisioned dashboard is under the Wheat Detection folder. The smoke-test Kubernetes
job can also be run with:

```sh
minikube kubectl -- delete job wheat-smoke-test -n wheat-mlops --ignore-not-found
minikube kubectl -- apply -f k8s/jobs/smoke-test.yaml
minikube kubectl -- wait -n wheat-mlops --for=condition=complete job/wheat-smoke-test --timeout=180s
minikube kubectl -- logs -n wheat-mlops job/wheat-smoke-test
```

The API deployment serves the trained model. Prometheus and Grafana are configured to run in the
cluster; Prometheus's `wheat-api` scrape target is healthy only if the deployed API exposes the
configured `/metrics` endpoint.

### 7. Run Pipeline-B CI with Jenkins

Docker must be running. From the repository root:

```sh
docker compose -f jenkins/docker-compose.yml up -d --build
docker exec -it $(docker ps | grep wheat-jenkins | cut -d ' ' -f 1) cat /var/jenkins_home/secrets/initialAdminPassword
```

Open [http://localhost:8081/](http://localhost:8081/) and enter the password printed by the command
above to unlock Jenkins. Choose **Install suggested plugins** (the recommended plugins). After
they finish installing, choose **Skip and Continue as Admin**. Set the Jenkins URL to
`http://localhost:8081/`, then finish setup. Sign in with username `admin` and the same password
printed by the command above.

Create a **Pipeline** job named `wheat-mlops-pipeline-b` with **Pipeline script from SCM**. Set
SCM to Git, repository URL to `https://github.com/Saatvik-Sinha/wheat-mlops.git`, branch
specifier to `*/pipeline-b`, and script path to `Jenkinsfile`; then save and select **Build Now**.
Jenkins runs Ruff lint/format checks and pytest in parallel, then checks the DVC DAG with
`dvc dag`. It polls SCM every five minutes. The first build installs dependencies and can take a
while.

Stop Jenkins when finished:

```sh
docker compose -f jenkins/docker-compose.yml down
```

Add `-v` only if you also want to remove Jenkins jobs, users, and build history.
