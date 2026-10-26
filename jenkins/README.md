# Jenkins (Pipeline B CI)

A local Jenkins controller that runs the root `Jenkinsfile`:

| Stage | What it does |
|---|---|
| Lint (ruff) | `ruff check` and `ruff format --check` |
| Test (pytest) | installs dependencies with CPU-only torch, runs `pytest`, publishes JUnit results |
| DVC pipeline check | `dvc dag`, to check that `dvc.yaml` is a valid pipeline |

Lint and Test run in parallel. Jenkins polls GitHub every 5 minutes, because GitHub webhooks
cannot reach a Jenkins on localhost.

## Start Jenkins

Docker must be running. From the repo root:

```sh
docker compose -f jenkins/docker-compose.yml up -d --build
```

Open http://localhost:8081. Get the unlock password with:

```sh
docker exec wheat-jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

In the setup wizard, choose **Select plugins to install → None** (the needed plugins are already
in the image), then create your admin user.

## Create the pipeline job

1. **New Item** → name `wheat-mlops-pipeline-b` → **Pipeline** → OK.
2. Under **Pipeline**, set **Definition** to *Pipeline script from SCM*.
3. **SCM**: Git, **Repository URL**: `https://github.com/Saatvik-Sinha/wheat-mlops.git`
   (public, so no credentials are needed).
4. **Branch Specifier**: `*/pipeline-b`. **Script Path**: `Jenkinsfile`.
5. Save, then **Build Now**. The first build is slow because it downloads Python packages;
   later builds reuse the uv cache.

## Stop Jenkins

```sh
docker compose -f jenkins/docker-compose.yml down
```

Add `-v` to also delete the Jenkins home volume (jobs, users and build history).
