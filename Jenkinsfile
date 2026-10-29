// Pipeline B CI (Part 6B): lint + test in parallel, then validate the DVC pipeline.
// Runs on the local Jenkins from jenkins/ (see jenkins/README.md), which has uv preinstalled.
pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    // Jenkins runs on localhost, so GitHub webhooks cannot reach it; poll the repo instead.
    triggers {
        pollSCM('H/5 * * * *')
    }

    environment {
        PYTHON_VERSION = '3.14'
        RUFF_VERSION = '0.17.2' // keep in sync with uv.lock
        UV_LINK_MODE = 'copy'
    }

    stages {
        stage('Lint & Test') {
            parallel {
                stage('Lint (ruff)') {
                    steps {
                        sh 'uvx ruff@${RUFF_VERSION} check .'
                        sh 'uvx ruff@${RUFF_VERSION} format --check .'
                    }
                }
                stage('Test (pytest)') {
                    steps {
                        // uv.lock pins CUDA torch (~3 GB of nvidia-* wheels). The tests never import
                        // torch, so install CPU-only wheels instead -- same trick as docker/Dockerfile.
                        sh '''
                            uv venv --clear --python "$PYTHON_VERSION"
                            uv export --frozen --all-groups --no-hashes --no-emit-project -o requirements.txt > /dev/null
                            grep -vE '^(nvidia-|triton)' requirements.txt > requirements.cpu.txt
                            uv pip install -r requirements.cpu.txt \
                                --extra-index-url https://download.pytorch.org/whl/cpu \
                                --index-strategy unsafe-best-match
                            uv pip install -r requirements-serve.txt httpx
                            uv pip install --no-deps -e .
                        '''
                        // --no-sync stops uv from re-syncing the venv to uv.lock (which would pull CUDA).
                        sh 'uv run --no-sync pytest -v --junitxml=junit.xml'
                    }
                    post {
                        always {
                            junit allowEmptyResults: true, testResults: 'junit.xml'
                        }
                    }
                }
            }
        }

        // Checks that dvc.yaml parses and its stages form a valid DAG (no data download needed).
        stage('DVC pipeline check') {
            steps {
                sh 'uv run --no-sync dvc dag'
            }
        }
    }
}
