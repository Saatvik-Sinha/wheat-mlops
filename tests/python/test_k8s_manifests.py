"""Static checks for the Kubernetes manifests. Needs neither Docker nor a cluster."""

from pathlib import Path

import pytest
import yaml

K8S_DIR = Path(__file__).resolve().parents[2] / "k8s"
NAMESPACE = "wheat-mlops"


@pytest.fixture(scope="module")
def docs():
    found = []
    for path in sorted(K8S_DIR.rglob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if doc:
                found.append(doc)
    assert found, f"no manifests found in {K8S_DIR}"
    return found


def _of_kind(docs, kind):
    return [d for d in docs if d["kind"] == kind]


def _containers(doc):
    return doc["spec"]["template"]["spec"]["containers"]


def test_every_manifest_has_basic_fields(docs):
    for doc in docs:
        assert doc.get("apiVersion") and doc.get("kind")
        assert doc["metadata"]["name"]


def test_namespace_exists_and_is_used_everywhere(docs):
    names = [d["metadata"]["name"] for d in _of_kind(docs, "Namespace")]
    assert names == [NAMESPACE]
    for doc in docs:
        if doc["kind"] != "Namespace":
            assert doc["metadata"]["namespace"] == NAMESPACE, doc["metadata"]["name"]


def test_one_pinned_image_everywhere(docs):
    images = {
        c["image"] for d in docs if d["kind"] in ("Deployment", "Job") for c in _containers(d)
    }
    assert len(images) == 1, f"manifests disagree on the image: {images}"
    (image,) = images
    last_part = image.split("/")[-1]
    assert ":" in last_part and not last_part.endswith(":latest")


def test_workloads_are_hardened_and_bounded(docs):
    workloads = _of_kind(docs, "Deployment") + _of_kind(docs, "Job")
    assert workloads
    for doc in workloads:
        for c in _containers(doc):
            for kind in ("requests", "limits"):
                assert {"cpu", "memory"} <= set(c["resources"][kind]), doc["metadata"]["name"]
            assert c["securityContext"]["runAsNonRoot"] is True
            assert c["securityContext"]["allowPrivilegeEscalation"] is False


def test_deployment_has_probes(docs):
    (deployment,) = _of_kind(docs, "Deployment")
    for c in _containers(deployment):
        assert "readinessProbe" in c and "livenessProbe" in c


def test_service_selects_deployment_pods(docs):
    (deployment,) = _of_kind(docs, "Deployment")
    (service,) = _of_kind(docs, "Service")
    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert deployment["spec"]["selector"]["matchLabels"].items() <= pod_labels.items()
    assert service["spec"]["selector"].items() <= pod_labels.items()
    port_names = {p["name"] for c in _containers(deployment) for p in c["ports"]}
    for port in service["spec"]["ports"]:
        assert port["targetPort"] in port_names


def test_smoke_job_runs_once(docs):
    (job,) = _of_kind(docs, "Job")
    assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"
    assert job["spec"]["backoffLimit"] == 0
