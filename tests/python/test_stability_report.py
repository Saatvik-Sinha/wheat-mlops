from __future__ import annotations

import mlflow
import pandas as pd
import pytest
import yaml
from PIL import Image
from wheat_mlops.perturbations import generate_dataset, load_config
from wheat_mlops.stability.collect import collect_all
from wheat_mlops.stability.metrics import run
from wheat_mlops.stability.report import build_report, severity_tables

CFG = yaml.safe_load(open("stability.yaml", encoding="utf-8"))["report"]


def _detector(paths):
    results = []
    for p in paths:
        level = int(p.parent.name[-1]) if p.parent.name != "clean" else 0
        shift = 1.5 * min(level, 4)  # boxes drift further as severity grows (IoU stays > 0.5)
        boxes = [(10 + shift, 10, 30 + shift, 30, 0.9), (50, 50, 70, 70, 0.8 - 0.05 * level)]
        results.append((96, 96, boxes[: 2 - (level == 5)]))  # one box lost at severity 5
    return results


@pytest.fixture
def pipeline(tmp_path):
    processed = tmp_path / "processed"
    for sub in ("images/val", "labels/val"):
        (processed / sub).mkdir(parents=True)
    rows = []
    for image_id in ("v1", "v2", "v3"):
        Image.new("RGB", (96, 96), (90, 100, 110)).save(
            processed / "images/val" / f"{image_id}.jpg"
        )
        (processed / "labels/val" / f"{image_id}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        rows.append({"image_id": image_id, "split": "val", "source": "a", "n_boxes": 1})
    pd.DataFrame(rows).to_csv(processed / "split.csv", index=False)
    perturbed = tmp_path / "perturbed"
    generate_dataset(
        processed, perturbed, config=load_config("perturbations.yaml"),
        config_path="perturbations.yaml",
    )  # fmt: skip
    out = tmp_path / "stability"
    collect_all(perturbed, out, _detector)
    run(out, 0.5)
    return out, perturbed


def test_report_creates_tables_and_plots(pipeline):
    out, perturbed = pipeline
    report = build_report(out, perturbed, CFG, mlflow_logging=False)
    plots = report / "plots"
    for name in (
        "stability_vs_severity.png",
        "heatmap_matched_ratio.png",
        "heatmap_mean_iou.png",
        "heatmap_center_disp_px.png",
        "iou_distribution.png",
    ):
        assert (plots / name).stat().st_size > 1000, name
    assert list(plots.glob("boxes_*.png"))
    md = (report / "tables" / "stability_tables.md").read_text(encoding="utf-8")
    assert "matched_ratio" in md and "gaussian_blur" in md
    table = pd.read_csv(report / "tables" / "matched_ratio_by_severity.csv", index_col=0)
    assert table.shape == (5, 5)
    # Severity 5 loses one of two boxes in the fake detector; severities 1-4 keep both.
    assert table["5"].eq(0.5).all() and table["1"].eq(1.0).all()


def test_severity_tables_shape(pipeline):
    out, _ = pipeline
    summary = pd.read_csv(out / "metrics" / "condition_summary.csv")
    tables = severity_tables(summary)
    assert tables["mean_iou"].index.tolist() == sorted(summary["corruption"].unique())
    assert tables["center_disp_px"].loc["gaussian_blur", 1] == pytest.approx(0.75)


def test_report_skips_overlay_without_images_and_logs_mlflow(pipeline, tmp_path, monkeypatch):
    out, _ = pipeline
    work = tmp_path / "work"
    work.mkdir()
    (work / "params.yaml").write_text(
        f"mlflow:\n  tracking_uri: sqlite:///{work / 'mlflow.db'}\n", encoding="utf-8"
    )
    monkeypatch.chdir(work)
    build_report(out, tmp_path / "missing", CFG, mlflow_logging=True)
    mlflow.set_tracking_uri(f"sqlite:///{work / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name(CFG["mlflow_experiment"])
    runs = mlflow.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs["metrics.worst_matched_ratio_mean"].iloc[0] == pytest.approx(0.5)
    history = mlflow.MlflowClient().get_metric_history(
        runs["run_id"].iloc[0], "mean_iou.gaussian_blur"
    )
    assert [m.step for m in history] == [1, 2, 3, 4, 5]
