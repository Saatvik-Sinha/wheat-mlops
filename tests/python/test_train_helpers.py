"""Tests for the pure helpers in train.py (no GPU, ultralytics or data needed)."""

from pathlib import Path

import pytest
import yaml

from wheat_mlops.train import (
    SMOKE_ARGS,
    extract_metrics,
    load_train_config,
    resolve_dataset_yaml,
    validate_config,
    with_smoke_overrides,
)

ROOT = Path(__file__).resolve().parents[2]


def test_repo_train_params_are_valid():
    validate_config(load_train_config(ROOT / "train_params.yaml"))


def test_train_imgsz_matches_serving_imgsz():
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    serving = params.get("model", {}).get("imgsz")
    if serving is None:
        pytest.skip("params.yaml has no model.imgsz")
    assert load_train_config(ROOT / "train_params.yaml")["args"]["imgsz"] == serving


def test_smoke_overrides_do_not_mutate_the_original():
    cfg = {"model": "m.pt", "args": {"epochs": 50, "imgsz": 640}}
    smoke = with_smoke_overrides(cfg)
    assert cfg["args"] == {"epochs": 50, "imgsz": 640}
    assert smoke["args"]["epochs"] == SMOKE_ARGS["epochs"] == 2


@pytest.mark.parametrize(
    "cfg",
    [
        {"model": "", "args": {}},
        {"model": "m.pt", "args": {"imgsz": 650}},
        {"model": "m.pt", "args": {"epochs": 0}},
    ],
)
def test_validate_rejects_bad_config(cfg):
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_resolve_dataset_yaml_adds_absolute_path(tmp_path):
    src = tmp_path / "processed" / "dataset.yaml"
    src.parent.mkdir()
    src.write_text("train: images/train\nval: images/val\nnc: 1\n", encoding="utf-8")
    out = resolve_dataset_yaml(src, tmp_path)
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert Path(cfg["path"]).is_absolute()
    assert Path(cfg["path"]) == src.parent.resolve()
    assert cfg["train"] == "images/train"


def test_extract_metrics_maps_ultralytics_keys():
    results = {
        "metrics/precision(B)": 0.9,
        "metrics/recall(B)": 0.8,
        "metrics/mAP50(B)": 0.95,
        "metrics/mAP50-95(B)": 0.6,
        "fitness": 0.7,
    }
    assert extract_metrics(results) == {
        "precision": 0.9,
        "recall": 0.8,
        "map50": 0.95,
        "map50_95": 0.6,
    }
    assert extract_metrics({})["map50"] is None
