"""Part C: tables, plots and MLflow logging of bounding-box stability vs degradation level.

Reads ``<out_dir>/metrics/`` (Part B) and writes to ``<out_dir>/report/``:

    tables/<metric>_by_severity.csv      corruption x severity table for every metric
    tables/stability_tables.md           the same tables as markdown (copy into a report)
    plots/stability_vs_severity.png      line plots, one panel per metric, one line per corruption
    plots/heatmap_matched_ratio.png      corruption x severity heatmaps of the key metrics
    plots/heatmap_mean_iou.png
    plots/heatmap_center_disp_px.png
    plots/iou_distribution.png           per-image stability_iou spread at each severity
    plots/boxes_<image_id>.png           clean vs degraded boxes drawn on one example image

Usage:
    python -m wheat_mlops.stability.report
    python -m wheat_mlops.stability.report --no-mlflow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from wheat_mlops.stability.collect import load_detections  # noqa: E402
from wheat_mlops.stability.metrics import MEAN_METRICS  # noqa: E402

LABELS = {
    "matched_ratio": "Matched detection ratio",
    "mean_iou": "Mean IoU (matched boxes)",
    "stability_iou": "Stability IoU (misses count as 0)",
    "center_disp_px": "Centre displacement (px)",
    "center_disp_rel": "Centre displacement (/ box diagonal)",
    "abs_count_diff": "|Detection count difference|",
    "count_diff": "Detection count difference",
    "width_change_pct": "Width change (%)",
    "height_change_pct": "Height change (%)",
    "log_area_ratio": "Mean log area ratio",
    "conf_diff": "Confidence change (mean)",
    "new_ratio": "New-box ratio",
    "matched_conf_abs_diff": "|Confidence change| (matched)",
}
# Heatmaps: (metric, colormap). Higher matched_ratio / IoU is better, so reversed for the rest.
HEATMAPS = [("matched_ratio", "viridis"), ("mean_iou", "viridis"), ("center_disp_px", "viridis_r")]


def severity_tables(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One corruption x severity pivot table per metric."""
    return {
        metric: summary.pivot(index="corruption", columns="severity", values=metric)
        for metric in MEAN_METRICS
        if metric in summary
    }


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Bounding-box stability by corruption and severity", ""]
    for metric, table in tables.items():
        table.round(4).to_csv(out_dir / f"{metric}_by_severity.csv")
        shown = table.round(3)
        shown.columns = [f"s{c}" for c in shown.columns]
        header = "| corruption | " + " | ".join(shown.columns) + " |"
        rule = "|---|" + "---|" * len(shown.columns)
        lines += [f"## {LABELS.get(metric, metric)} (`{metric}`)", "", header, rule]
        for name, row in shown.iterrows():
            lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
        lines.append("")
    (out_dir / "stability_tables.md").write_text("\n".join(lines), encoding="utf-8")


def plot_vs_severity(summary: pd.DataFrame, metrics: list[str], path: Path) -> None:
    cols = 3
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics, strict=False):
        for corruption, g in summary.groupby("corruption"):
            g = g.sort_values("severity")
            ax.errorbar(
                g["severity"],
                g[metric],
                yerr=g[f"{metric}_std"] / np.sqrt(g["n_images"]),
                marker="o",
                capsize=2,
                label=corruption,
            )
        ax.set_xlabel("severity level")
        ax.set_ylabel(LABELS.get(metric, metric))
        ax.set_xticks(sorted(summary["severity"].unique()))
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Bounding-box stability vs degradation level (error bars: standard error)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, title: str, cmap: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    image = ax.imshow(table.to_numpy(dtype=float), cmap=cmap, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [f"s{c}" for c in table.columns])
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.iat[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_distribution(per_image: pd.DataFrame, path: Path, metric: str = "stability_iou") -> None:
    corruptions = sorted(per_image["corruption"].unique())
    severities = sorted(per_image["severity"].unique())
    fig, axes = plt.subplots(
        1, len(corruptions), figsize=(3.2 * len(corruptions), 3.8), sharey=True
    )
    for ax, corruption in zip(np.atleast_1d(axes), corruptions, strict=True):
        g = per_image[per_image["corruption"] == corruption]
        data = [g.loc[g["severity"] == s, metric].dropna().to_numpy() for s in severities]
        ax.boxplot(data, tick_labels=[f"s{s}" for s in severities], showfliers=False)
        ax.set_title(corruption, fontsize=10)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel(LABELS.get(metric, metric))
    fig.suptitle("Per-image spread of box stability (outliers hidden)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pick_overlay_image(per_image: pd.DataFrame) -> str:
    """Image with a clearly visible effect: the one with median stability at the worst level."""
    worst = per_image[per_image["severity"] == per_image["severity"].max()]
    by_image = worst.groupby("image_id")["stability_iou"].mean().dropna().sort_values()
    if by_image.empty:
        return str(per_image["image_id"].iloc[0])
    return str(by_image.index[len(by_image) // 2])


def plot_overlay(
    image_id: str,
    boxes: pd.DataFrame,
    perturbed_dir: Path,
    manifest: pd.DataFrame,
    severities: list[int],
    path: Path,
) -> bool:
    """Draw clean boxes (green) and degraded boxes (red) on the degraded image."""
    from PIL import Image

    rows = manifest[manifest["image_id"] == image_id]
    corruptions = sorted(c for c in rows["corruption"].unique() if c != "clean")
    levels = [0, *severities]
    if rows.empty or not corruptions:
        return False

    def image_path(corruption: str, severity: int) -> Path | None:
        key = "clean" if severity == 0 else corruption
        sel = rows[
            (rows["corruption"] == key) & (rows["severity"] == (0 if key == "clean" else severity))
        ]
        return perturbed_dir / sel["generated_image"].iloc[0] if len(sel) else None

    fig, axes = plt.subplots(
        len(corruptions), len(levels), figsize=(3.2 * len(levels), 3.2 * len(corruptions)),
        squeeze=False,
    )  # fmt: skip
    drawn = False
    clean_boxes = boxes[(boxes["image_id"] == image_id) & (boxes["condition"] == "clean")]
    for r, corruption in enumerate(corruptions):
        for c, severity in enumerate(levels):
            ax = axes[r][c]
            ax.axis("off")
            img = image_path(corruption, severity)
            if img is None or not img.is_file():
                continue
            with Image.open(img) as opened:
                ax.imshow(opened.convert("RGB"))
            drawn = True
            condition = "clean" if severity == 0 else f"{corruption}_s{severity}"
            for frame, colour, width in ((clean_boxes, "lime", 1.2), (None, "red", 1.0)):
                if frame is None:
                    frame = boxes[
                        (boxes["image_id"] == image_id) & (boxes["condition"] == condition)
                    ]
                    if severity == 0:
                        continue
                for b in frame.itertuples():
                    ax.add_patch(
                        plt.Rectangle(
                            (b.x1, b.y1), b.x2 - b.x1, b.y2 - b.y1,
                            fill=False, edgecolor=colour, linewidth=width,
                        )
                    )  # fmt: skip
            ax.set_title(
                f"{corruption}\n{'clean' if severity == 0 else f's{severity}'}", fontsize=8
            )
    if not drawn:
        plt.close(fig)
        return False
    fig.suptitle(f"Image {image_id}: clean boxes (green) vs degraded-image boxes (red)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def log_to_mlflow(
    summary: pd.DataFrame, report_dir: Path, metrics_dir: Path, params: dict, cfg: dict
) -> None:
    import mlflow

    ml = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8")).get("mlflow", {})
    if ml.get("tracking_uri"):
        mlflow.set_tracking_uri(ml["tracking_uri"])
    mlflow.set_experiment(cfg.get("mlflow_experiment", "wheat-stability"))
    with mlflow.start_run(run_name="stability-report"):
        mlflow.log_params(params)
        # One metric series per corruption with severity as the step -> line charts in the UI.
        for row in summary.itertuples(index=False):
            for metric in MEAN_METRICS:
                value = getattr(row, metric)
                if pd.notna(value):
                    mlflow.log_metric(f"{metric}.{row.corruption}", float(value), step=row.severity)
        # Headline numbers at the worst severity, handy for comparing runs.
        worst = summary[summary["severity"] == summary["severity"].max()]
        for metric in ("matched_ratio", "mean_iou", "center_disp_px", "abs_count_diff"):
            mlflow.log_metric(f"worst_{metric}_mean", float(worst[metric].mean()))
        mlflow.log_artifacts(str(report_dir), artifact_path="stability_report")
        mlflow.log_artifacts(str(metrics_dir), artifact_path="stability_metrics")


def build_report(
    out_dir: str | Path,
    perturbed_dir: str | Path,
    cfg: dict,
    *,
    mlflow_logging: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    metrics_dir = out_dir / "metrics"
    per_image_path = metrics_dir / "per_image_stability.csv"
    summary_path = metrics_dir / "condition_summary.csv"
    if not (per_image_path.is_file() and summary_path.is_file()):
        raise FileNotFoundError(f"no metrics in {metrics_dir}; run stability.metrics first")
    per_image = pd.read_csv(per_image_path, dtype={"image_id": str})
    summary = pd.read_csv(summary_path)

    report_dir = out_dir / cfg.get("out_subdir", "report")
    plots = report_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    tables = severity_tables(summary)
    write_tables(tables, report_dir / "tables")

    metrics = [m for m in cfg["plot_metrics"] if m in summary]
    plot_vs_severity(summary, metrics, plots / "stability_vs_severity.png")
    for metric, cmap in HEATMAPS:
        plot_heatmap(tables[metric], LABELS[metric], cmap, plots / f"heatmap_{metric}.png")
    plot_distribution(per_image, plots / "iou_distribution.png")

    manifest_path = Path(perturbed_dir) / "manifest.csv"
    overlay_made = False
    if manifest_path.is_file():
        boxes, _ = load_detections(out_dir)
        manifest = pd.read_csv(manifest_path, dtype={"image_id": str})
        image_id = str(cfg.get("overlay_image") or pick_overlay_image(per_image))
        overlay_made = plot_overlay(
            image_id,
            boxes,
            Path(perturbed_dir),
            manifest,
            [int(s) for s in cfg.get("overlay_severities", [1, 3, 5])],
            plots / f"boxes_{image_id}.png",
        )
    if not overlay_made:
        print("Note: skipped box overlay (perturbed images or manifest not available).")

    if mlflow_logging:
        params = {
            "stability.n_images": int(per_image["image_id"].nunique()),
            "stability.n_conditions": int(summary["condition"].nunique()),
        }
        info = metrics_dir / "metrics_info.json"
        if info.is_file():
            import json

            params["stability.match_iou"] = json.loads(info.read_text())["match_iou"]
        try:
            log_to_mlflow(summary, report_dir, metrics_dir, params, cfg)
            print(f"Logged to MLflow experiment '{cfg.get('mlflow_experiment')}'.")
        except Exception as exc:  # reporting must not fail because tracking is unavailable
            print(f"Note: MLflow logging skipped: {exc}")
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="stability.yaml")
    parser.add_argument("--out-dir")
    parser.add_argument("--perturbed-dir")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = args.out_dir or config["collect"]["out_dir"]
    perturbed_dir = args.perturbed_dir or config["collect"]["perturbed_dir"]
    report_dir = build_report(
        out_dir, perturbed_dir, config["report"], mlflow_logging=not args.no_mlflow
    )
    print(f"Report written to {report_dir}")


if __name__ == "__main__":
    main()
