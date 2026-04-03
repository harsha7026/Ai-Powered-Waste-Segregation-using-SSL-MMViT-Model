"""Research pipeline runner for reproducible experiments.

This script orchestrates split creation, baseline training, ViT evaluation,
and comparison reporting using existing project scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINES_DIR = ROOT_DIR / "baselines"
RESEARCH_DIR = ROOT_DIR / "research"
MODEL_ORDER = ["SVM", "ResNet-50", "DenseNet-121", "SSL+MMViT"]


def run_step(step_name: str, command: List[str]) -> None:
    """Run one pipeline step and fail fast on any non-zero exit."""
    printable = " ".join(command)
    print("\n" + "=" * 90)
    print(f"STEP: {step_name}")
    print(f"COMMAND: {printable}")
    print("=" * 90)
    subprocess.run(command, cwd=str(ROOT_DIR), check=True)


def read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def collect_current_metrics() -> Dict[str, Dict[str, Any]]:
    """Read the latest metric JSON files produced by existing scripts."""
    metrics_files = {
        "SVM": BASELINES_DIR / "svm_baseline" / "metrics_svm.json",
        "ResNet-50": BASELINES_DIR / "resnet50_baseline" / "metrics_resnet50.json",
        "DenseNet-121": BASELINES_DIR / "densenet121_baseline" / "metrics_densenet121.json",
        "SSL+MMViT": BASELINES_DIR / "metrics_mvm_vit.json",
    }

    out: Dict[str, Dict[str, Any]] = {}
    for model_name, path in metrics_files.items():
        payload = read_json(path)
        if payload is None:
            continue
        out[model_name] = {
            "accuracy": payload.get("accuracy"),
            "macro_f1": payload.get("macro_f1"),
            "weighted_f1": payload.get("weighted_f1"),
        }
    return out


def write_trial_snapshot(
    run_dir: Path,
    trial_index: int,
    metrics: Dict[str, Dict[str, Any]],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    trial_path = run_dir / f"trial_{trial_index:02d}_metrics.json"
    payload = {
        "trial": trial_index,
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    trial_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return trial_path


def compute_stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mean_v = sum(values) / n
    if n > 1:
        var = sum((x - mean_v) ** 2 for x in values) / (n - 1)
        std_v = math.sqrt(var)
    else:
        std_v = 0.0
    ci95 = 1.96 * std_v / math.sqrt(n) if n > 0 else 0.0
    return {
        "n": n,
        "mean": mean_v,
        "std": std_v,
        "ci95": ci95,
    }


def aggregate_trials(trial_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate trial snapshots into mean/std/CI per model and metric."""
    aggregate: Dict[str, Any] = {
        "trials": len(trial_records),
        "models": {},
    }

    for model in MODEL_ORDER:
        metric_buckets: Dict[str, List[float]] = {
            "accuracy": [],
            "macro_f1": [],
            "weighted_f1": [],
        }
        for trial in trial_records:
            model_metrics = trial.get("metrics", {}).get(model)
            if not model_metrics:
                continue
            for key in metric_buckets:
                value = model_metrics.get(key)
                if isinstance(value, (int, float)):
                    metric_buckets[key].append(float(value))

        if not any(metric_buckets.values()):
            continue

        aggregate["models"][model] = {
            metric_name: compute_stats(values)
            for metric_name, values in metric_buckets.items()
            if values
        }

    return aggregate


def write_paper_tables(aggregate: Dict[str, Any], run_dir: Path) -> None:
    """Write paper-friendly outputs in markdown/csv/json."""
    run_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = run_dir / "aggregate_stats.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    csv_path = run_dir / "paper_results_table.csv"
    md_path = run_dir / "paper_results_table.md"

    csv_rows: List[List[str]] = [
        [
            "model",
            "n",
            "accuracy_mean",
            "accuracy_std",
            "accuracy_ci95",
            "macro_f1_mean",
            "macro_f1_std",
            "macro_f1_ci95",
            "weighted_f1_mean",
            "weighted_f1_std",
            "weighted_f1_ci95",
        ]
    ]

    md_lines = [
        "# Paper-Ready Results Table",
        "",
        f"Trials aggregated: {aggregate.get('trials', 0)}",
        "",
        "| Model | N | Accuracy (mean +/- std) | Macro-F1 (mean +/- std) | Weighted-F1 (mean +/- std) |",
        "|---|---:|---:|---:|---:|",
    ]

    for model in MODEL_ORDER:
        model_stats = aggregate.get("models", {}).get(model)
        if not model_stats:
            continue

        acc = model_stats.get("accuracy", {})
        macro = model_stats.get("macro_f1", {})
        weighted = model_stats.get("weighted_f1", {})
        n = int(acc.get("n", 0))

        csv_rows.append(
            [
                model,
                str(n),
                f"{acc.get('mean', 0.0):.6f}",
                f"{acc.get('std', 0.0):.6f}",
                f"{acc.get('ci95', 0.0):.6f}",
                f"{macro.get('mean', 0.0):.6f}",
                f"{macro.get('std', 0.0):.6f}",
                f"{macro.get('ci95', 0.0):.6f}",
                f"{weighted.get('mean', 0.0):.6f}",
                f"{weighted.get('std', 0.0):.6f}",
                f"{weighted.get('ci95', 0.0):.6f}",
            ]
        )

        md_lines.append(
            f"| {model} | {n} | {acc.get('mean', 0.0):.4f} +/- {acc.get('std', 0.0):.4f} "
            f"(CI95 +/- {acc.get('ci95', 0.0):.4f}) | "
            f"{macro.get('mean', 0.0):.4f} +/- {macro.get('std', 0.0):.4f} "
            f"(CI95 +/- {macro.get('ci95', 0.0):.4f}) | "
            f"{weighted.get('mean', 0.0):.4f} +/- {weighted.get('std', 0.0):.4f} "
            f"(CI95 +/- {weighted.get('ci95', 0.0):.4f}) |"
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(csv_rows)

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Keep latest report copies at research root for convenience.
    (RESEARCH_DIR / "aggregate_stats.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (RESEARCH_DIR / "paper_results_table.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    (RESEARCH_DIR / "paper_results_table.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")


def build_summary() -> str:
    """Create a markdown summary from generated metric files."""
    metrics_files = {
        "SVM": BASELINES_DIR / "svm_baseline" / "metrics_svm.json",
        "ResNet-50": BASELINES_DIR / "resnet50_baseline" / "metrics_resnet50.json",
        "DenseNet-121": BASELINES_DIR / "densenet121_baseline" / "metrics_densenet121.json",
        "SSL+MMViT": BASELINES_DIR / "metrics_mvm_vit.json",
    }

    rows = []
    for model_name, path in metrics_files.items():
        metrics = read_json(path)
        if metrics is None:
            rows.append((model_name, None, None, None, "missing"))
            continue
        rows.append(
            (
                model_name,
                metrics.get("accuracy"),
                metrics.get("macro_f1"),
                metrics.get("weighted_f1"),
                "ok",
            )
        )

    best_model = None
    best_macro_f1 = -1.0
    for model_name, _, macro_f1, _, status in rows:
        if status != "ok" or macro_f1 is None:
            continue
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model = model_name

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Research Pipeline Summary",
        "",
        f"Generated at: {now}",
        "",
        "## Aggregate Metrics",
        "",
        "| Model | Accuracy | Macro-F1 | Weighted-F1 | Status |",
        "|---|---:|---:|---:|---|",
    ]

    for model_name, accuracy, macro_f1, weighted_f1, status in rows:
        lines.append(
            f"| {model_name} | {format_metric(accuracy)} | {format_metric(macro_f1)} | "
            f"{format_metric(weighted_f1)} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Key Finding",
            "",
            f"Best model by Macro-F1: {best_model if best_model else 'N/A'}",
            "",
            "## Notes",
            "",
            "- This summary reads existing metric JSON files generated by training/evaluation scripts.",
            "- For publishable reporting, run multiple trials and report mean +/- std.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible research pipeline.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--resnet-epochs", type=int, default=None)
    parser.add_argument("--densenet-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--svm-mode", choices=["fast", "full"], default=None)
    args = parser.parse_args()

    python_exec = sys.executable

    resnet_epochs = args.resnet_epochs if args.resnet_epochs is not None else (8 if args.mode == "quick" else 30)
    densenet_epochs = args.densenet_epochs if args.densenet_epochs is not None else (8 if args.mode == "quick" else 30)
    svm_mode = args.svm_mode if args.svm_mode is not None else ("fast" if args.mode == "quick" else "full")
    trials = max(1, int(args.trials))
    run_tag = args.run_tag if args.run_tag else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESEARCH_DIR / "runs" / run_tag

    run_step("Create fixed train/val/test split", [python_exec, "baselines/create_splits.py"])

    trial_records: List[Dict[str, Any]] = []
    for trial_index in range(1, trials + 1):
        print("\n" + "#" * 90)
        print(f"TRIAL {trial_index}/{trials}")
        print("#" * 90)

        if not args.skip_train:
            run_step(
                "Train SVM baseline",
                [python_exec, "baselines/svm_baseline/train_svm.py", "--mode", svm_mode],
            )
            run_step(
                "Train ResNet-50 baseline",
                [
                    python_exec,
                    "baselines/resnet50_baseline/train_resnet50.py",
                    "--epochs",
                    str(resnet_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--patience",
                    str(args.patience),
                ],
            )
            run_step(
                "Train DenseNet-121 baseline",
                [
                    python_exec,
                    "baselines/densenet121_baseline/train_densenet121.py",
                    "--epochs",
                    str(densenet_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--patience",
                    str(args.patience),
                ],
            )

        if not args.skip_eval:
            run_step(
                "Evaluate SSL+MMViT on shared test split",
                [python_exec, "evaluate_on_shared_test.py"],
            )
            run_step(
                "Generate baseline comparison report",
                [python_exec, "baselines/compare_baselines.py"],
            )

        trial_metrics = collect_current_metrics()
        trial_records.append({"trial": trial_index, "metrics": trial_metrics})
        snapshot_path = write_trial_snapshot(run_dir=run_dir, trial_index=trial_index, metrics=trial_metrics)
        print(f"Saved trial metrics snapshot: {snapshot_path}")

    aggregate = aggregate_trials(trial_records)
    write_paper_tables(aggregate=aggregate, run_dir=run_dir)

    summary = build_summary()
    summary_path = RESEARCH_DIR / "last_run_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    print("\n" + "=" * 90)
    print("PIPELINE COMPLETE")
    print(f"Summary written to: {summary_path}")
    print(f"Run artifacts directory: {run_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
