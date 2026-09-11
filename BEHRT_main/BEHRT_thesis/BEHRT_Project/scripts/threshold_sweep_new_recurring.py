#!/usr/bin/env python3
"""
Per-subset threshold sweep for NEW vs RECURRING diagnosis predictions.

Loads a trained checkpoint from a `clean_run_*` directory (produced by
train_nextvisit_clean.py), recomputes test-set probabilities, and sweeps the
decision threshold independently for the "recurring" subset (codes already in
the patient's history) and the "new" subset (first occurrence for that
patient). This answers: does a single global threshold under-serve new
diagnoses even though their AUC/ranking quality is good?

Usage:
    RUN_DIR=data/models/clean_runs/clean_run_20260910_124255 \
        python3.12 scripts/threshold_sweep_new_recurring.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj
from model.utils import age_vocab
from dataLoader.NextXVisit import NextVisit
from model.NextXVisit import BertForMultiLabelPrediction

from scripts.train_nextvisit_clean import (
    BertConfig,
    build_label_subset,
    collect_eval_arrays,
    compute_new_recurring_metrics,
    compute_recurring_mask,
    format_label_vocab,
)


def find_latest_run_dir(project_root):
    runs_root = project_root / "data" / "models" / "clean_runs"
    candidates = [d for d in runs_root.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No clean_run_* directories with metrics.json found under {runs_root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def sweep_thresholds(y_true, y_prob, recurring_mask, thresholds):
    sweep = {"recurring": [], "new": []}
    for threshold in thresholds:
        metrics = compute_new_recurring_metrics(y_true, y_prob, threshold, recurring_mask)
        for subset in ("recurring", "new"):
            sweep[subset].append(
                {
                    "threshold": float(threshold),
                    "precision": metrics[subset]["precision"],
                    "recall": metrics[subset]["recall"],
                    "f1": metrics[subset]["f1"],
                }
            )
    return sweep


def best_threshold_by_f1(sweep_entries):
    best = max(sweep_entries, key=lambda row: row["f1"])
    return best


def main():
    project_root = PROJECT_ROOT
    data_dir = project_root / "data" / "processed"

    run_dir_env = os.getenv("RUN_DIR", "")
    run_dir = Path(run_dir_env) if run_dir_env else find_latest_run_dir(project_root)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        run_metrics = json.load(f)

    print(f"Loaded run: {run_dir}")

    model_config = run_metrics["model_config"]
    top_k_labels = int(run_metrics["run_controls"].get("top_k_labels", 0) or 0)
    min_label_freq = float(run_metrics["run_controls"].get("min_label_freq", 0.0) or 0.0)
    metric_exclude_labels = tuple(run_metrics["run_controls"].get("metric_exclude_labels", ["UNK"]))
    global_threshold = float(run_metrics["threshold_tuning"]["selected_threshold"])

    train_path = data_dir / run_metrics["data"]["train"]
    test_path = data_dir / run_metrics["data"]["test"]
    vocab_path = data_dir / run_metrics["data"]["vocab"].replace(".pkl", "")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    bert_vocab = load_obj(str(vocab_path))
    age_vocab_dict, _ = age_vocab(max_age=110, symbol=None)
    base_label_vocab = format_label_vocab(bert_vocab["token2idx"])

    label_vocab, train_df, _ = build_label_subset(
        train_df, base_label_vocab, top_k_labels=top_k_labels, min_label_freq=min_label_freq
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len_seq = model_config["max_position_embedding"]

    test_set = NextVisit(
        token2idx=bert_vocab["token2idx"],
        label2idx=label_vocab,
        age2idx=age_vocab_dict,
        dataframe=test_df,
        max_len=max_len_seq,
    )
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    mlb = MultiLabelBinarizer(classes=list(label_vocab.values()))
    mlb.fit([[x] for x in list(label_vocab.values())])

    conf = BertConfig(model_config)
    model = BertForMultiLabelPrediction(conf, num_labels=len(label_vocab), feature_dict={"word": True, "seg": True, "age": True, "position": True})
    model.load_state_dict(torch.load(run_dir / "behrt_nextvisit_ccsr_clean_best.pt", map_location=device))
    model = model.to(device)

    print("Recomputing test-set predictions...")
    y_true, y_prob = collect_eval_arrays(model, test_loader, mlb, device)
    recurring_mask = compute_recurring_mask(test_df, label_vocab)

    if metric_exclude_labels:
        excluded_indices = {int(label_vocab[label]) for label in metric_exclude_labels if label in label_vocab}
        keep_indices = [idx for idx in range(y_true.shape[1]) if idx not in excluded_indices]
        y_true = y_true[:, keep_indices]
        y_prob = y_prob[:, keep_indices]
        recurring_mask = recurring_mask[:, keep_indices]

    thresholds = np.arange(0.05, 0.65, 0.05)
    sweep = sweep_thresholds(y_true, y_prob, recurring_mask, thresholds)

    best_recurring = best_threshold_by_f1(sweep["recurring"])
    best_new = best_threshold_by_f1(sweep["new"])
    global_at_tuned = compute_new_recurring_metrics(y_true, y_prob, global_threshold, recurring_mask)

    print("=" * 88)
    print(f"Global tuned threshold: {global_threshold:.2f}")
    print(
        f"  recurring @ global: P={global_at_tuned['recurring']['precision']:.4f} "
        f"R={global_at_tuned['recurring']['recall']:.4f} F1={global_at_tuned['recurring']['f1']:.4f}"
    )
    print(
        f"  new       @ global: P={global_at_tuned['new']['precision']:.4f} "
        f"R={global_at_tuned['new']['recall']:.4f} F1={global_at_tuned['new']['f1']:.4f}"
    )
    print("-" * 88)
    print(f"Best per-subset threshold (max F1):")
    print(
        f"  recurring @ {best_recurring['threshold']:.2f}: P={best_recurring['precision']:.4f} "
        f"R={best_recurring['recall']:.4f} F1={best_recurring['f1']:.4f}"
    )
    print(
        f"  new       @ {best_new['threshold']:.2f}: P={best_new['precision']:.4f} "
        f"R={best_new['recall']:.4f} F1={best_new['f1']:.4f}"
    )
    print("-" * 88)
    print("Full sweep (new subset):")
    for row in sweep["new"]:
        print(f"  t={row['threshold']:.2f}  P={row['precision']:.4f}  R={row['recall']:.4f}  F1={row['f1']:.4f}")
    print("=" * 88)

    result = {
        "run_dir": str(run_dir.relative_to(project_root)),
        "global_tuned_threshold": global_threshold,
        "global_threshold_metrics": global_at_tuned,
        "sweep": sweep,
        "best_per_subset_threshold": {
            "recurring": best_recurring,
            "new": best_new,
        },
    }

    output_path = run_dir / "new_recurring_threshold_sweep.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved sweep results to: {output_path}")


if __name__ == "__main__":
    main()
