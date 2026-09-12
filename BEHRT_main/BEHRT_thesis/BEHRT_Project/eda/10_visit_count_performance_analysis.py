#!/usr/bin/env python3
"""
EDA/Model Step 10: Visit-count / sequence-position stratified performance.

Answers: does prediction quality depend on how much visit history a patient
has logged? Buckets test rows by VISIT COUNT (number of 'SEP' tokens in the
patient's full `code` history, i.e. number of prior visits recorded before
this prediction point) and recomputes precision/recall/F1/AUC/APS/top-k
metrics per bucket.

Caveat: the model's input sequence is truncated to the last
`max_position_embedding - 1` tokens (see dataLoader/NextXVisit.py), so for
patients with very long histories the model does not literally see every
visit. Visit-count buckets are still computed on the FULL history (not the
truncated one) because that's the quantity that determines whether
truncation/information-loss is happening at all - patients in the highest
bucket are exactly the ones whose early visits get silently dropped from
the model's input.

Usage:
    RUN_DIR=data/models/clean_runs/clean_run_20260910_124255 \
        python3.12 eda/10_visit_count_performance_analysis.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
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
    compute_top_k_metrics,
    format_label_vocab,
)

# Bucket boundaries chosen from the test-set visit-count distribution:
# median=4, p75=9, p90=18 (heavy-tailed, max=236 in this run).
VISIT_COUNT_BUCKETS = [
    ("1-2_visits", 1, 3),
    ("3-4_visits", 3, 5),
    ("5-9_visits", 5, 10),
    ("10-19_visits", 10, 20),
    ("20plus_visits", 20, float("inf")),
]


def find_latest_run_dir(project_root):
    runs_root = project_root / "data" / "models" / "clean_runs"
    candidates = [d for d in runs_root.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No clean_run_* directories with metrics.json found under {runs_root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def compute_visit_count(code_seq):
    return int(sum(1 for token in code_seq if token == "SEP"))


def assign_bucket(visit_count):
    for name, lower, upper in VISIT_COUNT_BUCKETS:
        if lower <= visit_count < upper:
            return name
    return "unknown"


def micro_prf1(y_true_bool, y_pred_bool):
    tp = int(np.sum(y_true_bool & y_pred_bool))
    fp = int(np.sum(~y_true_bool & y_pred_bool))
    fn = int(np.sum(y_true_bool & ~y_pred_bool))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def summarize_bucket(y_true_rows, y_prob_rows, threshold):
    y_true_bool = y_true_rows.astype(bool)
    y_pred_bool = (y_prob_rows >= threshold).astype(bool)

    prf1 = micro_prf1(y_true_bool, y_pred_bool)

    # Sample-wise AUC/APS (per-row averaged), skipping rows with no positives
    # or all positives (undefined for roc_auc_score).
    row_has_signal = (y_true_rows.sum(axis=1) > 0) & (y_true_rows.sum(axis=1) < y_true_rows.shape[1])
    if row_has_signal.sum() > 0:
        try:
            sample_auc = float(
                roc_auc_score(y_true_rows[row_has_signal], y_prob_rows[row_has_signal], average="samples")
            )
        except ValueError:
            sample_auc = float("nan")
        sample_aps = float(
            average_precision_score(y_true_rows[row_has_signal], y_prob_rows[row_has_signal], average="samples")
        )
    else:
        sample_auc = float("nan")
        sample_aps = float("nan")

    top_k = compute_top_k_metrics(y_true_rows, y_prob_rows, top_k_values=(5, 10))

    return {
        "num_rows": int(y_true_rows.shape[0]),
        "total_positive_positions": int(y_true_bool.sum()),
        "precision": prf1["precision"],
        "recall": prf1["recall"],
        "f1": prf1["f1"],
        "sample_wise_auc": sample_auc,
        "sample_wise_aps": sample_aps,
        **top_k,
    }


def main():
    project_root = PROJECT_ROOT
    data_dir = project_root / "data" / "processed"

    run_dir_env = os.getenv("RUN_DIR", "")
    run_dir = Path(run_dir_env) if run_dir_env else find_latest_run_dir(project_root)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir

    with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
        run_metrics = json.load(f)

    print(f"Loaded run: {run_dir}")

    model_config = run_metrics["model_config"]
    top_k_labels = int(run_metrics["run_controls"].get("top_k_labels", 0) or 0)
    min_label_freq = float(run_metrics["run_controls"].get("min_label_freq", 0.0) or 0.0)
    metric_exclude_labels = tuple(run_metrics["run_controls"].get("metric_exclude_labels", ["UNK"]))
    threshold = float(run_metrics["threshold_tuning"]["selected_threshold"])

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
    model = BertForMultiLabelPrediction(
        conf, num_labels=len(label_vocab), feature_dict={"word": True, "seg": True, "age": True, "position": True}
    )
    model.load_state_dict(torch.load(run_dir / "behrt_nextvisit_ccsr_clean_best.pt", map_location=device))
    model = model.to(device)

    print("Recomputing test-set predictions...")
    y_true, y_prob = collect_eval_arrays(model, test_loader, mlb, device)

    if metric_exclude_labels:
        excluded_indices = {int(label_vocab[label]) for label in metric_exclude_labels if label in label_vocab}
        keep_indices = [idx for idx in range(y_true.shape[1]) if idx not in excluded_indices]
        y_true = y_true[:, keep_indices]
        y_prob = y_prob[:, keep_indices]

    # Row order in y_true/y_prob matches test_df row order exactly (shuffle=False,
    # test_df has a default 0..N-1 RangeIndex).
    visit_counts = test_df["code"].apply(compute_visit_count).to_numpy()
    buckets = np.array([assign_bucket(vc) for vc in visit_counts])

    print(f"Visit-count distribution: min={visit_counts.min()} median={np.median(visit_counts):.0f} "
          f"p90={np.percentile(visit_counts, 90):.0f} max={visit_counts.max()}")

    bucket_order = [name for name, _, _ in VISIT_COUNT_BUCKETS]
    bucket_summaries = {}
    print("=" * 110)
    print(f"{'Bucket':<16} {'#Rows':>8} {'MeanVisits':>11} {'F1':>8} {'Recall':>8} {'Precision':>10} {'SampleAUC':>10} {'SampleAPS':>10}")
    print("-" * 110)
    for bucket_name in bucket_order:
        mask = buckets == bucket_name
        if mask.sum() == 0:
            continue
        summary = summarize_bucket(y_true[mask], y_prob[mask], threshold)
        summary["mean_visit_count"] = float(visit_counts[mask].mean())
        bucket_summaries[bucket_name] = summary
        print(
            f"{bucket_name:<16} {summary['num_rows']:>8} {summary['mean_visit_count']:>11.1f} "
            f"{summary['f1']:>8.4f} {summary['recall']:>8.4f} {summary['precision']:>10.4f} "
            f"{summary['sample_wise_auc']:>10.4f} {summary['sample_wise_aps']:>10.4f}"
        )
    print("=" * 110)

    # Truncation flag: rows whose full visit history exceeds what the model can see
    # (max_len_seq - 1 tokens after adding CLS), i.e. code length > max_len_seq - 1.
    code_lengths = test_df["code"].apply(len).to_numpy()
    truncated_mask = code_lengths > (max_len_seq - 1)
    print(f"\nRows with truncated input (full history > {max_len_seq - 1} tokens): "
          f"{int(truncated_mask.sum())}/{len(test_df)} ({truncated_mask.mean():.1%})")

    truncated_summary = summarize_bucket(y_true[truncated_mask], y_prob[truncated_mask], threshold) if truncated_mask.sum() > 0 else None
    not_truncated_summary = summarize_bucket(y_true[~truncated_mask], y_prob[~truncated_mask], threshold) if (~truncated_mask).sum() > 0 else None

    if truncated_summary and not_truncated_summary:
        print(f"  Truncated:     F1={truncated_summary['f1']:.4f} Recall={truncated_summary['recall']:.4f} SampleAUC={truncated_summary['sample_wise_auc']:.4f} (n={truncated_summary['num_rows']})")
        print(f"  Not truncated: F1={not_truncated_summary['f1']:.4f} Recall={not_truncated_summary['recall']:.4f} SampleAUC={not_truncated_summary['sample_wise_auc']:.4f} (n={not_truncated_summary['num_rows']})")

    # Monotonicity assessment: does F1/AUC trend with visit count?
    f1_by_bucket = [bucket_summaries[b]["f1"] for b in bucket_order if b in bucket_summaries]
    auc_by_bucket = [bucket_summaries[b]["sample_wise_auc"] for b in bucket_order if b in bucket_summaries]
    print("\nAssessment:")
    print(f"  F1 by bucket ({' -> '.join(bucket_order)}): {[round(v, 4) for v in f1_by_bucket]}")
    print(f"  Sample-wise AUC by bucket: {[round(v, 4) if not np.isnan(v) else None for v in auc_by_bucket]}")

    result = {
        "run_dir": str(run_dir.relative_to(project_root)),
        "threshold": threshold,
        "max_len_seq": max_len_seq,
        "visit_count_bucket_definitions": {
            name: {"min": lower, "max": upper if upper != float("inf") else None} for name, lower, upper in VISIT_COUNT_BUCKETS
        },
        "bucket_summaries": bucket_summaries,
        "truncation_analysis": {
            "truncation_length_threshold": max_len_seq - 1,
            "num_truncated_rows": int(truncated_mask.sum()),
            "num_not_truncated_rows": int((~truncated_mask).sum()),
            "truncated_metrics": truncated_summary,
            "not_truncated_metrics": not_truncated_summary,
        },
    }

    output_dir = project_root / "eda" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "10_visit_count_performance_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved visit-count performance analysis to: {output_path}")


if __name__ == "__main__":
    main()
