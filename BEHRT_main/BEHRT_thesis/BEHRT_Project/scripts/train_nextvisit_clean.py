#!/usr/bin/env python3
"""
Train BEHRT NextVisit on cleaned CCSR data without overwriting prior artifacts.

Outputs are written to a timestamped run directory under data/models/clean_runs/.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    import mlflow  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
import pytorch_pretrained_bert as Bert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj, create_folder
from common.pytorch import save_model
from model.utils import age_vocab
from dataLoader.NextXVisit import NextVisit
from model.NextXVisit import BertForMultiLabelPrediction
from model import optimiser


class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get("vocab_size"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=config.get("num_hidden_layers"),
            num_attention_heads=config.get("num_attention_heads"),
            intermediate_size=config.get("intermediate_size"),
            hidden_act=config.get("hidden_act"),
            hidden_dropout_prob=config.get("hidden_dropout_prob"),
            attention_probs_dropout_prob=config.get("attention_probs_dropout_prob"),
            max_position_embeddings=config.get("max_position_embedding"),
            initializer_range=config.get("initializer_range"),
        )
        self.seg_vocab_size = config.get("seg_vocab_size")
        self.age_vocab_size = config.get("age_vocab_size")


def format_label_vocab(token2idx):
    token2idx = token2idx.copy()
    for tok in ["PAD", "SEP", "CLS", "MASK"]:
        if tok in token2idx:
            del token2idx[tok]
    label_vocab = {}
    for i, token in enumerate(token2idx.keys()):
        label_vocab[token] = i
    return label_vocab


def targets_to_multihot(targets, mlb, device=None):
    """Convert padded label tensors into clean multi-hot vectors for loss/metrics."""
    target_rows = targets.detach().cpu().numpy()
    cleaned_targets = []
    for row in target_rows:
        cleaned = [int(x) for x in row if int(x) >= 0]
        cleaned_targets.append(cleaned)

    target_ml = torch.tensor(mlb.transform(cleaned_targets), dtype=torch.float32)
    if device is not None:
        target_ml = target_ml.to(device)
    return target_ml


def build_label_support_counts(train_df, label_vocab):
    support = Counter()
    for labels in train_df["label"]:
        for label in set(normalize_label_row(labels)):
            if label in label_vocab:
                support[label] += 1
    return support


def summarize_support_tiers(label_supports):
    tiers = [
        (0, 1),
        (1, 10),
        (10, 50),
        (50, 100),
        (100, 500),
        (500, 1000),
        (1000, 5000),
        (5000, float("inf")),
    ]
    values = np.array(list(label_supports.values()), dtype=np.int64)
    summary = []
    for lower, upper in tiers:
        if upper == float("inf"):
            mask = values >= lower
            tier_label = f">={lower}"
        else:
            mask = (values >= lower) & (values < upper)
            tier_label = f"[{lower}, {upper})"
        summary.append({"tier": tier_label, "labels": int(mask.sum())})
    return summary


def compute_top_k_metrics(y_true, y_prob, top_k_values=(5, 10)):
    metrics = {}
    _, num_labels = y_true.shape
    for top_k in top_k_values:
        k = min(int(top_k), num_labels)
        precisions = []
        recalls = []
        for row_true, row_prob in zip(y_true, y_prob):
            positives = row_true.sum()
            if positives == 0:
                continue
            top_indices = np.argpartition(row_prob, -k)[-k:]
            top_hits = row_true[top_indices].sum()
            precisions.append(float(top_hits) / float(k))
            recalls.append(float(top_hits) / float(positives))
        metrics[f"top_{k}_precision"] = float(np.mean(precisions)) if precisions else 0.0
        metrics[f"top_{k}_recall"] = float(np.mean(recalls)) if recalls else 0.0
    return metrics


def compute_recurring_mask(df, label_vocab):
    """Boolean mask of shape (num_rows, num_labels): True where a label's code has
    already appeared in the patient's own visit history (the `code` column, i.e.
    the model's input sequence), meaning it would count as a RECURRING diagnosis
    if predicted/true. False marks a NEW diagnosis (first occurrence for that
    patient). Computed independently of the true label, so it can mask both true
    positives and false positives/negatives.
    """
    non_code_tokens = {"PAD", "SEP", "CLS", "MASK"}
    num_rows = len(df)
    num_labels = len(label_vocab)
    recurring_mask = np.zeros((num_rows, num_labels), dtype=bool)

    for row_idx, history_codes in enumerate(df["code"]):
        history_set = set(normalize_label_row(history_codes)) - non_code_tokens
        for code in history_set:
            label_idx = label_vocab.get(code)
            if label_idx is not None:
                recurring_mask[row_idx, label_idx] = True

    return recurring_mask


def compute_new_recurring_metrics(y_true, y_prob, threshold, recurring_mask, top_k_values=(5, 10)):
    """Split evaluation into RECURRING (code already seen in patient history) vs
    NEW (first occurrence for that patient) diagnosis predictions. Reports
    micro precision/recall/F1/AUC/APS plus top-k recall computed independently
    on each subset.
    """
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)

    results = {}
    for name, mask in (("recurring", recurring_mask), ("new", ~recurring_mask)):
        sel_true = y_true_bool[mask]
        sel_pred = y_pred_bool[mask]
        sel_prob = y_prob[mask]

        tp = int(np.sum(sel_true & sel_pred))
        fp = int(np.sum(~sel_true & sel_pred))
        fn = int(np.sum(sel_true & ~sel_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int(sel_true.sum())

        if 0 < support < len(sel_true):
            try:
                auc = float(roc_auc_score(sel_true, sel_prob))
            except ValueError:
                auc = float("nan")
            aps = float(average_precision_score(sel_true, sel_prob))
        else:
            auc = float("nan")
            aps = float("nan")

        results[name] = {
            "support_positive_positions": support,
            "support_total_positions": int(mask.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": auc,
            "aps": aps,
        }

    # Row-level top-k recall, restricted to recurring-only or new-only ground truth
    # (predictions are still ranked over the full label space for each row).
    for name, mask in (("recurring", recurring_mask), ("new", ~recurring_mask)):
        masked_true = np.where(mask, y_true, 0)
        results[name]["top_k_metrics"] = compute_top_k_metrics(masked_true, y_prob, top_k_values=top_k_values)

    return results


def compute_per_class_metrics(y_true, y_prob, y_pred, label_vocab, train_supports, excluded_label_indices=None):
    idx_to_label = {idx: label for label, idx in label_vocab.items()}
    per_class = []
    excluded_label_indices = set(excluded_label_indices or [])

    for class_idx in range(y_true.shape[1]):
        if class_idx in excluded_label_indices:
            continue
        label = idx_to_label[class_idx]
        labels_true = y_true[:, class_idx]
        labels_prob = y_prob[:, class_idx]
        labels_pred = y_pred[:, class_idx]

        support = int(labels_true.sum())
        predicted_positive = int(labels_pred.sum())

        precision = precision_score(labels_true, labels_pred, zero_division=0)
        recall = recall_score(labels_true, labels_pred, zero_division=0)
        f1 = f1_score(labels_true, labels_pred, zero_division=0)

        if support == 0:
            average_precision = 0.0
            roc_auc = float("nan")
        else:
            average_precision = float(average_precision_score(labels_true, labels_prob))
            try:
                roc_auc = float(roc_auc_score(labels_true, labels_prob))
            except ValueError:
                roc_auc = float("nan")

        per_class.append(
            {
                "label": label,
                "class_idx": class_idx,
                "train_support": int(train_supports.get(label, 0)),
                "test_support": support,
                "predicted_positives": predicted_positive,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "average_precision": float(average_precision),
                "roc_auc": float(roc_auc),
            }
        )

    per_class.sort(key=lambda row: row["average_precision"], reverse=True)
    return per_class


def collect_eval_arrays(model, loader, mlb, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            age_ids, input_ids, posi_ids, segment_ids, att_mask, targets, _ = batch
            input_ids = input_ids.to(device)
            age_ids = age_ids.to(device)
            posi_ids = posi_ids.to(device)
            segment_ids = segment_ids.to(device)
            att_mask = att_mask.to(device)

            target_ml = targets_to_multihot(targets, mlb)
            logits = model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=att_mask)
            probs = sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(target_ml.numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    return y_true, y_prob


def evaluate_from_arrays(
    y_true,
    y_prob,
    threshold=0.5,
    label_vocab=None,
    train_supports=None,
    include_per_class=True,
    exclude_labels=None,
):
    excluded_label_indices = set()
    metric_true = y_true
    metric_prob = y_prob

    if exclude_labels and label_vocab is not None:
        excluded_label_indices = {
            int(label_vocab[label])
            for label in exclude_labels
            if label in label_vocab
        }
        keep_indices = [idx for idx in range(y_true.shape[1]) if idx not in excluded_label_indices]
        if not keep_indices:
            raise ValueError("No labels left for metric computation after exclusions")
        metric_true = y_true[:, keep_indices]
        metric_prob = y_prob[:, keep_indices]

    y_pred = (metric_prob >= float(threshold)).astype(np.int64)

    aps = average_precision_score(metric_true, metric_prob, average="samples")
    try:
        auc = roc_auc_score(metric_true, metric_prob, average="samples")
    except ValueError:
        auc = float("nan")

    micro_precision = precision_score(metric_true, y_pred, average="micro", zero_division=0)
    micro_recall = recall_score(metric_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(metric_true, y_pred, average="micro", zero_division=0)
    samples_precision = precision_score(metric_true, y_pred, average="samples", zero_division=0)
    samples_recall = recall_score(metric_true, y_pred, average="samples", zero_division=0)
    samples_f1 = f1_score(metric_true, y_pred, average="samples", zero_division=0)
    subset_accuracy = float(np.mean(np.all(metric_true == y_pred, axis=1)))
    hamming_accuracy = 1.0 - hamming_loss(metric_true, y_pred)
    top_k_metrics = compute_top_k_metrics(metric_true, metric_prob, top_k_values=(5, 10))

    per_class_metrics = []
    support_summary = []
    if include_per_class and label_vocab is not None and train_supports is not None:
        full_pred = (y_prob >= float(threshold)).astype(np.int64)
        per_class_metrics = compute_per_class_metrics(
            y_true,
            y_prob,
            full_pred,
            label_vocab,
            train_supports,
            excluded_label_indices=excluded_label_indices,
        )
        support_summary = summarize_support_tiers(train_supports)

    return {
        "threshold": float(threshold),
        "sample_wise_aps": float(aps),
        "sample_wise_auc": float(auc),
        "subset_accuracy": float(subset_accuracy),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "samples_precision": float(samples_precision),
        "samples_recall": float(samples_recall),
        "samples_f1": float(samples_f1),
        "hamming_accuracy": float(hamming_accuracy),
        "top_k_metrics": top_k_metrics,
        "per_class_metrics": per_class_metrics,
        "train_support_tiers": support_summary,
    }


def metric_for_threshold(metrics, metric_name):
    if metric_name in metrics:
        return float(metrics[metric_name])
    if metric_name.startswith("top_"):
        return float(metrics.get("top_k_metrics", {}).get(metric_name, 0.0))
    raise ValueError(f"Unsupported threshold metric: {metric_name}")


def tune_decision_threshold(
    y_true,
    y_prob,
    metric_name="micro_f1",
    threshold_min=0.1,
    threshold_max=0.9,
    threshold_step=0.05,
):
    thresholds = np.arange(float(threshold_min), float(threshold_max) + 1e-9, float(threshold_step))
    best_threshold = 0.5
    best_score = float("-inf")
    history = []

    for threshold in thresholds:
        metrics = evaluate_from_arrays(y_true, y_prob, threshold=threshold, include_per_class=False)
        score = metric_for_threshold(metrics, metric_name)
        history.append({"threshold": float(threshold), "score": float(score)})
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score, history


def flatten_scalar_metrics(metrics, prefix=""):
    flat = {}
    for key, value in metrics.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_scalar_metrics(value, prefix=f"{full_key}."))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            scalar = float(value)
            if np.isfinite(scalar):
                flat[full_key] = scalar
    return flat


def get_mlflow_setup(project_root, run_id):
    use_mlflow = os.getenv("MLFLOW_ENABLE", "1") == "1"
    if not use_mlflow:
        return {"enabled": False}

    if mlflow is None:
        print("MLflow logging disabled: install mlflow or set MLFLOW_ENABLE=0")
        return {"enabled": False}

    default_tracking_uri = f"file://{(project_root / 'mlruns').resolve()}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "BEHRT_Clean_NextVisit")
    run_name = os.getenv("MLFLOW_RUN_NAME", run_id)
    using_default_file_store = tracking_uri == default_tracking_uri

    return {
        "enabled": True,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "using_default_file_store": using_default_file_store,
        "tracking_note": (
            "MLflow file-based backend is convenient locally but deprecated long-term; "
            "prefer MLFLOW_TRACKING_URI=sqlite:///mlflow.db for durable experiment tracking."
        ),
    }


def normalize_label_row(labels):
    if isinstance(labels, np.ndarray):
        return [str(label) for label in labels.tolist()]
    if isinstance(labels, (list, tuple, set)):
        return [str(label) for label in labels]
    if pd.isna(labels):
        return []
    return [str(labels)]


def build_label_subset(train_df, base_label_vocab, top_k_labels=0, min_label_freq=0.0):
    if top_k_labels > 0 and min_label_freq > 0:
        raise ValueError("Set only one of TOP_K_LABELS or MIN_LABEL_FREQ")

    if top_k_labels <= 0 and min_label_freq <= 0:
        return base_label_vocab, train_df, None

    label_counts = Counter()
    for labels in train_df["label"]:
        label_counts.update(set(normalize_label_row(labels)))

    if top_k_labels > 0:
        selected_labels = [label for label, _ in label_counts.most_common(top_k_labels)]
        strategy = f"top-{top_k_labels} labels"
    else:
        selected_labels = [label for label, count in label_counts.items() if (count / len(train_df)) >= min_label_freq]
        strategy = f"labels with train prevalence >= {min_label_freq * 100:.2f}%"

    keep_set = set(selected_labels)
    selected_vocab_tokens = [label for label in base_label_vocab if label in keep_set]
    if "UNK" in base_label_vocab and "UNK" not in selected_vocab_tokens:
        selected_vocab_tokens.append("UNK")
    label_vocab = {label: idx for idx, label in enumerate(selected_vocab_tokens)}

    filtered_df = train_df.copy()
    filtered_df["label"] = filtered_df["label"].map(lambda labels: np.array([label for label in normalize_label_row(labels) if label in keep_set], dtype=object))
    filtered_df = filtered_df[filtered_df["label"].map(len) > 0].reset_index(drop=True)

    return label_vocab, filtered_df, strategy


def evaluate(
    model,
    loader,
    mlb,
    device,
    label_vocab=None,
    train_supports=None,
    threshold=0.5,
    include_per_class=True,
    exclude_labels=None,
):
    y_true, y_prob = collect_eval_arrays(model, loader, mlb, device)
    return evaluate_from_arrays(
        y_true,
        y_prob,
        threshold=threshold,
        label_vocab=label_vocab,
        train_supports=train_supports,
        include_per_class=include_per_class,
        exclude_labels=exclude_labels,
    )


def compute_pos_weight_from_train(train_df, label_vocab, max_pos_weight=30.0):
    """Compute stable per-class positive weights for BCEWithLogitsLoss.

    Weight formula: neg_count / pos_count, clipped to avoid extreme gradients.
    """
    num_classes = len(label_vocab)
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for labels in train_df["label"]:
        for code in set(labels):
            idx = label_vocab.get(code)
            if idx is not None:
                class_counts[idx] += 1

    n_samples = len(train_df)
    neg_counts = n_samples - class_counts

    safe_pos = np.maximum(class_counts, 1)
    pos_weight = neg_counts / safe_pos
    pos_weight = np.clip(pos_weight, 1.0, float(max_pos_weight))

    return torch.tensor(pos_weight, dtype=torch.float32)


def main():
    seed = int(os.getenv("SEED", "42"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"

    run_id = datetime.now().strftime("clean_run_%Y%m%d_%H%M%S")
    run_dir = project_root / "data" / "models" / "clean_runs" / run_id
    create_folder(str(run_dir))

    mlflow_setup = get_mlflow_setup(project_root, run_id)
    mlflow_enabled = mlflow_setup["enabled"]
    if mlflow_enabled:
        mlflow.set_tracking_uri(mlflow_setup["tracking_uri"])
        mlflow.set_experiment(mlflow_setup["experiment_name"])
        mlflow.start_run(run_name=mlflow_setup["run_name"])
        mlflow.set_tags(
            {
                "project": "BEHRT",
                "task": "next_visit_multilabel",
                "dataset": "MIMIC-IV-CCSR-clean",
                "run_id": run_id,
                "mlflow_tracking_note": mlflow_setup["tracking_note"],
            }
        )
        print(
            f"MLflow enabled | tracking_uri={mlflow_setup['tracking_uri']} "
            f"| experiment={mlflow_setup['experiment_name']}"
        )
        if mlflow_setup["using_default_file_store"]:
            print(
                "MLflow note: file store is fine for local dev, but for long-term usage "
                "set MLFLOW_TRACKING_URI=sqlite:///mlflow.db"
            )

    train_path = data_dir / "train_nextvisit_ccsr_clean.parquet"
    val_path = data_dir / "val_nextvisit_ccsr_clean.parquet"
    test_path = data_dir / "test_nextvisit_ccsr_clean.parquet"
    vocab_path = data_dir / "vocab_ccsr_clean"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # Optional quick-run controls for faster validation cycles.
    sample_limit = int(os.getenv("SAMPLE_LIMIT", "0"))
    if sample_limit > 0:
        train_df = train_df.sample(n=min(sample_limit, len(train_df)), random_state=seed).reset_index(drop=True)
        val_df = val_df.sample(n=min(max(1, sample_limit // 5), len(val_df)), random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(n=min(max(1, sample_limit // 5), len(test_df)), random_state=seed).reset_index(drop=True)

    bert_vocab = load_obj(str(vocab_path))
    age_vocab_dict, _ = age_vocab(max_age=110, symbol=None)
    base_label_vocab = format_label_vocab(bert_vocab["token2idx"])

    top_k_labels = int(os.getenv("TOP_K_LABELS", "0"))
    min_label_freq = float(os.getenv("MIN_LABEL_FREQ", "0.0"))
    threshold_tuning_enabled = os.getenv("TUNE_THRESHOLD", "1") == "1"
    threshold_metric = os.getenv("THRESHOLD_METRIC", "micro_f1")
    threshold_min = float(os.getenv("THRESHOLD_MIN", "0.1"))
    threshold_max = float(os.getenv("THRESHOLD_MAX", "0.9"))
    threshold_step = float(os.getenv("THRESHOLD_STEP", "0.05"))
    eval_threshold = float(os.getenv("EVAL_THRESHOLD", "0.5"))
    metric_exclude_labels = tuple(
        token.strip() for token in os.getenv("METRIC_EXCLUDE_LABELS", "UNK").split(",") if token.strip()
    )
    label_vocab, train_df, label_strategy = build_label_subset(
        train_df,
        base_label_vocab,
        top_k_labels=top_k_labels,
        min_label_freq=min_label_freq,
    )

    if label_strategy is not None:
        keep_set = set(label_vocab.keys())
        val_df = val_df.copy()
        test_df = test_df.copy()
        val_df["label"] = val_df["label"].map(lambda labels: np.array([label for label in normalize_label_row(labels) if label in keep_set], dtype=object))
        test_df["label"] = test_df["label"].map(lambda labels: np.array([label for label in normalize_label_row(labels) if label in keep_set], dtype=object))
        val_df = val_df[val_df["label"].map(len) > 0].reset_index(drop=True)
        test_df = test_df[test_df["label"].map(len) > 0].reset_index(drop=True)
        print(f"Using label subset strategy: {label_strategy}")
        print(f"Retained labels: {len(label_vocab)}")
        print(f"Filtered rows: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_label_supports = build_label_support_counts(train_df, label_vocab)

    global_params = {
        "batch_size": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_len_seq": 64,
    }

    model_config = {
        "vocab_size": len(bert_vocab["token2idx"]),
        "hidden_size": 288,
        "seg_vocab_size": 2,
        "age_vocab_size": len(age_vocab_dict),
        "max_position_embedding": global_params["max_len_seq"],
        "hidden_dropout_prob": 0.1,
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "attention_probs_dropout_prob": 0.1,
        "intermediate_size": 256,
        "hidden_act": "gelu",
        "initializer_range": 0.02,
    }

    if mlflow_enabled:
        mlflow.log_params(
            {
                "seed": seed,
                "sample_limit": sample_limit,
                "top_k_labels": top_k_labels,
                "min_label_freq": min_label_freq,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "label_count": len(label_vocab),
                "device": global_params["device"],
                "batch_size": global_params["batch_size"],
                "max_len_seq": global_params["max_len_seq"],
                "eval_threshold": eval_threshold,
                "tune_threshold": int(threshold_tuning_enabled),
                "threshold_metric": threshold_metric,
                "threshold_min": threshold_min,
                "threshold_max": threshold_max,
                "threshold_step": threshold_step,
                "metric_exclude_labels": ",".join(metric_exclude_labels),
            }
        )
        mlflow.log_params({f"model.{k}": v for k, v in model_config.items()})

    feature_dict = {"word": True, "seg": True, "age": True, "position": True}

    train_set = NextVisit(
        token2idx=bert_vocab["token2idx"],
        label2idx=label_vocab,
        age2idx=age_vocab_dict,
        dataframe=train_df,
        max_len=global_params["max_len_seq"],
    )
    val_set = NextVisit(
        token2idx=bert_vocab["token2idx"],
        label2idx=label_vocab,
        age2idx=age_vocab_dict,
        dataframe=val_df,
        max_len=global_params["max_len_seq"],
    )
    test_set = NextVisit(
        token2idx=bert_vocab["token2idx"],
        label2idx=label_vocab,
        age2idx=age_vocab_dict,
        dataframe=test_df,
        max_len=global_params["max_len_seq"],
    )

    train_loader = DataLoader(train_set, batch_size=global_params["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=global_params["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=global_params["batch_size"], shuffle=False, num_workers=0)

    num_epochs = int(os.getenv("EPOCHS", "3"))
    total_train_steps = len(train_loader) * num_epochs

    mlb = MultiLabelBinarizer(classes=list(label_vocab.values()))
    mlb.fit([[x] for x in list(label_vocab.values())])

    conf = BertConfig(model_config)
    model = BertForMultiLabelPrediction(conf, num_labels=len(label_vocab), feature_dict=feature_dict)
    model = model.to(global_params["device"])

    optim = optimiser.adam(
        params=list(model.named_parameters()),
        config={"lr": 5e-5, "warmup_proportion": 0.1, "weight_decay": 0.01, "t_total": total_train_steps},
    )

    best_val_aps = -1.0
    best_val_metrics = None
    best_epoch = None
    best_model_path = run_dir / "behrt_nextvisit_ccsr_clean_best.pt"
    log_every = int(os.getenv("LOG_EVERY", "100"))
    grad_clip_norm = float(os.getenv("GRAD_CLIP_NORM", "1.0"))
    early_stop_patience = int(os.getenv("EARLY_STOP_PATIENCE", "0"))
    use_pos_weight = os.getenv("USE_POS_WEIGHT", "0") == "1"
    max_pos_weight = float(os.getenv("MAX_POS_WEIGHT", "30.0"))
    epochs_without_improvement = 0

    if use_pos_weight:
        pos_weight = compute_pos_weight_from_train(train_df, label_vocab, max_pos_weight=max_pos_weight)
        pos_weight = pos_weight.to(global_params["device"])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight (max clip={max_pos_weight:.1f})")
    else:
        loss_fn = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            age_ids, input_ids, posi_ids, segment_ids, att_mask, targets, _ = batch
            input_ids = input_ids.to(global_params["device"])
            age_ids = age_ids.to(global_params["device"])
            posi_ids = posi_ids.to(global_params["device"])
            segment_ids = segment_ids.to(global_params["device"])
            att_mask = att_mask.to(global_params["device"])
            target_ml = targets_to_multihot(targets, mlb, device=global_params["device"])

            optim.zero_grad()
            if loss_fn is None:
                loss, _ = model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=att_mask, labels=target_ml)
            else:
                logits = model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=att_mask)
                loss = loss_fn(logits, target_ml)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1

            if step % max(1, log_every) == 0:
                print(f"Epoch {epoch+1}/{num_epochs} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        val_metrics = evaluate(
            model,
            val_loader,
            mlb,
            global_params["device"],
            label_vocab=label_vocab,
            train_supports=train_label_supports,
            threshold=eval_threshold,
            include_per_class=False,
            exclude_labels=metric_exclude_labels,
        )
        avg_loss = epoch_loss / max(1, n_batches)
        print(
            f"Epoch {epoch+1}/{num_epochs} - loss={avg_loss:.4f} "
            f"val_APS={val_metrics['sample_wise_aps']:.4f} "
            f"val_AUC={val_metrics['sample_wise_auc']:.4f} "
            f"val_micro_F1={val_metrics['micro_f1']:.4f} "
            f"val_top10_R={val_metrics['top_k_metrics']['top_10_recall']:.4f}"
        )

        if mlflow_enabled:
            mlflow.log_metric("train.loss", float(avg_loss), step=epoch + 1)
            val_scalar_metrics = flatten_scalar_metrics(
                {
                    k: v
                    for k, v in val_metrics.items()
                    if k not in ("per_class_metrics", "train_support_tiers")
                },
                prefix="val.",
            )
            mlflow.log_metrics(val_scalar_metrics, step=epoch + 1)

        if val_metrics["sample_wise_aps"] > best_val_aps:
            best_val_aps = val_metrics["sample_wise_aps"]
            best_val_metrics = val_metrics
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_model(str(best_model_path), model)
        else:
            epochs_without_improvement += 1

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs (patience={early_stop_patience}).")
            break

    # Load best model weights for final test metrics
    model.load_state_dict(torch.load(best_model_path, map_location=global_params["device"]))
    val_y_true, val_y_prob = collect_eval_arrays(model, val_loader, mlb, global_params["device"])

    if threshold_tuning_enabled:
        tuned_threshold, tuned_score, threshold_history = tune_decision_threshold(
            val_y_true,
            val_y_prob,
            metric_name=threshold_metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
        )
        print(
            f"Tuned threshold on val: {tuned_threshold:.3f} "
            f"using {threshold_metric}={tuned_score:.4f}"
        )
    else:
        tuned_threshold = eval_threshold
        tuned_score = float("nan")
        threshold_history = []

    val_metrics_threshold_0_5 = evaluate_from_arrays(
        val_y_true,
        val_y_prob,
        threshold=0.5,
        label_vocab=label_vocab,
        train_supports=train_label_supports,
        include_per_class=False,
        exclude_labels=metric_exclude_labels,
    )
    val_metrics_tuned_threshold = evaluate_from_arrays(
        val_y_true,
        val_y_prob,
        threshold=tuned_threshold,
        label_vocab=label_vocab,
        train_supports=train_label_supports,
        include_per_class=False,
        exclude_labels=metric_exclude_labels,
    )

    test_metrics_threshold_0_5 = evaluate(
        model,
        test_loader,
        mlb,
        global_params["device"],
        label_vocab=label_vocab,
        train_supports=train_label_supports,
        threshold=0.5,
        include_per_class=False,
        exclude_labels=metric_exclude_labels,
    )

    test_metrics = evaluate(
        model,
        test_loader,
        mlb,
        global_params["device"],
        label_vocab=label_vocab,
        train_supports=train_label_supports,
        threshold=tuned_threshold,
        include_per_class=True,
        exclude_labels=metric_exclude_labels,
    )

    per_class_metrics_path = run_dir / "per_class_metrics.csv"
    pd.DataFrame(test_metrics["per_class_metrics"]).to_csv(per_class_metrics_path, index=False)

    # NEW vs RECURRING diagnosis breakdown on the test set: does the model mostly
    # re-predict codes the patient already has, or does it also catch genuinely
    # new diagnoses for the next visit?
    test_y_true, test_y_prob = collect_eval_arrays(model, test_loader, mlb, global_params["device"])
    recurring_mask_full = compute_recurring_mask(test_df, label_vocab)

    metric_true_nr = test_y_true
    metric_prob_nr = test_y_prob
    metric_mask_nr = recurring_mask_full
    if metric_exclude_labels:
        excluded_indices_nr = {
            int(label_vocab[label]) for label in metric_exclude_labels if label in label_vocab
        }
        keep_indices_nr = [idx for idx in range(test_y_true.shape[1]) if idx not in excluded_indices_nr]
        metric_true_nr = test_y_true[:, keep_indices_nr]
        metric_prob_nr = test_y_prob[:, keep_indices_nr]
        metric_mask_nr = recurring_mask_full[:, keep_indices_nr]

    new_vs_recurring_metrics = compute_new_recurring_metrics(
        metric_true_nr, metric_prob_nr, tuned_threshold, metric_mask_nr
    )

    label_support_summary = {
        "train_label_count": len(train_label_supports),
        "support_tiers": summarize_support_tiers(train_label_supports),
        "top_20_labels": [
            {"label": label, "train_support": int(count)}
            for label, count in train_label_supports.most_common(20)
        ],
    }

    result = {
        "schema_version": "2.0",
        "run_id": run_id,
        "data": {
            "train": str(train_path.name),
            "val": str(val_path.name),
            "test": str(test_path.name),
            "vocab": str(vocab_path.name) + ".pkl",
        },
        "model_config": model_config,
        "training": {"epochs": num_epochs, "batch_size": global_params["batch_size"]},
        "run_controls": {
            "sample_limit": sample_limit,
            "log_every": log_every,
            "grad_clip_norm": grad_clip_norm,
            "early_stop_patience": early_stop_patience,
            "seed": seed,
            "use_pos_weight": use_pos_weight,
            "max_pos_weight": max_pos_weight,
            "eval_threshold": eval_threshold,
            "tune_threshold": threshold_tuning_enabled,
            "threshold_metric": threshold_metric,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "threshold_step": threshold_step,
            "metric_exclude_labels": list(metric_exclude_labels),
        },
        "label_support_summary": label_support_summary,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "threshold_tuning": {
            "enabled": threshold_tuning_enabled,
            "metric": threshold_metric,
            "selected_threshold": tuned_threshold,
            "selected_score": tuned_score,
            "history": threshold_history,
        },
        "val_metrics_threshold_0_5": val_metrics_threshold_0_5,
        "val_metrics_tuned_threshold": val_metrics_tuned_threshold,
        "metrics_threshold_0_5": test_metrics_threshold_0_5,
        "metrics": test_metrics,
        "new_vs_recurring_metrics": new_vs_recurring_metrics,
        "artifacts": {"best_model": str(best_model_path.relative_to(project_root))},
    }

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if mlflow_enabled:
        if best_epoch is not None:
            mlflow.log_metric("best.epoch", float(best_epoch))
        if best_val_metrics is not None:
            best_scalar_metrics = flatten_scalar_metrics(
                {
                    k: v
                    for k, v in best_val_metrics.items()
                    if k not in ("per_class_metrics", "train_support_tiers")
                },
                prefix="best_val.",
            )
            mlflow.log_metrics(best_scalar_metrics)

        test_scalar_metrics = flatten_scalar_metrics(
            {
                k: v
                for k, v in test_metrics.items()
                if k not in ("per_class_metrics", "train_support_tiers")
            },
            prefix="test.",
        )
        mlflow.log_metrics(test_scalar_metrics)
        test_fixed_scalar_metrics = flatten_scalar_metrics(
            {
                k: v
                for k, v in test_metrics_threshold_0_5.items()
                if k not in ("per_class_metrics", "train_support_tiers")
            },
            prefix="test_fixed_0_5.",
        )
        mlflow.log_metrics(test_fixed_scalar_metrics)
        val_tuned_scalar_metrics = flatten_scalar_metrics(
            {
                k: v
                for k, v in val_metrics_tuned_threshold.items()
                if k not in ("per_class_metrics", "train_support_tiers")
            },
            prefix="val_tuned.",
        )
        mlflow.log_metrics(val_tuned_scalar_metrics)
        new_recurring_scalar_metrics = flatten_scalar_metrics(new_vs_recurring_metrics, prefix="test_new_recurring.")
        mlflow.log_metrics(new_recurring_scalar_metrics)
        if threshold_tuning_enabled:
            mlflow.log_metric("threshold.selected", float(tuned_threshold))
            if np.isfinite(tuned_score):
                mlflow.log_metric("threshold.selected_score", float(tuned_score))
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(per_class_metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(best_model_path), artifact_path="models")
        mlflow.end_run()

    print("=" * 72)
    print(f"Saved run to: {run_dir}")
    print(f"Seed: {seed}")
    print(f"Test APS: {test_metrics['sample_wise_aps']:.4f}")
    print(f"Test AUC: {test_metrics['sample_wise_auc']:.4f}")
    print(f"Test micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"Test top-10 recall: {test_metrics['top_k_metrics']['top_10_recall']:.4f}")
    print(f"Threshold used for final test metrics: {tuned_threshold:.3f}")
    recurring_m = new_vs_recurring_metrics["recurring"]
    new_m = new_vs_recurring_metrics["new"]
    print(
        "Test RECURRING vs NEW diagnoses - "
        f"recurring (n={recurring_m['support_positive_positions']}): "
        f"P={recurring_m['precision']:.4f} R={recurring_m['recall']:.4f} "
        f"F1={recurring_m['f1']:.4f} AUC={recurring_m['auc']:.4f} | "
        f"new (n={new_m['support_positive_positions']}): "
        f"P={new_m['precision']:.4f} R={new_m['recall']:.4f} "
        f"F1={new_m['f1']:.4f} AUC={new_m['auc']:.4f}"
    )
    print(f"Metrics file: {metrics_path}")
    print(f"Per-class metrics CSV: {per_class_metrics_path}")


if __name__ == "__main__":
    main()
