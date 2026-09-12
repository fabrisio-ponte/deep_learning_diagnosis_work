#!/usr/bin/env python3
"""
Simple Interpretability: attention capture + CLS attention + attention-vs-
recency/recurring + code-embedding nearest neighbors.

This is the "must-do" pick from BEHRT_mechanism_interpretability/simple_interpretability/README.md
(items #1, #2, #3, #4, #6). It reuses the same "load run from metrics.json
-> rebuild vocab/loader -> load checkpoint -> recompute predictions" pattern
established in eda/08, eda/09, eda/10 of BEHRT_Project.

Caveat (state explicitly in any writeup): raw attention weights are a
DESCRIPTIVE signal, not a validated causal explanation of model behavior
(Jain & Wallace, 2019, "Attention is not Explanation"). Results here show
what the model attends to, not proof that this attention causes its
predictions -- for a causal/attribution-based claim, see
../deep_interpretability/ (Integrated Gradients).

Usage:
    RUN_DIR=data/models/clean_runs/clean_run_20260910_124255 \
        python3.12 BEHRT_mechanism_interpretability/simple_interpretability/attention_and_embeddings.py
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "BEHRT_main" / "BEHRT_thesis" / "BEHRT_Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_pretrained_bert as Bert  # noqa: E402

from common.common import load_obj  # noqa: E402
from model.utils import age_vocab  # noqa: E402
from dataLoader.NextXVisit import NextVisit  # noqa: E402
from model.NextXVisit import BertForMultiLabelPrediction  # noqa: E402

from scripts.train_nextvisit_clean import (  # noqa: E402
    BertConfig,
    build_label_subset,
    format_label_vocab,
    targets_to_multihot,
)

N_SAMPLE_PATIENTS = 500  # patients used for attention aggregation statistics
N_CASE_STUDIES = 8  # patients saved as full qualitative examples
TOP_K_NEIGHBORS = 10
EXAMPLE_CODES_FOR_NEIGHBORS = [
    "CCSR_END010",  # diabetes (most common code in dataset)
    "CCSR_CIR003",
    "CCSR_CIR014",
    "CCSR_MUS033",
    "CCSR_NVS011",
]


def find_latest_run_dir(project_root):
    runs_root = project_root / "data" / "models" / "clean_runs"
    candidates = [d for d in runs_root.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No clean_run_* directories with metrics.json found under {runs_root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


# ---------------------------------------------------------------------------
# Monkey-patch BertSelfAttention.forward to capture attention_probs.
# This does NOT change model weights or outputs - it just stashes the
# already-computed attention_probs tensor on the module instance before
# returning the (unchanged) context_layer.
# ---------------------------------------------------------------------------
_ORIGINAL_SELF_ATTENTION_FORWARD = Bert.modeling.BertSelfAttention.forward


def _capturing_forward(self, hidden_states, attention_mask):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_scores = attention_scores + attention_mask

    attention_probs = nn.Softmax(dim=-1)(attention_scores)
    # Capture point: [batch, heads, seq_len, seq_len], pre-dropout, detached.
    self.saved_attention_probs = attention_probs.detach().cpu()

    attention_probs = self.dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return context_layer


def install_attention_capture():
    Bert.modeling.BertSelfAttention.forward = _capturing_forward


def restore_attention_forward():
    Bert.modeling.BertSelfAttention.forward = _ORIGINAL_SELF_ATTENTION_FORWARD


def get_layer_attention_modules(model):
    """Returns list of BertSelfAttention submodules, in layer order."""
    return [layer.attention.self for layer in model.bert.encoder.layer]


def main():
    project_root = PROJECT_ROOT

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

    data_dir = project_root / "data" / "processed"
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

    # Sample a subset of the test set for attention-capture (running the full
    # 18k-row test set through a per-row Python capture loop is unnecessary
    # for aggregate statistics; N_SAMPLE_PATIENTS is enough for stable means).
    rng = np.random.default_rng(42)
    sample_size = min(N_SAMPLE_PATIENTS, len(test_df))
    sample_idx = rng.choice(len(test_df), size=sample_size, replace=False)
    sample_idx.sort()
    sample_df = test_df.iloc[sample_idx].reset_index(drop=True)

    sample_set = NextVisit(
        token2idx=bert_vocab["token2idx"],
        label2idx=label_vocab,
        age2idx=age_vocab_dict,
        dataframe=sample_df,
        max_len=max_len_seq,
    )
    sample_loader = DataLoader(sample_set, batch_size=32, shuffle=False, num_workers=0)

    mlb = MultiLabelBinarizer(classes=list(label_vocab.values()))
    mlb.fit([[x] for x in list(label_vocab.values())])

    conf = BertConfig(model_config)
    model = BertForMultiLabelPrediction(
        conf, num_labels=len(label_vocab), feature_dict={"word": True, "seg": True, "age": True, "position": True}
    )
    model.load_state_dict(torch.load(run_dir / "behrt_nextvisit_ccsr_clean_best.pt", map_location=device))
    model = model.to(device)
    model.eval()

    install_attention_capture()
    attention_modules = get_layer_attention_modules(model)
    num_layers = len(attention_modules)
    print(f"Model has {num_layers} transformer layers; attention capture installed.")

    idx_to_token = {idx: tok for tok, idx in bert_vocab["token2idx"].items()}

    # Per-position aggregation buckets (distance from END of sequence, i.e.
    # 0 = last real token before padding, larger = further back in history).
    max_distance_bucket = max_len_seq
    recency_attn_sum = np.zeros(max_distance_bucket, dtype=np.float64)
    recency_attn_count = np.zeros(max_distance_bucket, dtype=np.float64)

    # "target_hit" = history position whose code is one of the TRUE next-visit
    # target labels for that row (i.e. a code that will recur at the next
    # visit); "target_miss" = history position whose code will NOT recur as a
    # target label. This answers: does the model attend more to history codes
    # that are about to recur?
    target_hit_attn_sum = 0.0
    target_hit_attn_count = 0.0
    target_miss_attn_sum = 0.0
    target_miss_attn_count = 0.0

    sep_attn_mass = []  # fraction of total (non-pad) attention mass landing on SEP tokens, per row
    real_token_attn_mass = []  # fraction landing on non-special tokens

    case_studies = []

    print("Running attention-capture forward passes over sample...")
    row_offset = 0
    with torch.no_grad():
        for batch in sample_loader:
            age_ids, input_ids, posi_ids, segment_ids, att_mask, targets, patids = batch
            batch_size = input_ids.shape[0]
            input_ids_dev = input_ids.to(device)
            age_ids_dev = age_ids.to(device)
            posi_ids_dev = posi_ids.to(device)
            segment_ids_dev = segment_ids.to(device)
            att_mask_dev = att_mask.to(device)

            _ = model(input_ids_dev, age_ids_dev, segment_ids_dev, posi_ids_dev, attention_mask=att_mask_dev)

            target_ml = targets_to_multihot(targets, mlb).numpy().astype(bool)  # [batch, num_labels]

            # Last layer, head-averaged attention FROM the CLS token (position 0)
            # TO every position: shape [batch, heads, seq, seq] -> [batch, seq]
            last_layer_attn = attention_modules[-1].saved_attention_probs  # cpu tensor
            cls_attn = last_layer_attn[:, :, 0, :].mean(dim=1)  # [batch, seq_len], averaged over heads

            input_ids_np = input_ids.numpy()
            att_mask_np = att_mask.numpy()
            cls_attn_np = cls_attn.numpy()

            for b in range(batch_size):
                row_idx = row_offset + b
                seq_ids = input_ids_np[b]
                mask_row = att_mask_np[b].astype(bool)
                attn_row = cls_attn_np[b]

                valid_len = int(mask_row.sum())
                if valid_len <= 1:
                    continue

                tokens = [idx_to_token.get(int(t), "UNK") for t in seq_ids[:valid_len]]
                attn_valid = attn_row[:valid_len]
                attn_valid = attn_valid / (attn_valid.sum() + 1e-12)  # renormalize over valid (non-pad) positions

                # position 0 is CLS itself; distance-from-end buckets computed
                # over positions 1..valid_len-1 (the actual visit-history tokens).
                for pos in range(1, valid_len):
                    distance_from_end = (valid_len - 1) - pos  # 0 = last token before padding
                    if distance_from_end < max_distance_bucket:
                        recency_attn_sum[distance_from_end] += attn_valid[pos]
                        recency_attn_count[distance_from_end] += 1

                    token_str = tokens[pos]
                    label_idx = label_vocab.get(token_str)
                    if token_str == "SEP":
                        pass  # handled separately below
                    elif label_idx is not None:
                        if target_ml[b, label_idx]:
                            target_hit_attn_sum += attn_valid[pos]
                            target_hit_attn_count += 1
                        else:
                            target_miss_attn_sum += attn_valid[pos]
                            target_miss_attn_count += 1

                sep_mask = np.array([t == "SEP" for t in tokens[1:valid_len]])
                real_mask = ~sep_mask
                attn_no_cls = attn_valid[1:valid_len]
                total_mass = attn_no_cls.sum() + 1e-12
                sep_attn_mass.append(float(attn_no_cls[sep_mask].sum() / total_mass) if sep_mask.any() else 0.0)
                real_token_attn_mass.append(float(attn_no_cls[real_mask].sum() / total_mass) if real_mask.any() else 0.0)

                if len(case_studies) < N_CASE_STUDIES and valid_len > 5:
                    case_studies.append(
                        {
                            "patid": int(patids[b].item()),
                            "sample_row_index": row_idx,
                            "sequence_length": valid_len,
                            "tokens": tokens,
                            "cls_attention_last_layer_head_avg": [float(x) for x in attn_valid],
                        }
                    )

            row_offset += batch_size

    restore_attention_forward()

    valid_buckets = recency_attn_count > 0
    recency_curve = np.divide(
        recency_attn_sum, recency_attn_count, out=np.full_like(recency_attn_sum, np.nan), where=valid_buckets
    )
    # Trim to the largest distance actually observed (most rows are much shorter
    # than max_len_seq, so far buckets will mostly be empty/NaN).
    max_observed = int(np.max(np.nonzero(valid_buckets)[0])) if valid_buckets.any() else 0
    recency_curve = recency_curve[: max_observed + 1]

    mean_target_hit_attn = (
        float(target_hit_attn_sum / target_hit_attn_count) if target_hit_attn_count > 0 else float("nan")
    )
    mean_target_miss_attn = (
        float(target_miss_attn_sum / target_miss_attn_count) if target_miss_attn_count > 0 else float("nan")
    )

    print(f"\nSample size: {sample_size} patients")
    print(
        f"Mean CLS attention on history positions whose code IS a true next-visit "
        f"target: {mean_target_hit_attn:.6f} (n={int(target_hit_attn_count)})"
    )
    print(
        f"Mean CLS attention on history positions whose code is NOT a true "
        f"next-visit target: {mean_target_miss_attn:.6f} (n={int(target_miss_attn_count)})"
    )
    print(f"Mean fraction of attention mass on SEP tokens:  {np.mean(sep_attn_mass):.4f}")
    print(f"Mean fraction of attention mass on real codes:  {np.mean(real_token_attn_mass):.4f}")
    print(f"\nAttention vs. recency (distance from end of sequence, 0=most recent):")
    for d in range(min(20, len(recency_curve))):
        if not np.isnan(recency_curve[d]):
            print(f"  distance={d:>3}  mean_cls_attn={recency_curve[d]:.6f}")

    # --- Code embedding nearest neighbors (no forward pass needed) ---
    word_embeddings = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()  # [vocab, hidden]
    token2idx = bert_vocab["token2idx"]

    def nearest_neighbors(code, k=TOP_K_NEIGHBORS):
        if code not in token2idx:
            return None
        idx = token2idx[code]
        vec = word_embeddings[idx]
        norms = np.linalg.norm(word_embeddings, axis=1) * np.linalg.norm(vec) + 1e-12
        sims = (word_embeddings @ vec) / norms
        order = np.argsort(-sims)
        neighbors = []
        for j in order:
            if j == idx:
                continue
            tok = idx_to_token.get(int(j), "UNK")
            if tok in {"PAD", "SEP", "CLS", "MASK", "UNK"}:
                continue
            neighbors.append({"code": tok, "cosine_similarity": float(sims[j])})
            if len(neighbors) >= k:
                break
        return neighbors

    embedding_neighbors = {}
    print("\nCode embedding nearest neighbors:")
    for code in EXAMPLE_CODES_FOR_NEIGHBORS:
        neighbors = nearest_neighbors(code)
        embedding_neighbors[code] = neighbors
        if neighbors:
            print(f"  {code}:")
            for n in neighbors[:5]:
                print(f"    {n['code']}  (cos_sim={n['cosine_similarity']:.4f})")
        else:
            print(f"  {code}: not found in vocab")

    result = {
        "run_dir": str(run_dir.relative_to(project_root)),
        "sample_size": sample_size,
        "num_layers": num_layers,
        "attention_vs_true_next_visit_target": {
            "description": (
                "Compares mean CLS attention on history positions whose code "
                "IS one of the true next-visit target labels (target_hit) vs "
                "positions whose code is NOT a target label (target_miss)."
            ),
            "mean_cls_attention_target_hit": mean_target_hit_attn,
            "mean_cls_attention_target_miss": mean_target_miss_attn,
            "n_target_hit_positions": int(target_hit_attn_count),
            "n_target_miss_positions": int(target_miss_attn_count),
        },
        "attention_vs_recency": {
            "distance_from_end": list(range(len(recency_curve))),
            "mean_cls_attention": [None if np.isnan(v) else float(v) for v in recency_curve],
        },
        "sep_token_attention": {
            "mean_fraction_of_mass_on_sep": float(np.mean(sep_attn_mass)),
            "mean_fraction_of_mass_on_real_codes": float(np.mean(real_token_attn_mass)),
        },
        "code_embedding_nearest_neighbors": embedding_neighbors,
        "case_studies": case_studies,
        "caveat": (
            "Raw attention weights are descriptive, not a validated causal "
            "explanation (Jain & Wallace, 2019). See ../deep_interpretability/ "
            "for attribution-based (Integrated Gradients) follow-up."
        ),
    }

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "attention_and_embeddings.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
