#!/usr/bin/env python3
"""
Deep Interpretability: Attention Rollout (Abnar & Zuidema, 2020).

This is Item #2 from BEHRT_mechanism_interpretability/deep_interpretability/README.md
("cheap upgrade over raw attention, reuses existing capture code"). It directly
tests the open hypothesis noted after building
../simple_interpretability/attention_and_embeddings.py: that final-layer CLS
attention shows only mild recency bias (peaks at distance=1, decays by
distance=19) even though longer total history keeps improving performance
(eda/08_disease_level... / Step 10), implying distant-visit information is
aggregated via EARLIER layers / the residual stream rather than surfaced
directly in the last layer's raw attention.

Raw last-layer attention ignores each layer's residual connection
(output = attention(x) + x), so a token's final representation mixes
attention-routed information with information passed straight through from
earlier layers. Attention rollout accounts for this by recursively
multiplying (attention + identity) matrices across all layers, giving a
token-to-token "effective influence" estimate that is more faithful than any
single layer's raw attention.

Rollout formula per layer l (head-averaged attention A_l, shape [seq, seq]):
    A_hat_l = 0.5 * A_l + 0.5 * I        (accounts for the residual skip path)
    R_l = A_hat_l @ R_(l-1)              (R_0 = I)
The CLS row of the final R_L estimates, for the same final classification
readout used everywhere else in this project, how much each input token
ultimately contributes across all 6 layers -- not just the last one.

Reuses the same "load run from metrics.json -> rebuild vocab/loader -> load
checkpoint -> recompute predictions" pattern and the attention-capture
monkey-patch from ../simple_interpretability/attention_and_embeddings.py
(imported directly, no duplication).

Caveat (state explicitly in any writeup): rollout is still an attention-based
heuristic (not a validated causal attribution like Integrated Gradients) --
it corrects for the residual stream but still assumes attention weights
straightforwardly compose across layers, which is itself a simplifying
assumption (see Abnar & Zuidema's own caveats, and later critiques such as
"attention flow" being more rigorous but far more expensive).

Usage:
    RUN_DIR=data/models/clean_runs/clean_run_20260910_124255 \
        python3.12 BEHRT_mechanism_interpretability/deep_interpretability/attention_rollout.py
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

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "BEHRT_main" / "BEHRT_thesis" / "BEHRT_Project"
SIMPLE_INTERP_DIR = REPO_ROOT / "BEHRT_mechanism_interpretability" / "simple_interpretability"
for path in (PROJECT_ROOT, SIMPLE_INTERP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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

# Reuse the attention-capture monkey-patch + run-loading helper directly from
# the simple_interpretability script rather than duplicating it.
from attention_and_embeddings import (  # noqa: E402
    find_latest_run_dir,
    install_attention_capture,
    restore_attention_forward,
    get_layer_attention_modules,
)

N_SAMPLE_PATIENTS = 500  # same sample size as attention_and_embeddings.py, for direct comparability


def compute_rollout_cls_row(layer_attn_probs_list):
    """layer_attn_probs_list: list of [batch, heads, seq, seq] tensors, one per
    layer, in layer order (layer 0 first). Returns the CLS row (index 0) of
    the final rollout matrix: [batch, seq] -- how much each input token
    ultimately contributes to the CLS representation after accounting for
    the residual stream across all layers.
    """
    batch_size, _, seq_len, _ = layer_attn_probs_list[0].shape
    rollout = torch.eye(seq_len, dtype=torch.float64).unsqueeze(0).expand(batch_size, -1, -1).clone()
    identity = torch.eye(seq_len, dtype=torch.float64).unsqueeze(0)

    for layer_attn in layer_attn_probs_list:
        head_avg = layer_attn.mean(dim=1).double()  # [batch, seq, seq]
        augmented = 0.5 * head_avg + 0.5 * identity  # residual-aware, rows still sum to 1
        rollout = torch.bmm(augmented, rollout)

    return rollout[:, 0, :]  # CLS row: [batch, seq]


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

    rng = np.random.default_rng(42)  # same seed as attention_and_embeddings.py -> same sampled patients
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

    max_distance_bucket = max_len_seq
    recency_attn_sum = np.zeros(max_distance_bucket, dtype=np.float64)
    recency_attn_count = np.zeros(max_distance_bucket, dtype=np.float64)

    target_hit_attn_sum = 0.0
    target_hit_attn_count = 0.0
    target_miss_attn_sum = 0.0
    target_miss_attn_count = 0.0

    sep_attn_mass = []
    real_token_attn_mass = []

    print("Running attention-capture forward passes over sample (computing rollout)...")
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

            target_ml = targets_to_multihot(targets, mlb).numpy().astype(bool)

            layer_attn_probs_list = [m.saved_attention_probs for m in attention_modules]  # each [batch, heads, seq, seq], cpu
            cls_rollout = compute_rollout_cls_row(layer_attn_probs_list).numpy()  # [batch, seq]

            input_ids_np = input_ids.numpy()
            att_mask_np = att_mask.numpy()

            for b in range(batch_size):
                row_idx = row_offset + b
                seq_ids = input_ids_np[b]
                mask_row = att_mask_np[b].astype(bool)
                attn_row = cls_rollout[b]

                valid_len = int(mask_row.sum())
                if valid_len <= 1:
                    continue

                tokens = [idx_to_token.get(int(t), "UNK") for t in seq_ids[:valid_len]]
                attn_valid = attn_row[:valid_len]
                attn_valid = attn_valid / (attn_valid.sum() + 1e-12)

                for pos in range(1, valid_len):
                    distance_from_end = (valid_len - 1) - pos
                    if distance_from_end < max_distance_bucket:
                        recency_attn_sum[distance_from_end] += attn_valid[pos]
                        recency_attn_count[distance_from_end] += 1

                    token_str = tokens[pos]
                    label_idx = label_vocab.get(token_str)
                    if token_str == "SEP":
                        pass
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

            row_offset += batch_size

    restore_attention_forward()

    valid_buckets = recency_attn_count > 0
    recency_curve = np.divide(
        recency_attn_sum, recency_attn_count, out=np.full_like(recency_attn_sum, np.nan), where=valid_buckets
    )
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
        f"[ROLLOUT] Mean CLS attention on history positions whose code IS a true next-visit "
        f"target: {mean_target_hit_attn:.6f} (n={int(target_hit_attn_count)})"
    )
    print(
        f"[ROLLOUT] Mean CLS attention on history positions whose code is NOT a true "
        f"next-visit target: {mean_target_miss_attn:.6f} (n={int(target_miss_attn_count)})"
    )
    print(f"[ROLLOUT] Mean fraction of attention mass on SEP tokens:  {np.mean(sep_attn_mass):.4f}")
    print(f"[ROLLOUT] Mean fraction of attention mass on real codes:  {np.mean(real_token_attn_mass):.4f}")
    print(f"\n[ROLLOUT] Attention vs. recency (distance from end of sequence, 0=most recent):")
    for d in range(min(20, len(recency_curve))):
        if not np.isnan(recency_curve[d]):
            print(f"  distance={d:>3}  mean_rollout_cls_attn={recency_curve[d]:.6f}")

    # --- Compare against the raw last-layer attention already computed by
    # attention_and_embeddings.py, if present, to directly test the "distant
    # info aggregated via earlier layers/residual stream" hypothesis: rollout
    # should show a FLATTER recency curve (less concentrated on the most
    # recent positions) than raw last-layer attention if that hypothesis
    # holds.
    raw_results_path = SIMPLE_INTERP_DIR / "results" / "attention_and_embeddings.json"
    comparison = None
    if raw_results_path.exists():
        with open(raw_results_path, "r", encoding="utf-8") as f:
            raw_results = json.load(f)
        raw_recency = raw_results.get("attention_vs_recency", {}).get("mean_cls_attention", [])
        raw_recency = [v for v in raw_recency if v is not None]
        rollout_recency = [v for v in recency_curve if not np.isnan(v)]
        n_compare = min(len(raw_recency), len(rollout_recency), 20)
        if n_compare > 1:
            raw_arr = np.array(raw_recency[:n_compare])
            rollout_arr = np.array(rollout_recency[:n_compare])
            # Normalize each curve to sum to 1 over the compared range so we're
            # comparing SHAPE (concentration near distance=0) not overall scale.
            raw_norm = raw_arr / raw_arr.sum()
            rollout_norm = rollout_arr / rollout_arr.sum()
            # A concentration index: fraction of total (normalized) mass held by
            # the 3 most-recent positions (distance 0-2). Lower = flatter/more
            # long-range = more consistent with the "aggregated via earlier
            # layers" hypothesis.
            raw_recency_concentration = float(raw_norm[:3].sum())
            rollout_recency_concentration = float(rollout_norm[:3].sum())
            comparison = {
                "n_positions_compared": n_compare,
                "raw_last_layer_recency_concentration_top3": raw_recency_concentration,
                "rollout_recency_concentration_top3": rollout_recency_concentration,
                "rollout_flatter_than_raw": rollout_recency_concentration < raw_recency_concentration,
            }
            print(
                f"\n[COMPARISON] Fraction of recency-curve mass in 3 most-recent positions: "
                f"raw last-layer={raw_recency_concentration:.4f} vs rollout={rollout_recency_concentration:.4f} "
                f"(rollout flatter = {comparison['rollout_flatter_than_raw']})"
            )
    else:
        print(f"\n(No existing attention_and_embeddings.json found at {raw_results_path} -- skipping raw-vs-rollout comparison)")

    result = {
        "run_dir": str(run_dir.relative_to(project_root)),
        "sample_size": sample_size,
        "num_layers": num_layers,
        "method": "attention_rollout (Abnar & Zuidema, 2020): 0.5*A_l + 0.5*I per layer, recursively multiplied across all layers",
        "attention_vs_true_next_visit_target": {
            "description": (
                "Same comparison as ../simple_interpretability/attention_and_embeddings.py "
                "but using the ROLLOUT (all-layer, residual-aware) CLS attention instead of "
                "raw last-layer attention."
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
        "raw_vs_rollout_recency_comparison": comparison,
        "caveat": (
            "Attention rollout corrects for the residual stream across layers but is still "
            "an attention-based heuristic, not a validated causal attribution (unlike "
            "Integrated Gradients in ../deep_interpretability/integrated_gradients.py). "
            "It assumes attention composes roughly linearly across layers, which is itself "
            "a simplifying assumption (Abnar & Zuidema, 2020)."
        ),
    }

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "attention_rollout.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved attention rollout results to: {output_path}")


if __name__ == "__main__":
    main()
