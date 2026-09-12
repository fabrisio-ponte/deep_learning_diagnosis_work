#!/usr/bin/env python3
"""
Deep Interpretability: Integrated Gradients attribution of false-positive
predictions, validated against the observational confusion analysis.

This is Item #1 from BEHRT_mechanism_interpretability/deep_interpretability/README.md
("cheapest high-rigor win, directly upgrades the Step 9 error/confusion story
from correlational to attribution-based"). It reuses the same
"load run from metrics.json -> rebuild vocab/loader -> load checkpoint ->
recompute predictions" pattern established in eda/08, eda/09, eda/10 and in
../simple_interpretability/attention_and_embeddings.py.

For the top false-positive-offending labels identified by
eda/09_error_confusion_analysis.py, this script:
  1. Finds test rows where that label is a false positive.
  2. Runs Captum Integrated Gradients on the model's WORD embeddings
     (age/segment/position embeddings held fixed at their real values,
     baseline = all-PAD word embedding) to attribute each input code
     position's contribution to that label's predicted logit.
  3. Aggregates positive attribution mass by CCSR category to get an
     attribution-derived "which category of comorbid codes is driving this
     false positive" ranking per label.
  4. Compares that ranking against eda/09's co-occurrence-lift-based
     confusion categories for the same label, to check whether the
     correlational finding (Step 9) is corroborated by a causal/attribution
     method.

Caveat (state explicitly in any writeup): Integrated Gradients has formal
axioms (completeness, sensitivity) that raw attention lacks, so this is a
stronger causal-adjacent claim than ../simple_interpretability/. It still
only explains the TRAINED MODEL's learned function -- not ground-truth
clinical causality between diagnoses.

Usage:
    RUN_DIR=data/models/clean_runs/clean_run_20260910_124255 \
        python3.12 BEHRT_mechanism_interpretability/deep_interpretability/integrated_gradients.py
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "BEHRT_main" / "BEHRT_thesis" / "BEHRT_Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj  # noqa: E402
from model.utils import age_vocab  # noqa: E402
from dataLoader.NextXVisit import NextVisit  # noqa: E402
from model.NextXVisit import BertForMultiLabelPrediction  # noqa: E402

from scripts.train_nextvisit_clean import (  # noqa: E402
    BertConfig,
    build_label_subset,
    collect_eval_arrays,
    format_label_vocab,
    normalize_label_row,
)

TOP_N_TARGET_LABELS = 15  # top FP-offending labels to analyze (by fp_count)
MIN_FP_COUNT = 10  # same threshold used in eda/09_error_confusion_analysis.py
N_ROWS_PER_LABEL = 15  # false-positive test rows sampled per target label
IG_N_STEPS = 32
IG_BATCH_SIZE = 5
TOP_N_EDA09_COMPARISON = 3  # how many top eda/09 confusion categories to compare against
SPECIAL_TOKENS = {"PAD", "SEP", "CLS", "MASK", "UNK"}


def find_latest_run_dir(project_root):
    runs_root = project_root / "data" / "models" / "clean_runs"
    candidates = [d for d in runs_root.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No clean_run_* directories with metrics.json found under {runs_root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def label_category(label):
    """CCSR_END010 -> END. Falls back to the whole label if pattern doesn't match."""
    match = re.match(r"^CCSR_([A-Z]+)\d+$", label)
    return match.group(1) if match else label


def build_embedding_attribution_forward(model):
    """Returns a forward function taking WORD EMBEDDINGS (instead of input_ids)
    as the first argument, with age/segment/position ids and attention_mask
    passed as additional (non-attributed) forward args. Replicates
    BertModel.forward + BertForMultiLabelPrediction.forward without altering
    the underlying model code, so it can be handed straight to Captum's
    IntegratedGradients.
    """
    embeddings_module = model.bert.embeddings
    encoder = model.bert.encoder
    pooler = model.bert.pooler
    classifier = model.classifier
    dropout = model.dropout
    model_dtype = next(model.parameters()).dtype

    def forward_from_word_embeds(word_embeds, age_ids, seg_ids, posi_ids, attention_mask):
        segment_embed = embeddings_module.segment_embeddings(seg_ids)
        age_embed = embeddings_module.age_embeddings(age_ids)
        posi_embed = embeddings_module.posi_embeddings(posi_ids)

        embeddings = word_embeds + segment_embed + age_embed + posi_embed
        embeddings = embeddings_module.LayerNorm(embeddings)
        embeddings = embeddings_module.dropout(embeddings)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=model_dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = encoder(embeddings, extended_attention_mask, output_all_encoded_layers=False)
        sequence_output = encoded_layers[-1]
        pooled_output = pooler(sequence_output)
        pooled_output = dropout(pooled_output)
        logits = classifier(pooled_output)
        return logits

    return forward_from_word_embeds


def compute_category_baseline_shares(df):
    """Fraction of all HISTORY code positions (the `code` column, i.e. the
    model's input tokens) belonging to each CCSR category, across the given
    dataframe. Used to normalize IG attribution mass by category -- without
    this, categories that are simply more common overall (e.g. CIR, END)
    dominate every label's "top attribution" ranking regardless of any
    target-specific relationship (confirmed empirically: the most frequent
    input categories are exactly the ones that kept appearing at the top for
    almost every target label before this normalization was added).
    """
    counts = {}
    total = 0
    for history_codes in df["code"]:
        for code in normalize_label_row(history_codes):
            if code in SPECIAL_TOKENS:
                continue
            cat = label_category(code)
            counts[cat] = counts.get(cat, 0) + 1
            total += 1
    return {cat: count / total for cat, count in counts.items()}


def load_eda09_confusions(project_root):
    path = project_root / "eda" / "results" / "09_error_confusion_analysis.json"
    if not path.exists():
        print(f"  (eda/09 results not found at {path} -- skipping comparison)")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("label_confusions", {})


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
    threshold = float(run_metrics["threshold_tuning"]["selected_threshold"])

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
    idx_to_label = {idx: label for label, idx in label_vocab.items()}
    idx_to_token = {idx: tok for tok, idx in bert_vocab["token2idx"].items()}

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
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # IG only needs gradients w.r.t. the input embeddings

    print("Recomputing test-set predictions (to find false positives per label)...")
    y_true, y_prob = collect_eval_arrays(model, test_loader, mlb, device)
    y_pred = y_prob >= threshold
    y_true_bool = y_true.astype(bool)
    num_labels = y_true.shape[1]

    print(f"Threshold used for predictions: {threshold:.2f}")

    fp_counts = (y_pred & ~y_true_bool).sum(axis=0)
    candidate_indices = [
        idx
        for idx in range(num_labels)
        if idx_to_label[idx] not in SPECIAL_TOKENS and fp_counts[idx] >= MIN_FP_COUNT
    ]
    candidate_indices.sort(key=lambda idx: fp_counts[idx], reverse=True)
    target_indices = candidate_indices[:TOP_N_TARGET_LABELS]

    print(
        f"Analyzing {len(target_indices)} labels (of {len(candidate_indices)} with "
        f">= {MIN_FP_COUNT} false positives) via Integrated Gradients..."
    )

    eda09_confusions = load_eda09_confusions(project_root)
    category_baseline_share = compute_category_baseline_shares(train_df)

    forward_fn = build_embedding_attribution_forward(model)
    ig = IntegratedGradients(forward_fn)

    word_embedding_table = model.bert.embeddings.word_embeddings
    pad_idx = bert_vocab["token2idx"]["PAD"]

    rng = np.random.default_rng(42)
    results_by_label = []

    for l_idx in target_indices:
        l_str = idx_to_label[l_idx]
        l_category = label_category(l_str)

        fp_row_indices = np.where(y_pred[:, l_idx] & ~y_true_bool[:, l_idx])[0]
        n_sample = min(N_ROWS_PER_LABEL, len(fp_row_indices))
        sampled_rows = rng.choice(fp_row_indices, size=n_sample, replace=False)
        sampled_rows.sort()

        row_df = test_df.iloc[sampled_rows].reset_index(drop=True)
        row_set = NextVisit(
            token2idx=bert_vocab["token2idx"],
            label2idx=label_vocab,
            age2idx=age_vocab_dict,
            dataframe=row_df,
            max_len=max_len_seq,
        )
        row_loader = DataLoader(row_set, batch_size=IG_BATCH_SIZE, shuffle=False, num_workers=0)

        category_attr_pos = {}
        total_pos_attr = 0.0
        n_positions_analyzed = 0

        for batch in row_loader:
            age_ids, input_ids, posi_ids, segment_ids, att_mask, _, _ = batch
            input_ids_dev = input_ids.to(device)
            age_ids_dev = age_ids.to(device)
            posi_ids_dev = posi_ids.to(device)
            segment_ids_dev = segment_ids.to(device)
            att_mask_dev = att_mask.to(device)

            word_embeds = word_embedding_table(input_ids_dev).detach()
            baseline_ids = torch.full_like(input_ids_dev, pad_idx)
            baseline_embeds = word_embedding_table(baseline_ids).detach()

            attributions = ig.attribute(
                inputs=word_embeds,
                baselines=baseline_embeds,
                target=l_idx,
                additional_forward_args=(age_ids_dev, segment_ids_dev, posi_ids_dev, att_mask_dev),
                n_steps=IG_N_STEPS,
            )
            token_attr = attributions.sum(dim=-1).cpu().numpy()  # [batch, seq_len], signed

            input_ids_np = input_ids.numpy()
            att_mask_np = att_mask.numpy()

            for b in range(input_ids_np.shape[0]):
                mask_row = att_mask_np[b].astype(bool)
                valid_len = int(mask_row.sum())
                for pos in range(1, valid_len):  # skip CLS at position 0
                    token_str = idx_to_token.get(int(input_ids_np[b, pos]), "UNK")
                    if token_str in SPECIAL_TOKENS:
                        continue
                    attr_val = float(token_attr[b, pos])
                    if attr_val <= 0:
                        continue  # only positive attribution: pushes the FP logit UP
                    cat = label_category(token_str)
                    category_attr_pos[cat] = category_attr_pos.get(cat, 0.0) + attr_val
                    total_pos_attr += attr_val
                    n_positions_analyzed += 1

        if total_pos_attr <= 0:
            continue

        category_fractions = {
            cat: val / total_pos_attr for cat, val in sorted(category_attr_pos.items(), key=lambda kv: -kv[1])
        }
        # Normalize by each category's baseline share of all history-code
        # positions (a lift, analogous to eda/09's cooccur-rate/baseline-rate
        # lift) so that categories which are simply common overall (CIR, END,
        # ...) don't dominate every label's ranking by default.
        category_lift = {
            cat: frac / category_baseline_share.get(cat, frac) for cat, frac in category_fractions.items()
        }
        other_category_ranking = sorted(
            (c for c in category_lift if c != l_category), key=lambda c: -category_lift[c]
        )[:5]

        eda09_entry = eda09_confusions.get(l_str)
        eda09_top_categories = []
        if eda09_entry:
            for c in eda09_entry.get("top_confusions", []):
                cat = label_category(c["confused_with_label"])
                if cat not in eda09_top_categories:
                    eda09_top_categories.append(cat)
        eda09_top_categories = eda09_top_categories[:TOP_N_EDA09_COMPARISON]

        agreement_top1_in_eda09_top3 = (
            bool(other_category_ranking and other_category_ranking[0] in eda09_top_categories)
            if eda09_top_categories
            else None
        )

        results_by_label.append(
            {
                "label": l_str,
                "category": l_category,
                "fp_count_total": int(fp_counts[l_idx]),
                "n_fp_rows_analyzed": n_sample,
                "n_positions_analyzed": n_positions_analyzed,
                "category_attribution_fraction": category_fractions,
                "category_attribution_lift": category_lift,
                "top_attribution_categories_excluding_own": other_category_ranking,
                "eda09_top_confusion_categories": eda09_top_categories,
                "agreement_top1_in_eda09_top3": agreement_top1_in_eda09_top3,
            }
        )

        print(f"\n{l_str:<16} (category={l_category}, fp_count={int(fp_counts[l_idx])}, rows_analyzed={n_sample})")
        print(f"  IG top attribution categories (excl. own, lift-ranked): {other_category_ranking}")
        print(f"  eda/09 top confusion categories:           {eda09_top_categories}")
        print(f"  agreement (IG #1 in eda/09 top-3):         {agreement_top1_in_eda09_top3}")

    agreements = [r["agreement_top1_in_eda09_top3"] for r in results_by_label if r["agreement_top1_in_eda09_top3"] is not None]
    overall_agreement_rate = float(np.mean(agreements)) if agreements else None

    print("\n" + "=" * 100)
    if overall_agreement_rate is not None:
        print(
            f"Overall: IG's #1 attributed category matched eda/09's top-{TOP_N_EDA09_COMPARISON} "
            f"co-occurrence-lift confusion categories for {sum(agreements)}/{len(agreements)} labels "
            f"({overall_agreement_rate:.1%})"
        )
    else:
        print("No eda/09 comparison data available (run eda/09_error_confusion_analysis.py first).")

    result = {
        "run_dir": str(run_dir.relative_to(project_root)),
        "threshold": threshold,
        "top_n_target_labels": TOP_N_TARGET_LABELS,
        "min_fp_count": MIN_FP_COUNT,
        "n_rows_per_label": N_ROWS_PER_LABEL,
        "ig_n_steps": IG_N_STEPS,
        "num_labels_analyzed": len(results_by_label),
        "overall_agreement_rate_top1_in_eda09_top3": overall_agreement_rate,
        "category_baseline_share": category_baseline_share,
        "label_results": results_by_label,
        "caveat": (
            "Integrated Gradients satisfies completeness/sensitivity axioms, giving a "
            "stronger causal-adjacent attribution than raw attention "
            "(../simple_interpretability/). It still only explains the TRAINED MODEL's "
            "learned function, not ground-truth clinical causality between diagnoses. "
            "Raw positive-attribution mass by category is confounded by each category's "
            "overall base rate among input codes (the most frequent input categories -- "
            "CIR, END, DIG, MBD, GEN -- dominated every label's raw ranking before this "
            "was corrected); category_attribution_lift divides by category_baseline_share "
            "to control for this, analogous to eda/09's cooccur-rate/baseline-rate lift. "
            "Comparisons against eda/09 use the lift-ranked categories. "
            "Baseline = all-PAD word embedding; age/segment/position embeddings held at "
            "their real values so only each code's identity is attributed."
        ),
    }

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "integrated_gradients_confusion_validation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved Integrated Gradients confusion-validation results to: {output_path}")


if __name__ == "__main__":
    main()
