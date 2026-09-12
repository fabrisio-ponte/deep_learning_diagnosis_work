#!/usr/bin/env python3
"""
Baseline (no pos_weight) vs positive-class-weighted (pos_weight=30) comparison.

docs/publication_readiness_summary.md cites "Positive-class weighting improves
rare code detection (APS: 0.262 -> 0.274, +4.6%)" as a headline result, but no
script in the repo reproduces that specific number from the actual
clean_runs/ metrics -- this script fills that gap so the comparison used in
any thesis/paper table is regenerated directly from metrics.json files rather
than a possibly-stale hand-computed figure.

Matching criteria for a fair (data/seed/architecture-controlled) comparison:
  - same train/test parquet files
  - epochs == 3, sample_limit == 0 (full run, not a smoke-test subset)
  - same seed
  - only run_controls.use_pos_weight / max_pos_weight differ

Usage:
    python3.12 scripts/baseline_vs_posweight_comparison.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_full_runs(project_root):
    runs_root = project_root / "data" / "models" / "clean_runs"
    runs = []
    for run_dir in sorted(runs_root.glob("clean_run_*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rc = data.get("run_controls", {})
        training = data.get("training", {})
        run_data = data.get("data", {})
        metrics = data.get("metrics", {})

        # Only keep FULL runs: 3 epochs, no sample_limit (smoke-test subsampling).
        if training.get("epochs") != 3 or rc.get("sample_limit") not in (0, None):
            continue
        if rc.get("seed") is None:
            continue

        threshold_tuning = data.get("threshold_tuning") or {}

        runs.append(
            {
                "run_id": data.get("run_id", run_dir.name),
                "run_dir": str(run_dir.relative_to(project_root)),
                "seed": rc.get("seed"),
                "use_pos_weight": bool(rc.get("use_pos_weight")),
                "max_pos_weight": rc.get("max_pos_weight"),
                "train_file": run_data.get("train"),
                "test_file": run_data.get("test"),
                "top_k_labels": rc.get("top_k_labels"),
                "min_label_freq": rc.get("min_label_freq"),
                "metric_exclude_labels": rc.get("metric_exclude_labels"),
                "schema_version": data.get("schema_version"),
                "threshold_tuning_enabled": threshold_tuning.get("enabled"),
                "selected_threshold": threshold_tuning.get("selected_threshold"),
                "sample_wise_aps": metrics.get("sample_wise_aps"),
                "sample_wise_auc": metrics.get("sample_wise_auc"),
                "mtime": metrics_path.stat().st_mtime,
            }
        )
    return runs


# Only runs on the CURRENT evaluation schema (UNK excluded from metrics, and a
# tuned decision threshold) are eligible for the primary comparison. Older
# runs (schema_version unset) used a different, no-longer-current eval
# pipeline and produced inconsistent/anomalous metrics (see
# `excluded_legacy_schema_runs` in the output) -- mixing them in would silently
# compare apples to oranges.
CURRENT_SCHEMA_EXCLUDE_LABELS = ["UNK"]


def is_current_schema(run):
    return (
        run["metric_exclude_labels"] == CURRENT_SCHEMA_EXCLUDE_LABELS
        and run["schema_version"] is not None
        and run["threshold_tuning_enabled"] is True
    )


def main():
    project_root = PROJECT_ROOT
    all_runs = load_full_runs(project_root)
    print(f"Found {len(all_runs)} full runs (epochs=3, sample_limit=0, seed set).")

    legacy_runs = [r for r in all_runs if not is_current_schema(r)]
    runs = [r for r in all_runs if is_current_schema(r)]
    print(
        f"  {len(runs)} use the CURRENT eval schema (metric_exclude_labels=['UNK'], "
        f"tuned threshold) -- these are used for the comparison below.\n"
        f"  {len(legacy_runs)} use an older/legacy eval schema and are EXCLUDED from the "
        f"comparison (listed in 'excluded_legacy_schema_runs' in the output JSON) because "
        f"they are not eval-config-comparable (e.g. some produced anomalous/duplicate "
        f"APS values around 0.33-0.40 vs the ~0.25-0.27 seen consistently under the current schema)."
    )

    # Group by (seed, use_pos_weight, max_pos_weight, train_file, test_file,
    # top_k_labels, min_label_freq) -- runs with identical config that were
    # re-executed should produce (near-)identical metrics; this doubles as a
    # reproducibility check.
    def config_key(r):
        return (
            r["seed"],
            r["use_pos_weight"],
            r["max_pos_weight"],
            r["train_file"],
            r["test_file"],
            r["top_k_labels"],
            r["min_label_freq"],
        )

    groups = {}
    for r in runs:
        groups.setdefault(config_key(r), []).append(r)

    print("\nDistinct configurations found (seed, use_pos_weight, max_pos_weight, n_runs, APS values):")
    for key, group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        seed, use_pw, max_pw = key[0], key[1], key[2]
        aps_values = [g["sample_wise_aps"] for g in group]
        print(f"  seed={seed} use_pos_weight={use_pw} max_pos_weight={max_pw}  n_runs={len(group)}  APS={aps_values}")

    # Pick the most RECENT run for each (seed, use_pos_weight, max_pos_weight,
    # train_file, test_file) config as the representative value.
    representative = {}
    for key, group in groups.items():
        representative[key] = max(group, key=lambda r: r["mtime"])

    # Baseline configs: use_pos_weight == False.
    baseline_runs = {k: v for k, v in representative.items() if k[1] is False}
    # Pos-weight=30 configs: use_pos_weight == True and max_pos_weight == 30.
    posweight30_runs = {k: v for k, v in representative.items() if k[1] is True and k[2] == 30.0}

    # Seed-42 matched pair (same train/test files, same label-vocab controls)
    # is the primary, most tightly controlled comparison since it's the only
    # seed with both a baseline AND a pos_weight=30 full run available.
    matched_pairs = []
    for b_key, b_run in baseline_runs.items():
        b_seed, _, _, b_train, b_test, b_topk, b_minfreq = b_key
        for p_key, p_run in posweight30_runs.items():
            p_seed, _, _, p_train, p_test, p_topk, p_minfreq = p_key
            if (b_seed, b_train, b_test, b_topk, b_minfreq) == (p_seed, p_train, p_test, p_topk, p_minfreq):
                matched_pairs.append((b_run, p_run))

    print("\n" + "=" * 100)
    print("Matched baseline vs pos_weight=30 pairs (same seed/data/label-vocab, only pos_weight differs):")
    print("-" * 100)
    comparison_table = []
    for b_run, p_run in matched_pairs:
        aps_delta = p_run["sample_wise_aps"] - b_run["sample_wise_aps"]
        auc_delta = p_run["sample_wise_auc"] - b_run["sample_wise_auc"]
        aps_pct = aps_delta / b_run["sample_wise_aps"] * 100 if b_run["sample_wise_aps"] else float("nan")
        auc_pct = auc_delta / b_run["sample_wise_auc"] * 100 if b_run["sample_wise_auc"] else float("nan")
        row = {
            "seed": b_run["seed"],
            "baseline_run_id": b_run["run_id"],
            "baseline_aps": b_run["sample_wise_aps"],
            "baseline_auc": b_run["sample_wise_auc"],
            "posweight30_run_id": p_run["run_id"],
            "posweight30_aps": p_run["sample_wise_aps"],
            "posweight30_auc": p_run["sample_wise_auc"],
            "aps_delta": aps_delta,
            "aps_pct_change": aps_pct,
            "auc_delta": auc_delta,
            "auc_pct_change": auc_pct,
        }
        comparison_table.append(row)
        print(
            f"seed={row['seed']}: baseline APS={row['baseline_aps']:.4f} AUC={row['baseline_auc']:.4f} "
            f"({b_run['run_id']})\n"
            f"           pos_weight=30 APS={row['posweight30_aps']:.4f} AUC={row['posweight30_auc']:.4f} "
            f"({p_run['run_id']})\n"
            f"           Delta: APS {row['aps_delta']:+.4f} ({row['aps_pct_change']:+.2f}%), "
            f"AUC {row['auc_delta']:+.4f} ({row['auc_pct_change']:+.2f}%)"
        )
    print("=" * 100)

    # Baseline run-to-run variance across seeds (context: how much of the
    # pos_weight effect could be noise vs a genuine seed-42-only comparison).
    baseline_by_seed = {k[0]: v for k, v in baseline_runs.items()}
    baseline_aps_values = [r["sample_wise_aps"] for r in baseline_by_seed.values()]
    baseline_auc_values = [r["sample_wise_auc"] for r in baseline_by_seed.values()]
    seed_variance = None
    if len(baseline_aps_values) >= 2:
        import statistics

        seed_variance = {
            "n_seeds": len(baseline_aps_values),
            "seeds": sorted(baseline_by_seed.keys()),
            "aps_mean": statistics.mean(baseline_aps_values),
            "aps_std": statistics.stdev(baseline_aps_values),
            "auc_mean": statistics.mean(baseline_auc_values),
            "auc_std": statistics.stdev(baseline_auc_values),
        }
        print(
            f"\nBaseline (no pos_weight) run-to-run variance across {seed_variance['n_seeds']} seeds "
            f"{seed_variance['seeds']}:\n"
            f"  APS mean={seed_variance['aps_mean']:.4f} std={seed_variance['aps_std']:.4f}\n"
            f"  AUC mean={seed_variance['auc_mean']:.4f} std={seed_variance['auc_std']:.4f}"
        )
        if matched_pairs:
            primary_delta = comparison_table[0]["aps_delta"]
            print(
                f"\n  Note: only seed=42 has a full pos_weight=30 run to compare against, so the "
                f"pos_weight effect above (APS {primary_delta:+.4f}) is a SINGLE-SEED comparison; "
                f"baseline seed-to-seed noise alone is std={seed_variance['aps_std']:.4f} APS, "
                f"i.e. the effect is {'LARGER' if abs(primary_delta) > seed_variance['aps_std'] else 'NOT clearly larger'} "
                f"than baseline run-to-run variance."
            )
    else:
        print("\n(Not enough baseline seeds to compute run-to-run variance.)")

    result = {
        "all_full_run_configs": [
            {
                "seed": key[0],
                "use_pos_weight": key[1],
                "max_pos_weight": key[2],
                "n_runs_with_this_config": len(group),
                "aps_values": [g["sample_wise_aps"] for g in group],
                "auc_values": [g["sample_wise_auc"] for g in group],
            }
            for key, group in groups.items()
        ],
        "matched_seed42_comparison": comparison_table,
        "baseline_seed_variance": seed_variance,
        "excluded_legacy_schema_runs": [
            {
                "run_id": r["run_id"],
                "seed": r["seed"],
                "use_pos_weight": r["use_pos_weight"],
                "sample_wise_aps": r["sample_wise_aps"],
                "sample_wise_auc": r["sample_wise_auc"],
                "schema_version": r["schema_version"],
                "metric_exclude_labels": r["metric_exclude_labels"],
            }
            for r in legacy_runs
        ],
        "caveat": (
            "Only seed=42 has both a baseline (use_pos_weight=False) and a pos_weight=30 "
            "full run (epochs=3, sample_limit=0) available under the CURRENT eval schema, "
            "so the pos_weight effect is a single-seed comparison, not averaged across seeds. "
            "Baseline run-to-run variance across seeds 42/43/44 is reported above for context "
            "(note: full pos_weight=30 runs only exist for seed=42, so this variance describes "
            "the baseline arm only). Runs using an older eval schema (no UNK exclusion, no "
            "tuned threshold) produced substantially different, sometimes duplicate-looking "
            "APS/AUC values and were excluded from this comparison as not eval-config-comparable; "
            "see 'excluded_legacy_schema_runs'."
        ),
    }

    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_vs_posweight_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved baseline-vs-posweight comparison to: {output_path}")


if __name__ == "__main__":
    main()
