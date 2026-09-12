# BEHRT Project TODO

## ✅ COMPLETED

### Data Quality & Cleaning
- [x] Discovered XXX000 generic catch-all code (8.05% frequency, 148K instances)
- [x] Created data cleaning script (`clean_data.py`)
- [x] Generated cleaned datasets: `*_clean.parquet` + `vocab_ccsr_clean.pkl`
- [x] Verified 99.98% data retention, removed problematic codes
- [x] `eda/07_cleaning_impact_analysis.py`: quantified cleaning's effect on
      labels (label vs label_original), cross-referenced against
      `clean_data.py`'s documented XXX/000/999/UNK removal strategy

### EDA (Steps 1-6)
- [x] `eda/01_dataset_structure.py` - dataset structure analysis
- [x] `eda/02_patient_level_analysis.py` - patient-level analysis
- [x] `eda/03_label_support_analysis.py` - label support/frequency tiers
- [x] `eda/04_temporal_structure_analysis.py` - temporal structure analysis
- [x] `eda/05_label_cooccurrence_analysis.py` - label co-occurrence analysis
- [x] `eda/06_split_representativeness_analysis.py` - train/val/test split checks

### Full Model Training (HIGH PRIORITY - DONE)
- [x] Trained full BEHRT model on cleaned data (`train_nextvisit_clean.py`)
- [x] Config: 6 layers, 288 hidden size, 12 heads, full ~143k train rows, 3 epochs
- [x] Latest full run: `data/models/clean_runs/clean_run_20260910_124255/`
      (P=0.205, R=0.283, F1=0.238 micro @ tuned threshold 0.60;
      sample-wise AUC=0.901, APS=0.274)

### Advanced Evaluation Analysis
- [x] NEW vs RECURRING diagnosis stratification
      (`compute_recurring_mask`/`compute_new_recurring_metrics` in
      `train_nextvisit_clean.py`): recurring F1=0.403/AUC=0.681 vs
      new F1=0.061/AUC=0.865 - model ranks new diagnoses well but the
      global threshold suppresses almost all of them
- [x] `scripts/threshold_sweep_new_recurring.py`: confirmed threshold
      miscalibration; shared threshold 0.50 beats tuned 0.60 for both subsets
- [x] `eda/08_disease_level_performance_breakdown.py`: joined per-class
      metrics with support tiers - mean AUC is NOT monotonic with support;
      139/469 labels have zero recall despite decent AUC in several tiers
- [x] `scripts/per_tier_threshold_calibration.py`: per-support-tier decision
      thresholds (validated on val set, scored on test set) - overall F1
      0.238->0.259, recall 0.284->0.379, zero-recall labels 330->305/469.
      Ultra-rare tier (mean test support ~0.43) stays at F1=0 regardless of
      threshold - a structural data-scarcity limit, not a calibration issue

---

## 🚀 NEXT STEPS

### 1. Error / Confusion Analysis (HIGH PRIORITY)
- [ ] Which specific CCSR codes get predicted for each other (false-positive
      co-occurrence/confusion pairs), not just aggregate precision/recall
- [ ] Identify clinically-related code confusions vs random noise

### 2. Sequence-Position / Visit-Count Effects
- [ ] Does performance vary with number of prior visits per patient
      (early vs. late in sequence)?

### 3. Interpretability
- [ ] Attention-weight inspection: what does the transformer attend to
      when predicting a given diagnosis?

### 4. Advanced Analysis (OPTIONAL)
- [ ] Disease category deep-dive (Circulatory, Endocrine focus)
- [ ] Rare/ultra-rare disease handling strategies (data augmentation,
      few-shot/transfer approaches - since thresholding alone can't fix
      the ultra-rare tier)
- [ ] Clinical validation with domain experts

### 5. Production Considerations (FUTURE)
- [ ] Model deployment pipeline
- [ ] Real-time prediction API
- [ ] Performance monitoring dashboard
- [ ] Clinical decision support integration

---

## 📁 KEY FILES

**Data:**
```
data/processed/
├── train_nextvisit_ccsr_clean.parquet
├── val_nextvisit_ccsr_clean.parquet
├── test_nextvisit_ccsr_clean.parquet
└── vocab_ccsr_clean.pkl
```

**Training / Evaluation Scripts:**
- `scripts/train_nextvisit_clean.py` - main training + evaluation entry point
- `scripts/threshold_sweep_new_recurring.py` - per-subset threshold sweep
- `scripts/per_tier_threshold_calibration.py` - per-support-tier calibration
- `eda/07_cleaning_impact_analysis.py`, `eda/08_disease_level_performance_breakdown.py`

**Latest Run:**
- `data/models/clean_runs/clean_run_20260910_124255/` (metrics.json,
  per_class_metrics.csv, new_recurring_threshold_sweep.json,
  per_tier_threshold_calibration.json)

---

## 💡 KEY INSIGHTS DISCOVERED

✅ **Data cleaning is critical**: XXX000 was masking true performance
✅ **Model ranks well but thresholds poorly**: AUC is decent-to-good across
   support tiers (even best for ultra-rare), but a single global threshold
   zeroes out recall for ~30% of labels
✅ **Per-tier threshold calibration is a free win**: no retraining needed,
   meaningfully improves recall/F1 with zero model changes
✅ **Ultra-rare labels are a data problem, not a threshold problem**: mean
   test support <1 per label means most never appear in the test set at all
✅ **NEW diagnoses rank well (AUC=0.865) but are threshold-starved**: because
   positive rate for "new" positions is ~150x sparser than "recurring"

---

## 🎯 IMMEDIATE NEXT ACTION

**Build error/confusion analysis**: identify which CCSR codes are most
frequently confused with each other (false positives correlated with true
labels), to explain *why* precision is low, not just that it is.