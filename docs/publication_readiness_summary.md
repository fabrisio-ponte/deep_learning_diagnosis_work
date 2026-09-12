# Publication Readiness Summary for the BEHRT-Style EHR Prediction Project
**Updated: August 29, 2026 - After comprehensive EDA analysis**

---

## 🔍 Key EDA Discoveries (August 2026)

**Dataset Characterization:**
- **179,713 samples** from **~25,000 unique patients** (~7 longitudinal snapshots each)
- **463 unique CCSR codes** used as prediction targets
- **Median patient:** 62 years old, 44 diagnoses in history, 9 codes at next visit
- **Population:** 88% adults 40+, 46% age 65+, 0% under age 18

**Critical Insight #1: This is recurrence prediction, not onset prediction**
- **70% of target codes are RECURRENCES** (already in patient history)
- **30% are truly new** diagnoses
- Model learns "which chronic conditions will be active next" not "which diseases will develop"

**Critical Insight #2: Code type heterogeneity**
- **85.6%** disease diagnoses
- **7.0%** symptoms (chest pain, abnormal labs, etc.)
- **4.0%** administrative/encounter codes (routine checkup, preventive care)
- **2.7%** injuries/trauma (unpredictable acute events)
- **0.7%** pregnancy and congenital conditions
- **14.4% of targets are not disease predictions**

**Critical Insight #3: Extreme class imbalance**
- **Most common code:** CCSR_END010 (diabetes) - 52,559 samples (36.6%)
- **Rarest codes:** 7 diseases with only 1 sample each
- **Imbalance ratio:** 52,559:1
- **133 rare diseases** (<100 samples, 29% of all diseases) have weak signal
- **Overall sparsity:** 48:1 (negative:positive ratio), only 2% of possible labels are positive

**Critical Insight #4: Performance dominated by common diseases**
- **Top 5 codes** (diabetes, hypertension, routine checkup, esophageal disorders, dysrhythmias) appear in 20-37% of samples
- **Bottom 50% of diseases** appear in <1% of samples each
- Model performance driven by chronic disease recurrence in sick, elderly patients

**Implication:** This reframes the entire thesis from "disease onset prediction" to **"next-visit clinical code recurrence prediction for multimorbid adults"**

---

## 1) What the project ACTUALLY is (after EDA findings)

This project is a BEHRT-style transformer for **longitudinal clinical code prediction** in multimorbid patients, trained on cleaned CCSR-coded sequences to predict which codes (diagnoses, symptoms, encounters) will appear at the next visit.

**Critical EDA findings that reframe the work:**
- **70% of target codes are RECURRENCES** (already in patient history), 30% are new
- This is **NOT "disease onset prediction"** — it's **"chronic disease recurrence and next-visit code prediction"**
- Dataset: **~25,000 unique patients**, each with ~7 longitudinal snapshots (179,713 total samples)
- Patient demographics: **median age 62 years, 88% adults 40+, 0% under 18**
- Code types: **85.6% diseases, 7% symptoms, 4% administrative, 2.7% injuries, 0.7% pregnancy/congenital**

The strongest framing is:

- a clinical sequence model for **next-visit clinical code prediction** in multimorbid adults
- trained on longitudinal EHR trajectories with **median 44 diagnoses per patient history**
- evaluated as a multilabel ranking problem with **extreme class imbalance** (52,559:1 ratio)
- predicting **which chronic conditions will be active/documented** at next encounter (70% recurrence + 30% new)
- using positive-class weighting to address rare code prediction

This is a valid and defensible applied ML / clinical informatics problem **when framed honestly**.

---

## 2) What the model can reasonably claim

The model can credibly claim that it learns **temporal patterns of clinical code recurrence** in multimorbid patients. In practical terms, it can:

- **predict which clinical codes will appear at next visit** (AUC 0.90, APS 0.27 with pos_weight)
- **identify which chronic conditions will likely recur** (70% of predictions)
- **detect some new diagnoses** (30% of predictions)
- **rank rare vs common codes** using positive-class weighting (improves APS by +7.6%, verified: 0.2543 -> 0.2735)
- **capture temporal persistence patterns** in chronic disease management
- **support care coordination** by predicting ~9 active conditions per next visit

**Clinical utility:** Helps allocate resources, schedule specialists, and coordinate care for complex multimorbid patients.

This is a meaningful contribution **when framed as clinical decision support for chronic disease management**, not as general disease screening.

---

## 3) Where the analogy breaks

This is the critical point to state clearly in any thesis or paper:

The model does not “understand medicine,” “know disease mechanisms,” or “reason clinically” in the way a physician does. It learns statistical associations in coded medical events.

The analogy breaks at several points:

**Conceptual limits:**
- **prediction ≠ understanding** — model predicts codes, not disease pathophysiology
- **correlation ≠ causality** — high AUC for diabetes recurrence doesn't explain why
- **recurrence ≠ onset** — 70% of predictions are chronic conditions already in history
- **code prediction ≠ disease prediction** — includes symptoms (7%), admin codes (4%), injuries (2.7%)

**Population limits:**
- **multimorbid adults only** — median age 62, median 44 prior diagnoses, 0% under 18
- **cannot generalize to healthy/young populations** — trained on sick, elderly patients
- **cannot predict acute events** — injuries are unpredictable, yet appear in targets

**Performance limits:**
- **dominated by common diseases** — top 5 diseases (diabetes, hypertension) drive most signal
- **133 rare diseases (<100 samples)** have insufficient data for reliable prediction
- **extreme imbalance (52,559:1 ratio)** means even pos_weight can't fully address rare codes
- **14.4% of targets are non-disease codes** (admin, symptoms, injuries)

This is not a weakness of the work; it is the **honest boundary** that must be documented to avoid overclaiming.

---

## 4) Key assumptions that underlie the work

The current project relies on several assumptions **validated or challenged by EDA**:

**Validated assumptions:**
- ✓ Prior diagnosis history contains signal (70% recurrence shows strong temporal persistence)
- ✓ Train/val/test splits are proper (no patient overlap confirmed)
- ✓ Distributions are consistent across splits (confirmed by EDA)
- ✓ Class weighting improves rare label learning (APS: 0.2543 → 0.2735, +7.6% — verified via `scripts/baseline_vs_posweight_comparison.py`, `results/baseline_vs_posweight_comparison.json`; supersedes the earlier unverified 0.262→0.274/+4.6% figure, which mixed runs from an older eval schema before UNK-label exclusion and threshold tuning were added)

**Assumptions requiring careful framing:**
- ⚠ **"Next-visit prediction" is primarily recurrence prediction** (70%), not onset (30%)
- ⚠ **CCSR labels include non-disease codes** (14.4%: symptoms, admin, injuries)
- ⚠ **Dataset represents multimorbid adults**, not general population (median age 62)
- ⚠ **Extreme class imbalance** (52,559:1) limits rare disease prediction fundamentally
- ⚠ **Model learns persistence patterns**, not causal disease progression

**Key dataset facts:**
- 179,713 samples from ~25,000 unique patients (~7 snapshots each)
- 463 unique CCSR codes: 85.6% diseases, 7% symptoms, 4% admin, 2.7% injuries, 0.7% other
- Median patient: 62 years, 44 diagnoses in history, 9 codes at next visit
- Top 5 diseases (diabetes, hypertension, etc.) appear in 20-37% of samples
- Bottom 133 diseases appear in <100 samples each (29% of all disease types)

These assumptions are reasonable **when limitations are stated clearly**, not as general disease prediction.

---

## 5) Main limitations that need to be faced honestly

The main limitations are not optional extras; they are central to the paper.

**Critical limitations discovered through EDA:**

1. **Task is recurrence prediction, not onset prediction**
   - 70% of target codes already in patient history
   - Cannot distinguish new disease from chronic disease flare-up without explicit modeling
   - Model learns "which conditions will be documented" not "what diseases will develop"

2. **Population is highly specific**
   - Median age 62 years, 88% adults 40+, 0% under age 18
   - Median 44 prior diagnoses (multimorbid, not healthy)
   - Median 9 next-visit codes (very sick patients)
   - **Cannot generalize to pediatric, young adult, or disease-free populations**

3. **Code type heterogeneity**
   - 14.4% of targets are NOT diseases: symptoms (7%), admin encounters (4%), injuries (2.7%)
   - Top-3 predictor is "routine checkup" (CCSR_FAC025, 28% of samples)
   - Performance metrics mix disease, symptom, and administrative code prediction

4. **Extreme class imbalance**
   - Most common disease: 52,559 samples (36.6%)
   - Rarest diseases: 1 sample each
   - Imbalance ratio: 52,559:1
   - 133 diseases (<100 samples, 29% of disease types) have weak predictive signal
   - Overall sparsity: 48:1 (negative:positive ratio)

5. **Performance dominated by common diseases**
   - Top 5 diseases (diabetes, hypertension, routine checkup, esophageal disorders, dysrhythmias) drive most signal
   - Rare diseases effectively unpredictable despite pos_weight
   - No causal interpretation of disease progression

6. **Acute events are unpredictable**
   - 2.7% of targets are injuries/trauma (falls, fractures)
   - These are random acute events, not learnable patterns
   - Inflates prediction difficulty artificially

**Standard limitations:**
- Modest data scale (~25k patients vs millions in foundation models)
- Limited to coded diagnoses, no clinical notes or labs
- Threshold sensitivity and calibration concerns
- Potential information loss from CCSR aggregation

These limitations are exactly what make a paper **credible when written honestly**.

---

## 6) What makes it publishable

This project is publishable if the thesis or paper is positioned as a focused clinical prediction study, not a broad medical intelligence claim.

To make it publishable, you need:

- a clean result story with a strong experimental comparison
- the same data, model, and training setup across the key ablation
- fair reporting of APS, AUC, F1, threshold tuning, and label coverage
- disease-level evaluation for clinically relevant conditions
- calibration and threshold analysis
- explicit limitations and failure-mode discussion
- honest framing that this is a predictive model, not a causal or mechanistic model

This is enough to support a strong applied ML or clinical informatics paper.

---

## 7) What is not worth doing right now

Do not spend much time on:

- deep mechanistic interpretability of the transformer
- trying to “explain the whole model” in biological terms
- broad architecture experimentation without a hypothesis
- a long list of random improvements that are not tied to a clear research question

Those directions are lower ROI for a thesis with limited time.

---

## 8) Best next research direction

The strongest next direction is not “more model complexity,” but better and more defensible evaluation and clinical relevance.

Priority order:

1. finalize the best current model and compare it rigorously against baseline
2. run disease-level performance analysis
3. analyze calibration and threshold sensitivity
4. study subgroup and rare-label behavior
5. write the limitations and scope section early
6. only then consider extra modeling improvements such as class-specific strategies, longer history windows, or a more tailored objective

This keeps the project clinically grounded and publication-focused.

---

## 9) The strongest thesis-level takeaway

A thesis-safe statement would be:

> **"This work demonstrates that a transformer-based model can learn temporal patterns of clinical code recurrence from longitudinal EHR sequences, achieving meaningful next-visit prediction performance (AUC 0.90, APS 0.27) for multimorbid adult patients. Positive-class weighting improves rare code detection by 7.6% APS over baseline (0.2543 -> 0.2735). However, the task is primarily chronic disease recurrence prediction (70% of targets) rather than new disease onset, performance is dominated by common conditions, and the model remains a statistical predictor of coded clinical events rather than a mechanistic account of disease progression. The work's practical value lies in supporting care coordination and resource allocation for complex patients, with clear limitations regarding population generalizability and interpretability."**

**Reframed research question:**
> "Can transformer models effectively predict which clinical codes (diagnoses, symptoms, encounters) will appear at a patient's next visit, learning from longitudinal EHR sequences with extreme class imbalance and chronic disease recurrence patterns?"

**Key contributions:**
1. **Methodological:** Demonstrates transformer effectiveness on longitudinal EHR with 52,559:1 class imbalance
2. **Technical:** Positive-class weighting improves rare code detection (capped at MAX_POS_WEIGHT=30)
3. **Empirical:** Dataset characterization: 70% recurrence, 30% new codes, 14.4% non-disease codes
4. **Clinical:** Supports next-visit care coordination for multimorbid patients (median 9 active conditions)

This is intellectually honest and academically strong.

---

## 10) Final judgment

Yes, this project **is publishable** when reframed correctly, but the route is:

**❌ DON'T claim:**
- "General disease prediction model"
- "Predicts new disease onset"
- "Learns disease mechanisms"
- "Generalizes to all populations"
- "Solves rare disease prediction"

**✓ DO claim:**
- "Next-visit clinical code prediction for multimorbid adults"
- "Learns temporal recurrence patterns (70%) with some new diagnosis detection (30%)"
- "Supports care coordination and resource allocation"
- "Demonstrates transformer effectiveness on longitudinal EHR with extreme imbalance"
- "Positive-class weighting improves rare code detection by 7.6% APS (verified: 0.2543 → 0.2735)"

**Clinical framing:**
This is **NOT** a disease screening tool for healthy populations.
This **IS** a clinical decision support tool for managing complex, multimorbid patients.

**Target venues:**
- Applied ML in healthcare (CHIL, ML4H)
- Clinical informatics (JAMIA, JBI)
- Medical AI with honest limitations (Nature Digital Medicine, npj Digital Medicine)

The honest framing makes it **stronger**, not weaker. Reviewers will appreciate the careful dataset analysis and transparent limitations.

---

## 11) Immediate next steps (updated after EDA)

**EDA work (in progress):**
- \u2713 Step 1: Dataset structure analysis (completed)
- \u2713 Step 2: Patient-level analysis (completed - discovered 70/30 recurrence split)
- \u2713 Step 3: Label support & class imbalance (completed - found 52,559:1 ratio, 14.4% non-disease codes)
- \u23f3 Step 4: Temporal structure analysis (next)
- \u23f3 Step 5: Label co-occurrence patterns
- \u23f3 Step 6: Split representativeness
- \u23f3 Step 7: Cleaning impact analysis

**Analysis priorities after EDA:**
1. ✓ **Comparison table:** Baseline vs pos_weight, same seed/data/architecture (`scripts/baseline_vs_posweight_comparison.py` → `results/baseline_vs_posweight_comparison.json`). Verified result: APS 0.2543 → 0.2735 (+7.6%), AUC 0.8911 → 0.9006 (+1.07%). Only seed=42 has a full run under the current eval schema for both arms, so this is a single-seed comparison; flagged as a limitation.
2. **Disease-level performance:** Break down by common vs rare, new vs recurrent
3. ✓ **Code-type analysis:** Separate performance on diseases vs symptoms vs admin codes (`eda/11_code_type_performance_breakdown.py` → `eda/results/11_code_type_performance_breakdown.json`). Confirms 85.7% disease / 7.1% symptom / 4.3% admin / 2.6% injury / 0.3% pregnancy-perinatal test-occurrence split, and shows injury codes have near-zero F1 (0.046) vs disease F1 (0.200), supporting the "acute events are unpredictable" limitation.
4. **NEW vs RECURRING performance:** Compare model effectiveness on 70% recurrent vs 30% new codes
5. **Calibration analysis:** Threshold sensitivity, per-class calibration

**Novel analysis opportunity discovered:**
- **Recurrence vs onset stratified evaluation** — this would be a unique contribution showing:
  - Model excels at predicting chronic disease recurrence (70% of task)
  - Weaker at predicting truly new diagnoses (30% of task)
  - Different optimal strategies for each (pos_weight helps new codes more)

**Documentation priorities:**
1. Update methods section with dataset characterization
2. Write limitations section FIRST (before adding experiments)
3. Frame results around care coordination utility, not general prediction
4. Add honest discussion of what 0.90 AUC means in this context

**What NOT to do:**
- Deep mechanistic interpretability (low ROI)
- Architecture experiments without hypothesis
- Trying to fix the 52,559:1 imbalance fundamentally (it's a data limit)
- Overclaiming about rare disease prediction

This keeps the project on a **publishable, thesis-constrained, honest path**.
