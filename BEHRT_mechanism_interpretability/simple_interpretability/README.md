# Simple Interpretability (Low Effort, Fast Turnaround)

**Goal:** Get a defensible, "what does the model look at" story with minimal
new infrastructure — mostly single forward-pass extraction, no gradients
w.r.t. inputs beyond a basic saliency pass, no new libraries required
beyond what's already installed (`torch`, `numpy`, `pandas`, `sklearn`).

Everything here reuses the existing "load run from `metrics.json` → rebuild
vocab/loader → load checkpoint → recompute predictions" pattern already
established in `eda/08`, `eda/09`, `eda/10`.

---

## 1. Raw attention extraction (monkey-patch capture)
- `pytorch_pretrained_bert`'s `BertSelfAttention.forward` computes
  `attention_probs` internally but never returns them.
- Monkey-patch `forward` at runtime to stash `attention_probs` on the module
  (`self.saved_attention_probs`) without altering model weights or output —
  no retraining needed.
- Gives `[batch, heads=12, seq_len, seq_len]` per layer (6 layers).
- **Caveat to state explicitly in the writeup:** raw attention is not a
  validated explanation (Jain & Wallace, 2019 — "Attention is not
  Explanation"). Treat this as descriptive, not causal.

## 2. CLS-token attention (last layer, head-averaged)
- Since classification reads from the final `CLS` representation, average
  CLS→token attention across heads in the last layer.
- Directly answers "which positions does the model's final decision draw
  from" for a sample of patients.

## 3. Attention vs. recency
- Plot mean CLS→token attention as a function of distance from sequence end.
- Directly complements the existing Step 10 finding (longer history helps)
  — check whether the model is recency-biased or genuinely uses long-range
  context.

## 4. Attention vs. recurring vs. new codes
- Reuse the existing recurring-code mask (`compute_recurring_mask` in
  `scripts/train_nextvisit_clean.py`).
- Check whether CLS attention is systematically higher on positions holding
  codes that recur later — ties into the NEW vs RECURRING finding.

## 5. SEP-token attention ("attention sink" check)
- Known phenomenon in BERT-family models: attention concentrates on
  separator/CLS tokens as aggregation points.
- Quantify what fraction of total attention mass lands on `SEP` tokens vs
  real diagnosis codes — a sanity/limitation check, not a positive finding.

## 6. Code embedding nearest-neighbors
- Pull the learned code-embedding matrix (`nn.Embedding` weights) straight
  off the trained model — no forward pass needed at all.
- For a handful of common CCSR codes, list top-10 nearest neighbors by
  cosine similarity. Check informally whether clinically related codes
  (e.g. same CCSR category, known comorbidities) cluster together.
- Cheapest possible interpretability artifact — can be done in a few
  minutes.

## 7. Occlusion / leave-one-out saliency
- For a sample of patients, remove one code position at a time (replace
  with PAD), rerun forward pass, measure change in predicted probability
  for the actually-predicted labels.
- Model-agnostic, no gradients required, trivial to implement given
  existing `collect_eval_arrays`-style plumbing.
- Rank each patient's codes by how much removing them shifts predictions —
  a simple per-example "importance" list.

## 8. Qualitative case studies
- Pick 5-10 representative test patients (mix of short/long history,
  high/low performing predictions from Step 10 buckets).
- For each: show code sequence, per-position CLS attention (heatmap-style
  text or simple matplotlib plot), predicted vs. actual next-visit codes.
- Good for a thesis figure; cheap to produce once #1/#2 exist.

---

## Effort estimate
Items 1-2 and 6 can be built and run in a single script
(`simple_interpretability/attention_and_embeddings.py`) reusing the existing
run-loading boilerplate. Items 3-5 are aggregations over the same captured
attention tensors. Item 7 is a separate lightweight script. Item 8 is a
plotting pass over outputs from 1-2 and 7.

## What NOT to claim
- Attention weights are descriptive, not causal or mechanistic.
- This tells you *what the model attends to*, not *why* or *whether that
  attention is actually used* to produce the output (that requires the
  causal methods in `../deep_interpretability/`).
