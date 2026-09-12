# Deep Interpretability (Rigorous / Research-Grade)

**Goal:** Methods with stronger theoretical or causal grounding than raw
attention — the kind of evidence a reviewer would ask for if a claim like
"the model learned comorbidity relationships" appears in the thesis.
Higher effort, new dependencies, and (per
`../../docs/publication_readiness_summary.md` Section 7) explicitly
**lower ROI for the current thesis timeline** — this folder is a menu to
pick from selectively, not a full checklist to complete.

New dependency needed for most of these: `captum` (PyTorch's official
interpretability library — implements Integrated Gradients, Layer
Conductance, and more). Not currently in `requirements.txt`.

---

## 1. Integrated Gradients (Captum)
- Attribution method with formal axioms (completeness, sensitivity) that
  raw attention lacks.
- Attributes each input code's contribution to a specific predicted label's
  logit, integrated along a path from a baseline (all-PAD sequence) to the
  actual input.
- Directly comparable across patients/labels — enables aggregate
  "which code categories drive which predictions" statistics, not just
  single-example anecdotes.
- Moderate effort: `captum.attr.IntegratedGradients` wraps around the
  existing `BertForMultiLabelPrediction` model with the embedding layer as
  the attribution input.

## 2. Attention Rollout / Attention Flow
- Abnar & Zuidema (2020): aggregates attention across all 6 layers while
  accounting for residual connections (raw last-layer attention alone
  ignores how earlier layers already mixed information).
- "Rollout" = recursive matrix multiplication of per-layer attention (with
  identity added for the residual stream) — gives a token-to-token
  influence estimate more faithful than any single layer's raw attention.
- "Flow" variant treats attention as a max-flow problem on a token graph —
  more rigorous but noticeably more expensive to compute.
- Reuses the same captured `attention_probs` tensors from
  `../simple_interpretability/`.

## 3. Probing classifiers (representation analysis)
- Train small linear classifiers on frozen intermediate-layer hidden
  states to test whether specific clinical concepts are linearly decodable
  at each layer (e.g. "patient has a CIR-category code in history",
  "age bracket", "visit count bucket").
- Classic mechanistic-interpretability technique: reveals *what the model
  represents*, layer by layer, independent of what it attends to.
- Needs: hidden states per layer (already obtainable via
  `output_all_encoded_layers=True`, no monkey-patching required, unlike
  attention) + a simple `LogisticRegression`/small MLP probe per layer.

## 4. Head / layer ablation study
- Zero out (or mean-ablate) one attention head at a time, rerun the test
  set, measure the drop in micro-F1 / sample AUC.
- Ranks heads by causal importance to overall performance — much cheaper
  than full activation patching, and answers "does this head actually
  matter" rather than "what does it attend to."
- Fully reuses existing eval plumbing (`collect_eval_arrays`) with the
  monkey-patched attention module additionally zeroing one head's output
  before the value-weighted sum.

## 5. Causal mediation / activation patching
- Borrowed from mechanistic-interpretability circuit-analysis literature
  (e.g. transformer circuits work on GPT-style models).
- Run the model on a "clean" patient sequence and a "corrupted" variant
  (e.g. one diagnosis code swapped/removed), cache clean activations, then
  patch individual clean activations into the corrupted run to see which
  component(s) causally restore the clean prediction.
- Highest-effort, highest-rigor method here — identifies *which specific
  head/layer* is responsible for a specific behavior (e.g. "which
  component links CIR codes to MUS false positives," tying directly back
  to the Step 9 error/confusion finding).
- Realistically a stretch goal; only pursue if Items 1-4 raise a specific
  question worth chasing causally.

## 6. Counterfactual code-insertion / comorbidity graph extraction
- Systematically insert a single diagnosis code into synthetic/template
  patient histories and measure the shift in predicted probability for
  every other label.
- Aggregating this across many "inserted codes" builds a model-derived
  comorbidity graph, directly comparable to the Step 9 cross-category
  confusion clusters (e.g. does inserting a CIR code really raise MUS
  predictions the way the FP analysis suggested?).
- Good bridge between the "simple" observational findings (Step 9) and a
  causal explanation for them.

## 7. Concept Activation Vectors (TCAV-style) — advanced/optional
- Define a "concept direction" in hidden-state space (e.g. difference in
  mean activation between sequences with vs. without a cardiovascular
  comorbidity), then test how sensitive specific predictions are to moving
  along that direction.
- More setup (need concept-positive/negative example sets), best treated
  as a stretch item.

## 8. Sparse autoencoders / dictionary learning — future work only
- Cutting-edge mechanistic-interpretability technique (find monosemantic
  features in superposed activations).
- Explicitly out of scope for this thesis timeline; listed only for
  completeness / future-work section.

---

## Suggested order if pursued
1. Integrated Gradients (1) — cheapest high-rigor win, directly upgrades
   the Step 9 error/confusion story from correlational to attribution-based.
2. Attention rollout (2) — cheap upgrade over raw attention, reuses
   existing capture code.
3. Head ablation (4) — cheap causal signal, no new libraries.
4. Probing classifiers (3) — moderate effort, interesting "what layer
   knows what" narrative.
5. Counterfactual insertion (6) — directly tests the Step 9 hypothesis
   causally; good payoff if time allows.
6. Activation patching (5), CAVs (7), SAEs (8) — only if time remains and
   a specific mechanistic question has emerged from 1-5.

## Explicit ROI warning (per project docs)
`../../docs/publication_readiness_summary.md` states deep mechanistic
interpretability is **not worth spending much time on** for this thesis.
Treat this folder as an opportunistic menu — pick at most 1-2 items only if
they directly strengthen a claim already made elsewhere (e.g. the Step 9
comorbidity-confusion finding), not as a work stream to complete in full.
