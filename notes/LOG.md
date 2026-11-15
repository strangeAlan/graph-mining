# Experiment log

## 11/15 new exploration: evidence-conditioned fusion (Codex)
- Frozen base: `5e818ea5dbf76d72e42177bebcb232ae6cb5fd12`; clean working tree at start;
  branch `research/evidence-gated-fusion`. Existing CoE experiments/report remain the baseline.
- Constraint from prior work: global CoE margin did not beat stacking; small-graph fusion did not
  reliably beat voting; products benefit mainly comes from LP. Do not repeat margin sweeps.
- Hypothesis: entropy alone cannot distinguish one-source certainty from agreement among many
  label sources. The diffusion influence concentration statistic
  `ESS_i = (sum_j H_ij)^2 / sum_j H_ij^2`, for labelled sources j, may help a node-wise router
  combine feature/GNN experts with LP. This is not a claim of statistically independent labels.
- Implementation: exact source influence on small graphs (<=256 labels), 32 seeded Rademacher
  projections otherwise; LP abstention; out-of-fold expert predictions AND evidence; a small linear
  softmax router regularised towards a global mixture. No self-training in the first experiment.
- Paired controls on identical cached predictions: best validation expert, mean/accuracy voting,
  global convex mixture, confidence-only node gate, full evidence gate, old stacking and CoE.
  Small-graph experts match old seven-expert suite; products uses MLP/LP/SIGN.
- Protocol: `configs/evidence.yaml`; public split, seeds 0/1/2 on Cora and Citeseer, normal model
  epoch budgets/early stopping. Split hashes, code hashes, hyperparameters and cache fingerprints
  saved in manifests; every selected gate is written before test scoring. Test labels only used
  for final evaluation/diagnostic strata. Cora is the success-protection case, Citeseer the sparse
  label-coverage case. Products seed 0 is a possible independent scalability/domain-shift check.
- Stop rule: complete the paired small-graph experiment; if evidence fails against confidence-only
  and global routing on validation, diagnose implementation once rather than iterate test-guided
  patches. No claim of gain before the experiment completes.
- Related work checked: NODE-MOE (arXiv:2406.03464), Mowst (arXiv:2311.05185), and CoE
  (arXiv:2505.20853). Node-wise gating is established; proposed course-level contribution is the
  source-concentration evidence and its leakage-safe use, not generic mixture-of-experts novelty.
- Mathematical checks: 4 CPU unit tests pass (same entropy/different source count, exact operator
  and random-sign estimate, held-out labels replaced with invalid values leave predictions unchanged,
  and no-signal LP receives exactly zero router weight). CSR propagation used for all paired methods.
- Completed public-split seeds 0/1/2, `results/1115-evidence` (test %, population std): Cora
  best validation expert 82.80±0.54, confidence gate 82.57±0.25, evidence gate 82.80±0.45,
  stacking 82.90±0.16, CoE 82.37±0.25. Citeseer best validation expert 71.93±0.60,
  confidence gate 71.67±0.68, evidence gate 71.87±0.65, stacking 71.60±0.59, CoE 71.70±0.49.
  Evidence-vs-confidence mean validation delta: Cora +0.20 pp, Citeseer -0.13 pp.
  **No convincing small-graph gain**; no small-graph hyperparameter patching or seed expansion.
- One implementation/mechanism diagnosis using validation labels: high-confidence LP (>=.9) on
  Citeseer has accuracy 41/56=73.2% for ESS<=2, versus 15/15 for ESS in (2,8]; on Cora the
  corresponding accuracies are 34/36=94.4% and 40/41=97.6%. Sample sizes are small. Across ALL
  nodes, larger ESS is not monotonic in LP accuracy (diffusion also crosses class boundaries).
  ESS must be interpreted jointly with disagreement/entropy, never as calibrated correctness.
- Numerical approximation check, 32 probes vs exact influence on validation nodes with signal:
  relative ESS error median/p90 = 11.0%/32.1% (Cora), 10.0%/23.5% (Citeseer), seed 9173.
- products seed 0 started as the predeclared independent large-graph check, GPU0 only, after both
  citation experiments completed and checking there were no other GPU jobs. No cv-3Drecon work
  is running at launch. The sparse CSR/minibatch path avoids full-batch GCN/SAGE memory cost.
- Before products fusion results are available: confirmation will be run at unchanged settings on
  seeds 1/2 only if evidence gating beats BOTH confidence-only gating and stacking by >=0.10 pp
  on validation in seed 0. Otherwise stop this design after the seed-0 large-graph result.
- products seed 0 COMPLETE (1171 s excluding preprocessing): SIGN 77.877%, global convex mixture
  77.951%, confidence gate 78.196%, evidence gate 78.278%, stacking 79.622%, CoE 79.539%.
  Evidence validation 93.2762% vs confidence 93.2686% and stacking 93.2660%: confirmation threshold
  NOT met; no extra expert-training seeds. The convex-router design is not a performance innovation.
- Targeted structural diagnosis: the convex global mixture already trails stacking by 1.67 pp.
  The proposed gate discarded CoE's class-to-class correction matrices, so it is not a clean
  node-conditioning extension of the working baseline. One final aligned check will freeze the
  fitted stacking matrix and apply a zero-initialised node gate to its expert contributions, with
  epoch 0 exactly reproducing stacking. Compare confidence-only vs evidence gates with identical
  class matrices and cached experts. No new backbone fits, no test-guided hyperparameter changes.
  If this matched check does not improve validation over both controls, stop the routing direction.
- Capacity-aligned check COMPLETE: `scripts/run_conditional_coe.py`,
  `results/1115-conditional-coe`, same expert caches and seeds. Frozen stacking matrices are
  reconstructed at the previously validation-selected settings (validation reproduction asserted).
  Gate contribution is `sum_j K softmax(b+Wx)_j Theta_j p_j`, with b=W=0 at epoch 0.
  Static class priors from the uniform LP row are retained so epoch zero exactly matches stacking;
  no-signal status is an input to the adaptive gate, not a new hard mask on the baseline.
  'context' removes ESS only, retaining LP compatibility, coverage, availability and degree.
- Aligned test accuracy (%; small graphs 3 seeds, products 1):

  | Method | Cora | Citeseer | products |
  |---|---:|---:|---:|
  | Frozen stacking | 82.90 +/- 0.16 | 71.60 +/- 0.59 | 79.622 |
  | Global expert rescaling | 82.90 +/- 0.16 | 72.97 +/- 0.90 | 79.431 |
  | Confidence-conditioned rescaling | 82.90 +/- 0.16 | 72.70 +/- 0.75 | 79.544 |
  | Context gate (no ESS) | 82.63 +/- 0.41 | 71.43 +/- 0.21 | 80.147 |
  | Evidence gate (with ESS) | 82.60 +/- 0.45 | 71.87 +/- 0.21 | 80.272 |

- products validation: frozen stack 93.2660, global 93.3093, confidence 93.3347,
  context 93.3398, evidence 93.3321 (%). Evidence improves test by +0.650 pp over stacking,
  but is one validation node below confidence and three below context. The extra ESS feature
  improves test only +0.124 pp over context; it has NO validation advantage. These are exploratory
  single-seed results, not established superiority or evidence for ESS as the main contribution.
- Decision: stop gate variants and extra seed launches under the predeclared validation rule.
  Retain class-aware context/evidence conditioning as a products candidate, not a universal method.
  Small-graph source-aware routing fails to improve consistently; Citeseer actually benefits more
  from simple global rescaling. The current evidence does not justify making ESS the report's
  primary innovation. Any future work needs a new hypothesis (e.g. label-conditioned representations
  or genuinely new structural information), rather than another confidence/regularisation tweak.
- Final checks: 5 unit tests passed, including exact zero-gate reconstruction of stacking and
  non-zero gate gradients while class matrices stay frozen; compilation and `git diff --check` pass.
  First experiment's exact implementation/results are frozen in commit `a4135d4`.
  The follow-up also removes an empty-tensor std warning in the global router; it does not alter
  that router's numerical definition. All runs completed; no training processes left running.

## 11/13 setup
- Env: conda `graphmine` (py3.10, torch 2.5.1+cu124, PyG 2.8.1, OGB 1.3.6), packages via Aliyun PyPI (tsinghua 403 on this server; conda via NJU mirror with CONDARC=/dev/null).
- Data (gitignored, `data/`): Planetoid public split (raw files from GitHub via proxy, ~2 MB);
  ogbn-products raw CSVs taken from hf-mirror `zkchen/OGBProducts` (official snap.stanford.edu ~10 KB/s direct),
  processed by OGB itself; stats match official (2,449,029 nodes, 47 classes, split 196,615/39,323/2,213,091).
- Method under test (`lpst`): self-training whose pseudo-labels need confidence >= tau AND agreement with
  label propagation from the training labels; validation nodes never pseudo-labelled; stage picked on val;
  C&S on top with strengths (incl. 0 = off) picked on val. Controls: `base`, `st` (confidence only).
- Cora 2-seed interface check: GCN 82.2 -> lpst 82.9, pseudo-label precision 97-99%; C&S with alpha>0 hurt GCN on Cora (82.2 -> 80.0), hence 0 added to the grid.

## 11/13 runs launched (detached)
- `scripts/run_planetoid.sh results/1113-planetoid` (cuda:0, 10 seeds)
- `scripts/run_products.sh results/1113-products cuda:1` (SIGN 5 seeds, full-batch GCN/SAGE 3 seeds; untested on products before launch)

## 11/14 direction: Cooperation of Experts (user's ICML'25 paper, arXiv 2505.20853, code github.com/strangeAlan/CoE)
- CoE combine step (Cooperation.py): expert class distributions stacked as G [kC x n]; cooperation matrix Theta [C x kC]
  initialised from expert train accuracies; prediction softmax(Theta G); loss = CE + gamma * large margin
  (true-class prob minus smooth max of the others, logsumexp alpha=100, gamma=100); fitted on expert predictions of the training nodes.
- Transfer to semi-supervised node classification (`src/graphmine/ensemble.py`):
  experts = MLP (features only), LP (labels only), GCN, GAT, APPNP, SAGE, SIGN;
  Theta fitted on 5-fold out-of-fold predictions (in-sample predictions are ~100% on 140 labels, LP returns labels verbatim);
  Theta_j = 10 * acc_j * I init (= accuracy-weighted voting), L2 pull to init; (gamma, reg) chosen on validation;
  controls: mean vote, accuracy vote, stacking (gamma = 0); optional one round of ensemble self-training.
- Fixes in the same commit: LP abstains (uniform, has_signal=False) on nodes without label mass instead of an all-zero row
  (argmax = class 0; Citeseer ~31.6% of nodes, found by Astra review); lpst keeps such nodes on confidence alone;
  true and pseudo-label losses averaged separately (CE_true + w * CE_pseudo).
- Cora 1-seed check: experts 59.2 (MLP) .. 83.4 (APPNP); mean vote 82.4, acc vote 82.6, stack 83.1, CoE 82.7, CoE+C&S 83.1;
  after ensemble self-training (tau 0.7, 1973 pseudo-labels, precision 0.842) experts rise to 82.0..84.4, CoE 83.1.
- Launched: results/1114-planetoid-coe (10 seeds), results/1114-planetoid (re-run of the sweep with the fixes),
  products SIGN retune (hops 5, input_drop 0.2, 1000 epochs) because SIGN 77.7 < published 80.5.
- products SIGN retune (1 seed each, val-selected): hops 3 old config 77.70 (5-seed mean); hops 5 + input_drop 0.2: 78.02 (batch 50k), 78.19 (batch 10k), 78.09 (600 fixed epochs, no early stop).
  Val saturates at ~93.1-93.2 early; longer training does not move test. Keep hops 5 / batch 50k (78.0) and stop tuning; gap to published 80.5 unexplained.

## 11/14 Cora CoE, 10 seeds (results/1114-planetoid-coe/cora_coe.json; fixed baselines in results/1114-planetoid)
- Stage 0: best single expert APPNP 83.24±0.87; mean vote 82.79, acc vote 82.92, stacking 82.97, CoE 82.72±0.52, CoE+C&S 82.93.
  -> fusion does not beat the best expert; the large-margin term does not beat plain stacking.
- Stage 1 (ensemble pseudo-labels, tau 0.7, ~2000 nodes, precision ~0.84): every expert rises (GCN 83.43, APPNP 84.32, MLP 81.98);
  fused 83.66 still below APPNP after self-training. Single-model self-training (fixed code): GCN st 83.36, APPNP st 84.04, lpst 84.11.
- Reading: on Cora, 140 labels are too few to learn the fusion and the GNN experts are highly correlated; the gain comes from
  self-training, not from fusion. Products (196k labels, diverse experts) is the real test of CoE.
- products CoE launched with experts MLP/LP/SIGN (3 folds, 3 seeds, GPU0); SAGE full-batch uses ~36 GB and >12 min per fit, left out for now.

## 11/14 05:41 incident: GPU sharing with cv-3Drecon and lost runs
- My products jobs on GPU1 (SAGE full-batch ~36 GB from 03:43) made cv-3Drecon COLMAP-RADIAL shard 1 (GPU1) hit CUDA OOM in SIFT
  extraction at 03:54-03:56: 1113-main "playground random 20 views s2" and all 5 shard-1 cases of 1113-allviews
  (relief_2, pipes, office, electro, delivery_area) FAILED with 0 registered. Its runtimes after 11/13 ~17:25 shared GPUs with my jobs
  (peak_gpu_gb is per-process, unaffected).
- A GPU guard I wrote to free both GPUs before the cv-3Drecon dense stage self-matched its pgrep pattern and fired at once,
  killing all graph-mining jobs: Citeseer CoE (7/10 seeds, not saved: JSON written only at the end), products CoE (2.5 h, no seed yet),
  products GCN baseline (1 seed: 77.44, +C&S 78.69), the queued SIGN re-run, and the tail of results/1114-planetoid.
- Policy from now: no graph-mining GPU jobs while cv-3Drecon GPU stages run; per-seed incremental saving.
- Recovery: cv-3Drecon finish_1113.sh paused (SIGSTOP), the 6 failed cases moved to cv-3Drecon/runs/_oom_failed_20261114 and re-run
  by cv-3Drecon/scripts/rerun_1114_oom_cases.sh (resumes the finish script at the end). playground random 20 s2: 12/20 registered
  on re-run (was FAILED) -> the failure was caused by the OOM.
- Scripts now save the JSON after every seed (`complete` flag).
- Graph GPU work queued behind cv-3Drecon: scripts/queue_after_3drecon.sh waits for finish_1113.sh to exit and for each GPU to have
  no compute process; order: products GCN (3 seeds) + SIGN retuned (5 seeds) baselines -> Citeseer CoE (10) + products CoE (3).
  Log: results/queue_after_3drecon.log.

## 11/14-15 queue results (after cv-3Drecon finished; GPUs exclusive)
- products baselines, 3 seeds (results/1114-products): GCN 77.57±0.18 (+C&S 78.71), SIGN hops5 77.95±0.07 (+C&S 78.78). SAGE 78.76±0.3 (1113).
- Citeseer CoE, 10 seeds: stage0 best expert GAT 72.04; mean vote 71.27, acc vote 71.57, stack 71.54, CoE 71.61.
  stage1 (ensemble self-training): experts 70.9-73.2 (APPNP 73.19, GAT 73.10), CoE 72.30.
  Single-model self-training (1114-planetoid): GCN st 73.49, lpst 73.56; APPNP st 72.77 -> ensemble ST no better than single-model ST.
- Cora: ensemble-ST APPNP 84.32±0.34 vs single APPNP st 84.04±0.44 (within noise).
- products CoE, 1 seed, experts MLP/LP/SIGN (3 folds, 3.1 h): stage0 SIGN 77.92, LP 71.26, MLP 60.25;
  mean vote 79.06, acc vote 78.87, stacking 79.63, CoE 79.53 (gamma 1). Fusion +1.6 over the best expert and +0.75 over SIGN+C&S (78.78);
  large-margin term = stacking (no gain). Stage1 (1.66M pseudo-labels, precision 0.914): SIGN 80.48, LP 78.79, CoE 80.82.
- Reading: CoE-style fusion helps where labels are plentiful and experts use different information (products); on the
  140-label graphs it equals voting and stays below the best expert. Margin term adds nothing over plain stacking anywhere.
- 11/15 launched (scripts/run_1115_products.sh, results/1115-products): SIGN single-model self-training (control for ensemble ST),
  CoE with MLP+SIGN only, stage 0 (does the gain need the LP expert?).

## 11/15 controls and random split
- products CoE with MLP + SIGN only (no LP), stage 0: 78.31 (SIGN 78.12, mean vote 78.23, stack 78.34) -> the products fusion gain
  comes mostly from the LP expert; CoE = learned combination of LP with a feature model, better than C&S (SIGN+C&S 78.78, SAGE+C&S 79.0).
- products SIGN single-model self-training (1 seed): val 93.18 / 93.16 / 93.05 and test 77.91 / 79.42 / 80.00 for stages 0/1/2
  (pseudo precision 0.914, 0.857); val-selection picks stage 0. Validation does not track test across self-training rounds on the
  sales-rank split. Compared at one round: SIGN on its own pseudo-labels 79.42 vs on CoE pseudo-labels 80.48 (CoE 80.82).
  The report uses the one-round number for both (fixed in the ensemble design, not chosen on test).
- Cora random 60/20/20 (fixed split, 1624 train labels), 10 seeds, stage 0: experts APPNP 87.05, GCN 86.70, SAGE 86.38 (val-best),
  GAT 85.55, LP 83.03; acc vote 86.44, stack 86.48, CoE 86.49±0.34. reg chosen mostly 0-0.01 (Theta free), still no gain
  -> label budget is not the reason; GNN experts make similar errors.
- Citeseer random 60/20/20 (1996 train labels), 10 seeds: experts SAGE 77.10, APPNP 76.40 (val-best), GCN 76.34, GAT 76.07,
  LP 69.97; mean vote 77.34, acc vote 77.40, stack 77.52, CoE 77.52±0.57 -> combinations ~0.4 above the best expert, CoE = voting.
