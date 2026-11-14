# Experiment log

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
