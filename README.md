# Graph Data Mining: node classification on Cora, Citeseer and ogbn-products

Course project (Big Data Processing, Section 5, Graph Data Mining).
Base task: node classification on Cora and Citeseer. Extension: ogbn-products with the official OGB evaluator.
On top of the reference models we adapt Cooperation of Experts (CoE, Wang et al., ICML 2025) to node classification,
and test a node-wise extension of it.

- Report (ACM sigconf, English): `report/report.tex` → `report/build/report.pdf` (`bash report/build.sh report`)
- Experiment log with every run, number and decision: `notes/LOG.md`

## Status

- Reference models on all three datasets: complete (Cora/Citeseer 10 seeds, ogbn-products 3-5 seeds).
- CoE for node classification, with ensemble self-training, controls and a random 60/20/20 split study: complete.
- Node-wise (evidence-conditioned) cooperation: one exploratory round, complete; not adopted as a result (see below).
- Report: complete draft (v1.3); the author affiliation is still a placeholder.

## Layout

```
src/graphmine/
  data.py       datasets, splits (public / random 60-20-20), accuracy (OGB evaluator for products)
  prop.py       sparse normalised adjacency, SIGN hop features, label propagation, Correct & Smooth
  models.py     MLP, GCN, GAT, APPNP, GraphSAGE (full-batch, sparse matmul), SIGN (mini-batch)
  train.py      training loop with early stopping, self-training (st / lpst), C&S search on validation
  ensemble.py   Cooperation of Experts: out-of-fold expert predictions, cooperation matrix, large-margin loss
  evidence.py   label-source evidence (coverage, effective source count) and node-wise gates
scripts/
  run.py                  one model x method x seeds   -> results/<run>/<dataset>_<model>_<method>.json
  run_ensemble.py         CoE (+ ensemble self-training) -> results/<run>/<dataset>_coe<tag>.json
  run_evidence.py         node-wise routers on cached out-of-fold experts -> results/1115-evidence
  run_conditional_coe.py  node-wise gate on top of the fitted stacking matrices -> results/1115-conditional-coe
  summarize_evidence.py   comparison.csv for the two experiments above
  run_planetoid.sh, run_planetoid_coe.sh, run_1115_*.sh   sweeps used for the report
  reproduce.sh            all commands behind the report numbers
  make_tables.py          results JSON -> report/tables/*.tex
configs/default.yaml   hyper-parameters (per dataset, per model, self-training, CoE grid)
configs/evidence.yaml  protocol of the node-wise experiments
tests/                 unit tests of the evidence statistics (leakage, exactness, gradients)
results/               JSON summaries of all runs (logs and prediction caches are not tracked)
report/                LaTeX source, ACM template, generated tables
```

## Setup

```bash
conda create -n graphmine python=3.10
conda activate graphmine
pip install torch==2.5.1 torch_geometric ogb scikit-learn pandas matplotlib pyyaml tqdm
python -m unittest discover -s tests
```

Data goes to `data/` (not tracked). Cora/Citeseer: PyG `Planetoid` (public split). ogbn-products: OGB
`PygNodePropPredDataset` under `data/ogb/`. On our server the raw files came from the hf-mirror copy
`zkchen/OGBProducts` because the Stanford host is very slow from there; node, edge and split counts match OGB.

## Running

```bash
python scripts/run.py --dataset cora --model gcn --method base --seeds 10 --out results/<run>
python scripts/run.py --dataset products --model sign --method st --seeds 1 --out results/<run>
python scripts/run_ensemble.py --dataset citeseer --seeds 10 --out results/<run>
python scripts/run_ensemble.py --dataset cora --split random60 --no-self-train --tag _random60 --out results/<run>
python scripts/run_evidence.py --dataset cora --device cuda:0 --out results/1115-evidence
python scripts/run_conditional_coe.py --dataset products --device cuda:0
python scripts/make_tables.py && bash report/build.sh report
```

`run.py` methods: `base`, `lp` (label propagation only), `st` (self-training with confident pseudo-labels),
`lpst` (pseudo-labels must also agree with label propagation). Every run also reports the result after
Correct & Smooth, with its strengths (including "off") chosen on validation accuracy.
All model selection uses validation accuracy only; test labels are used for reporting.

## Results (test accuracy, %)

Reference models (report Table 2):

| Model | Cora | Citeseer | ogbn-products |
|---|---|---|---|
| MLP | 58.4 | 60.0 | 60.3 |
| Label propagation | 71.0 | 49.8 | 71.3 |
| GCN | 82.0 | 71.2 | 77.6 |
| GAT | 82.3 | 72.0 | – |
| APPNP | 83.2 | 71.1 | – |
| GraphSAGE | 81.3 | 69.5 | 78.8 |
| SIGN | 72.5 | 63.2 | 78.0 |
| best model + C&S | 83.2 | 72.0 | 79.0 |

Cooperation of Experts (report Tables 3 and 4; ogbn-products one seed):

| | Cora | Citeseer | ogbn-products |
|---|---|---|---|
| best expert, chosen on validation | 82.3 (GAT) | 71.1 (APPNP) | 77.9 (SIGN) |
| best expert on test (reference) | 83.2 (APPNP) | 72.0 (GAT) | 77.9 (SIGN) |
| accuracy-weighted voting | 82.9 | 71.6 | 78.9 |
| stacking (CoE without the margin term) | 83.0 | 71.5 | 79.6 |
| CoE | 82.7 | 71.6 | 79.5 |
| CoE without the label-propagation expert | – | – | 78.3 |
| single-model self-training | 82.7 (GAT) | 72.8 (APPNP) | 79.4 (SIGN, one round) |
| CoE after ensemble self-training | 83.7 | 72.3 | 80.8 |

Findings:

1. On ogbn-products CoE is 1.6 points above its best expert and 0.5 above the best reference result (GraphSAGE + C&S);
   seed std there is 0.1-0.2. Most of the gain comes from the label-propagation expert (without it: +0.2 over SIGN).
2. On Cora/Citeseer CoE is no better than voting, also with a random 60/20/20 split (Cora 86.5 vs best expert 87.0,
   Citeseer 77.5 vs voting 77.4): the GNN experts make similar errors; the label budget is not the reason.
3. The CoE large-margin term never beats plain stacking.
4. Ensemble self-training lifts every expert; on the small graphs it equals single-model self-training, on ogbn-products
   the CoE pseudo-labels beat SIGN's own after one round (80.5 vs 79.4). On ogbn-products validation accuracy stays at
   ~93% across self-training rounds while test accuracy rises (sales-rank split), so a fixed single round is reported.
5. Node-wise cooperation (`evidence.py`, `run_conditional_coe.py`): a gate on top of the fitted stacking matrices, using
   label coverage, LP/model agreement, degree and an effective label-source count, reaches 80.27 on ogbn-products against
   79.62 for stacking (one seed); without the source count 80.15, so the count itself is not what helps. Validation does not
   separate the variants on ogbn-products, and on Cora/Citeseer (3 seeds) there is no gain (82.60 / 71.87 vs 82.90 / 71.60).
   Kept as an exploratory result.

## Practical notes

- ogbn-products is slow: full-batch GCN/GraphSAGE take 16-30 min and ~36 GB per fit; one CoE run (3 experts, 3 folds,
  plus self-training) takes ~3 h. Check that no other job is using the GPUs before launching.
- `pgrep -f` / `pkill -f` patterns also match the shell that launched them; kill jobs by PID.
- Label propagation returns a uniform row with `has_signal = False` for nodes whose component has no labelled node
  (Citeseer has 48 isolated nodes and several unlabelled components).
- The node-wise experiments cache out-of-fold expert predictions under `results/1115-evidence/cache/` (~1.5 GB, gitignored),
  so fusion variants can be re-run without retraining experts.
