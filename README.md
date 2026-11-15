# Graph Data Mining: node classification on Cora, Citeseer and ogbn-products

Course project (Big Data Processing, Section 5, Graph Data Mining).
Base task: node classification on Cora and Citeseer. Extension: ogbn-products with the official OGB evaluator.
On top of the reference models we adapt Cooperation of Experts (CoE, Wang et al., ICML 2025) to node classification.

- Report (ACM sigconf, English): `report/report.tex` → `report/build/report.pdf` (`bash report/build.sh report`)
- Experiment log with every number and decision: `notes/LOG.md`
- State of the work and open directions for whoever continues: `HANDOFF.md`

## Evidence-conditioned fusion experiment (11/15)

The independent branch `research/evidence-gated-fusion` tests whether node-wise fusion
benefits from the concentration of label-source influence, beyond prediction confidence.
For propagation coefficients from training sources, it computes
`ESS(i) = (sum_j H_ij)^2 / sum_j H_ij^2`: one repeatedly propagated source still counts
as one effective source. This is an influence-concentration statistic, not a count of
statistically independent labels. Small graphs use exact coefficients; large graphs use
32 seeded random-sign projections. No N-by-N dense matrix is constructed.

Expert predictions and graph evidence are both constructed out of fold. LP abstains
when it receives no label mass. Three paired routers share the same frozen experts:
global mixture weights, confidence-only node-wise weights, and evidence-conditioned
node-wise weights. Existing voting, stacking and CoE are evaluated on those exact
same predictions. The initial experiment has no self-training.

```bash
python -m unittest discover -s tests -v
python scripts/run_evidence.py --dataset cora --device cuda:0 --out results/1115-evidence
python scripts/run_evidence.py --dataset citeseer --device cuda:1 --out results/1115-evidence
# Large-graph feasibility experiment; check shared GPU availability first:
python scripts/run_evidence.py --dataset products --seeds 0 --device cuda:0 --out results/1115-evidence
python scripts/summarize_evidence.py results/1115-evidence
```

The fixed protocol and stopping rule are in `configs/evidence.yaml`. Results include
split/code hashes, selected hyperparameters, paired metrics and source-support strata.
Each expert's predictions are cached atomically in gitignored `.pt` files to avoid
retraining when inspecting fusion. Validation selection is saved before test scoring.
This is an experimental hypothesis; consult `notes/LOG.md` for measured outcomes.

## Layout

```
src/graphmine/
  data.py       datasets, splits (public / random 60-20-20), accuracy (OGB evaluator for products)
  prop.py       sparse normalised adjacency, SIGN hop features, label propagation, Correct & Smooth
  models.py     MLP, GCN, GAT, APPNP, GraphSAGE (full-batch, sparse matmul), SIGN (mini-batch)
  train.py      training loop with early stopping, self-training (st / lpst), C&S search on validation
  ensemble.py   Cooperation of Experts: out-of-fold expert predictions, cooperation matrix, large-margin loss
scripts/
  run.py            one model x method x seeds  -> results/<run>/<dataset>_<model>_<method>.json
  run_ensemble.py   CoE (+ ensemble self-training) -> results/<run>/<dataset>_coe<tag>.json
  run_planetoid.sh, run_planetoid_coe.sh, run_1115_*.sh   sweeps used for the report
  reproduce.sh      all commands behind the report numbers
  make_tables.py    results JSON -> report/tables/*.tex
configs/default.yaml   all hyper-parameters (per dataset, per model, self-training, CoE grid)
results/               JSON summaries of the runs used in the report (logs are not tracked)
report/                LaTeX source, ACM template, generated tables
```

## Setup

```bash
conda create -n graphmine python=3.10
conda activate graphmine
pip install torch==2.5.1 torch_geometric ogb scikit-learn pandas matplotlib pyyaml tqdm
```

Data goes to `data/` (not tracked). Cora/Citeseer: PyG `Planetoid` (public split). ogbn-products: OGB
`PygNodePropPredDataset` under `data/ogb/`; on this server the raw files came from the hf-mirror copy
`zkchen/OGBProducts` because the Stanford host is very slow from here (statistics checked against OGB).
For package mirrors on this server see `notes/LOG.md` (Tsinghua mirrors return 403; Aliyun PyPI works).

## Running

```bash
python scripts/run.py --dataset cora --model gcn --method base --seeds 10 --out results/<run>
python scripts/run.py --dataset products --model sign --method st --seeds 1 --out results/<run>
python scripts/run_ensemble.py --dataset citeseer --seeds 10 --out results/<run>
python scripts/run_ensemble.py --dataset cora --split random60 --no-self-train --tag _random60 --out results/<run>
python scripts/make_tables.py && bash report/build.sh report
```

`run.py` methods: `base`, `lp` (label propagation only), `st` (self-training with confident pseudo-labels),
`lpst` (pseudo-labels must also agree with label propagation). Every run also reports the result after Correct & Smooth,
with its strengths (including "off") chosen on validation accuracy. All model selection uses validation accuracy only.

## Main results (test accuracy, %)

Reference models (mean over 10 seeds on Cora/Citeseer, 3-5 seeds on ogbn-products; see report Table 2):

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

Cooperation of Experts (see report Table 3 and `HANDOFF.md` for the full comparison):

| | Cora | Citeseer | ogbn-products |
|---|---|---|---|
| best expert, chosen on validation | 82.3 | 71.1 | 77.9 |
| CoE | 82.7 | 71.6 | 79.5 |
| CoE after ensemble self-training | 83.7 | 72.3 | 80.8 |
