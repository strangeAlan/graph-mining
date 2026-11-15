#!/bin/bash
# Commands behind every number in the report (run from the repo root, conda env `graphmine`).
# Each block writes JSON summaries into results/<run>/; scripts/make_tables.py turns them into report/tables/*.tex.
# Approximate single-A100 times are given per block. Do not share a GPU with other projects' jobs (see notes/LOG.md, 11/14).
set -e

# Cora / Citeseer, public split, 10 seeds: all reference models, label propagation,
# single-model self-training (st) and LP-agreement self-training (lpst).                       (~1.5 h)
scripts/run_planetoid.sh results/1114-planetoid

# ogbn-products reference models.                                                             (~4 h)
python scripts/run.py --dataset products --method lp --out results/1113-products
python scripts/run.py --dataset products --model mlp  --method base --seeds 5 --out results/1113-products
python scripts/run.py --dataset products --model sage --method base --seeds 3 --out results/1113-products
python scripts/run.py --dataset products --model gcn  --method base --seeds 3 --out results/1114-products
python scripts/run.py --dataset products --model sign --method base --seeds 3 --out results/1114-products

# Cooperation of Experts with one round of ensemble self-training.
scripts/run_planetoid_coe.sh results/1114-planetoid-coe                                       # (~0.5 h alone)
python scripts/run_ensemble.py --dataset products --seeds 1 --experts mlp lp sign \
    --out results/1114-products-coe                                                           # (~3 h)

# Controls on ogbn-products: single-model self-training of SIGN, CoE without the LP expert.    (~3 h)
scripts/run_1115_products.sh

# Cora / Citeseer with a random 60/20/20 split: does fusion help with many labels?             (~1 h)
scripts/run_1115_random60.sh

# Node-wise cooperation (exploratory, Discussion): routers on cached out-of-fold experts, then a gate on the
# fitted stacking matrices. Protocol in configs/evidence.yaml.                                  (~1.5 h)
python scripts/run_evidence.py --dataset cora --device cuda:0 --out results/1115-evidence
python scripts/run_evidence.py --dataset citeseer --device cuda:0 --out results/1115-evidence
python scripts/run_evidence.py --dataset products --seeds 0 --device cuda:0 --out results/1115-evidence
for ds in products cora citeseer; do python scripts/run_conditional_coe.py --dataset $ds --device cuda:0; done
python scripts/summarize_evidence.py results/1115-evidence
python scripts/summarize_evidence.py results/1115-conditional-coe

python scripts/make_tables.py
bash report/build.sh report
