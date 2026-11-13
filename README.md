# Graph Data Mining: node classification on Cora, Citeseer and ogbn-products

Course project (Big Data Processing, Section 5).

## Setup
```bash
conda create -n graphmine python=3.10
pip install torch==2.5.1 torch_geometric ogb scikit-learn pandas matplotlib pyyaml tqdm
```
Data goes to `data/` (not tracked): Planetoid public split via PyG, ogbn-products via OGB.

## Run
```bash
python scripts/run.py --dataset cora --model gcn --method lpst --seeds 10 --out results/<run>
scripts/run_planetoid.sh results/<run>          # all Cora / Citeseer experiments
scripts/run_products.sh results/<run> cuda:0    # ogbn-products experiments
```
Models: `mlp gcn gat appnp sage sign`. Methods: `base`, `lp` (label propagation),
`st` (confidence self-training), `lpst` (self-training with label-propagation agreement).
Every run also reports the result after Correct & Smooth, with its strength chosen on validation.
Hyper-parameters are in `configs/default.yaml`; experiment notes are in `notes/LOG.md`.
