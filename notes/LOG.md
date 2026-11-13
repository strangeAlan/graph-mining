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
