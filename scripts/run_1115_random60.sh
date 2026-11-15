#!/bin/bash
# 11/15: CoE on Cora / Citeseer with a random 60/20/20 split (fixed split, 10 model seeds), stage 0 only.
# Question: does fusion beat the best expert once the small graphs have many labels, as on ogbn-products?
cd /home/hsy/graph-mining
source ~/miniconda3/etc/profile.d/conda.sh; conda activate graphmine
OUT=results/1115-planetoid-random60; mkdir -p $OUT/logs
for ds in cora citeseer; do
  python scripts/run_ensemble.py --dataset $ds --split random60 --seeds 10 --no-self-train --tag _random60 \
      --device cuda:0 --out $OUT > $OUT/logs/${ds}_coe_random60.log 2>&1
done
