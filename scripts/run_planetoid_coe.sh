#!/bin/bash
# CoE on Cora / Citeseer, 10 seeds.
OUT=${1:-results/1114-planetoid-coe}
mkdir -p $OUT/logs
for ds in cora citeseer; do
  python scripts/run_ensemble.py --dataset $ds --seeds 10 --out $OUT > $OUT/logs/${ds}_coe.log 2>&1
done
