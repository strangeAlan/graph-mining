#!/bin/bash
# Full Cora / Citeseer sweep: baselines + label propagation + self-training variants, 10 seeds each.
OUT=${1:-results/1113-planetoid}
mkdir -p $OUT/logs
for ds in cora citeseer; do
  python scripts/run.py --dataset $ds --method lp --out $OUT > $OUT/logs/${ds}_lp.log 2>&1
  for m in mlp gcn gat appnp sage sign; do
    python scripts/run.py --dataset $ds --model $m --method base --seeds 10 --out $OUT > $OUT/logs/${ds}_${m}_base.log 2>&1
  done
  for m in gcn gat appnp sign; do for meth in st lpst; do
    python scripts/run.py --dataset $ds --model $m --method $meth --seeds 10 --out $OUT > $OUT/logs/${ds}_${m}_${meth}.log 2>&1
  done; done
done
