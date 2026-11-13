#!/bin/bash
# ogbn-products sweep; each run independent so one failure does not block the rest.
OUT=${1:-results/1113-products}; DEV=${2:-cuda:1}
mkdir -p $OUT/logs
r() { python scripts/run.py --dataset products --device $DEV --out $OUT "$@" > $OUT/logs/$(echo "$@" | tr ' =' '__').log 2>&1; }
r --method lp
r --model sign --method base --seeds 5
r --model sign --method lpst --seeds 5
r --model sign --method st --seeds 5
r --model mlp --method base --seeds 5
r --model sage --method base --seeds 3
r --model gcn --method base --seeds 3
r --model sage --method lpst --seeds 3
