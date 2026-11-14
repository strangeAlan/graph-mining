#!/bin/bash
# After the 11/13 products sweep has produced its GCN baseline, stop it (its last item, SAGE+lpst, is superseded)
# and run the SIGN baseline with the retuned config (hops 5) on the same GPU.
OLD_SWEEP_PID=$1
until [ -f results/1113-products/ogbn-products_gcn_base.json ]; do sleep 60; done
pkill -P $OLD_SWEEP_PID; kill $OLD_SWEEP_PID
sleep 30
OUT=results/1114-products; mkdir -p $OUT/logs
python scripts/run.py --dataset products --model sign --method base --seeds 5 --device cuda:1 --out $OUT > $OUT/logs/products_sign_base.log 2>&1
