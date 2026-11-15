#!/bin/bash
# Graph-mining GPU work queued behind cv-3Drecon: starts only after its finish script (PID $1) has exited,
# and every job waits until its GPU has no compute process at all. Baselines first, exploration after.
cd /home/hsy/graph-mining
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphmine
FINISH_PID=$1
log() { echo "$(date -u '+%m-%d %H:%M') $*"; }

while kill -0 "$FINISH_PID" 2>/dev/null; do sleep 60; done
log "cv-3Drecon finish script ended"

gpu_idle() { [ -z "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader)" ]; }
wait_gpu() { until gpu_idle "$1"; do sleep 60; done; }

P=results/1114-products; mkdir -p $P/logs
# 1) baselines (3 seeds: products test std is 0.1-0.3 pt): GCN on GPU1, SIGN (retuned) on GPU0
( wait_gpu 1; log "gcn start"
  python scripts/run.py --dataset products --model gcn --method base --seeds 3 --device cuda:1 --out $P > $P/logs/products_gcn_base.log 2>&1
  log "gcn done" ) &
( wait_gpu 0; log "sign start"
  python scripts/run.py --dataset products --model sign --method base --seeds 3 --device cuda:0 --out $P > $P/logs/products_sign_base.log 2>&1
  log "sign done" ) &
wait

# 2) exploration: Citeseer CoE on GPU0 (10 seeds, cheap), products CoE on GPU1 (1 seed first; more only if the gap is near the noise)
C=results/1114-planetoid-coe; E=results/1114-products-coe; mkdir -p $C/logs $E/logs
( wait_gpu 0; log "citeseer coe start"
  python scripts/run_ensemble.py --dataset citeseer --seeds 10 --device cuda:0 --out $C > $C/logs/citeseer_coe.log 2>&1
  log "citeseer coe done" ) &
( wait_gpu 1; log "products coe start"
  python scripts/run_ensemble.py --dataset products --seeds 1 --experts mlp lp sign --device cuda:1 --out $E > $E/logs/products_coe_mlp-lp-sign.log 2>&1
  log "products coe done" ) &
wait
log "queue done"
