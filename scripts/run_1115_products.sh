#!/bin/bash
# 11/15 products controls (1 seed each; products seed std is 0.1-0.2 pt):
#  GPU0: single-model self-training of SIGN (control for ensemble self-training)
#  GPU1: CoE without the label-propagation expert, stage 0 only (is the fusion gain more than label information?)
cd /home/hsy/graph-mining
source ~/miniconda3/etc/profile.d/conda.sh; conda activate graphmine
P=results/1115-products; mkdir -p $P/logs
python scripts/run.py --dataset products --model sign --method st --seeds 1 --device cuda:0 --out $P > $P/logs/products_sign_st.log 2>&1 &
python scripts/run_ensemble.py --dataset products --seeds 1 --experts mlp sign --no-self-train --tag _mlp-sign --device cuda:1 --out $P > $P/logs/products_coe_mlp-sign.log 2>&1 &
wait
