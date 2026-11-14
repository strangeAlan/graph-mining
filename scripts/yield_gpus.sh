#!/bin/bash
# Free both GPUs for the cv-3Drecon dense stage: as soon as its finish script starts evaluating 1113-allviews
# (the step right before the GPU-heavy run_da3ba --dense), stop every graph-mining job and do not return.
until pgrep -f "evaluate.py --no-da3-protocol --workers 24 configs/1113-allviews" > /dev/null \
   || pgrep -f "run_da3ba.py" > /dev/null; do sleep 10; done
for pat in queue_products_1114.sh run_products.sh run_planetoid_coe.sh run_planetoid.sh scripts/run_ensemble.py scripts/run.py; do
  for pid in $(pgrep -f "$pat"); do [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null; done
done
echo "$(date -u '+%m-%d %H:%M') graph-mining jobs stopped for cv-3Drecon dense stage"
