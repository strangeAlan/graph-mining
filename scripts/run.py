"""Run one (dataset, model, method) over several seeds and write a JSON summary.

  python scripts/run.py --dataset cora --model gcn --method lpst --seeds 10 --out results/<run>
  method: base | st (confidence self-training) | lpst (confidence + LP agreement) | lp (label propagation only)
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..', 'src'))
from graphmine.data import accuracy, load_graph  # noqa: E402
from graphmine.prop import gcn_adj, hop_features, label_propagation, mean_adj  # noqa: E402
from graphmine.train import run_seed  # noqa: E402


def model_config(path, dataset, model):
    with open(path) as f:
        conf = yaml.safe_load(f)[dataset]
    cfg = dict(conf['common'])
    cfg.update(conf.get('models', {}).get(model, {}))
    for k in ('st', 'cs_grid', 'lp'):
        cfg[k] = conf[k]
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--model', default='gcn')
    p.add_argument('--method', default='base', choices=['base', 'st', 'lpst', 'lp'])
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--config', default=osp.join(osp.dirname(osp.abspath(__file__)), '..', 'configs', 'default.yaml'))
    p.add_argument('--set', nargs='*', default=[], help='override, e.g. st.tau=0.8 lr=0.005')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    cfg = model_config(args.config, args.dataset, args.model)
    for kv in args.set:
        k, v = kv.split('=')
        d = cfg
        for part in k.split('.')[:-1]:
            d = d[part]
        d[k.split('.')[-1]] = yaml.safe_load(v)

    t0 = time.time()
    g = load_graph(args.dataset)
    dev = torch.device(args.device)
    ctx = {'x': g.x.to(dev), 'y': g.y.to(dev), 'edge_index': g.edge_index.to(dev),
           'split': {k: v.to(dev) for k, v in g.split.items()}}
    ctx['adj'] = gcn_adj(ctx['edge_index'], g.num_nodes)
    if args.model == 'sage':
        ctx['adj_mean'] = mean_adj(ctx['edge_index'], g.num_nodes)
    if args.model == 'sign':
        ctx['feats'] = hop_features(ctx['adj'], ctx['x'], cfg['hops'])
    if args.method in ('st', 'lpst', 'lp'):
        ctx['lp'] = label_propagation(ctx['adj'], ctx['y'], ctx['split']['train'], g.num_classes,
                                      cfg['lp']['alpha'], cfg['lp']['iters'])
    print(f'[{args.dataset}] loaded + preprocessed in {time.time() - t0:.1f}s', flush=True)

    if args.method == 'lp':  # deterministic, a single "seed"
        pred = ctx['lp'].argmax(1)
        runs = [{'seed': 0, 'val': accuracy(g, pred, ctx['split']['valid']),
                 'test': accuracy(g, pred, ctx['split']['test'])}]
    else:
        runs = []
        for seed in range(args.seeds):
            r = run_seed(g, ctx, args.model, args.method, cfg, seed)
            runs.append(r)
            print(f"seed {seed}: val {r['val']:.4f} test {r['test']:.4f} | +C&S val {r['cs_val']:.4f} "
                  f"test {r['cs_test']:.4f} | stage {r['chosen_stage']} | {r['time_s']:.0f}s", flush=True)

    summary = {'dataset': g.name, 'model': args.model, 'method': args.method, 'config': cfg,
               'overrides': args.set, 'runs': runs}
    for key in ('val', 'test', 'cs_val', 'cs_test'):
        vals = [r[key] for r in runs if key in r]
        if vals:
            summary[key + '_mean'], summary[key + '_std'] = float(np.mean(vals)), float(np.std(vals))
    print({k: round(v, 4) for k, v in summary.items() if k.endswith(('_mean', '_std'))}, flush=True)
    os.makedirs(args.out, exist_ok=True)
    name = f'{g.name}_{args.model if args.method != "lp" else "lp"}_{args.method}'
    with open(osp.join(args.out, name + '.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
