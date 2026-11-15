"""Cooperation of Experts on one dataset over several seeds.

  python scripts/run_ensemble.py --dataset cora --seeds 10 --out results/<run>

Per seed it records every expert, mean / accuracy-weighted voting, learned stacking (gamma = 0),
CoE (large margin), CoE + C&S, and the same after one round of ensemble self-training.
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
from graphmine.ensemble import combine, train_experts  # noqa: E402
from graphmine.prop import gcn_adj, hop_features, mean_adj  # noqa: E402
from graphmine.train import cs_search  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from run import model_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--config', default=osp.join(osp.dirname(osp.abspath(__file__)), '..', 'configs', 'default.yaml'))
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--experts', nargs='*', help='override the expert list of the config')
    p.add_argument('--no-self-train', action='store_true', help='stage 0 only')
    p.add_argument('--tag', default='', help='suffix of the output file name')
    p.add_argument('--split', default='public', choices=['public', 'random60'])
    args = p.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)[args.dataset]
    ens = conf['ensemble']
    if args.experts:
        ens['experts'] = args.experts
    if args.no_self_train:
        ens['self_train'] = False
    cfg = {e: model_config(args.config, args.dataset, e) for e in ens['experts'] if e != 'lp'}
    cfg['lp'], cfg['st'] = conf['lp'], {'weight': ens['pseudo_weight']}

    t0 = time.time()
    g = load_graph(args.dataset, args.split)
    dev = torch.device(args.device)
    ctx = {'x': g.x.to(dev), 'y': g.y.to(dev), 'edge_index': g.edge_index.to(dev),
           'split': {k: v.to(dev) for k, v in g.split.items()}}
    ctx['adj'] = gcn_adj(ctx['edge_index'], g.num_nodes)
    if 'sage' in cfg:
        ctx['adj_mean'] = mean_adj(ctx['edge_index'], g.num_nodes)
    if 'sign' in cfg:
        ctx['feats'] = hop_features(ctx['adj'], ctx['x'], cfg['sign']['hops'])
    print(f'[{args.dataset}] loaded + preprocessed in {time.time() - t0:.1f}s', flush=True)

    split, y = ctx['split'], ctx['y']
    ev = lambda probs, part: accuracy(g, probs.argmax(1), split[part])

    def evaluate(full, res):
        out = {f'expert_{e}': {'val': ev(full[j], 'valid'), 'test': ev(full[j], 'test')}
               for j, e in enumerate(ens['experts'])}
        for key in ('mean_vote', 'acc_vote', 'stack', 'coe'):
            out[key] = {'val': ev(res[key], 'valid'), 'test': ev(res[key], 'test')}
        out['coe'].update(gamma=res['coe_gamma'], reg=res['coe_reg'])
        cs = cs_search(g, ctx, res['coe'], conf['cs_grid'])
        out['coe_cs'] = {'val': cs['cs_val'], 'test': cs['cs_test'], 'alpha1': cs['cs_alpha1'], 'alpha2': cs['cs_alpha2']}
        out['expert_oof_acc'] = res['expert_oof_acc']
        return out

    os.makedirs(args.out, exist_ok=True)

    def save(runs, final):
        """Written after every seed, so finished seeds survive an interrupted run."""
        summary = {'dataset': g.name, 'config': conf, 'complete': final, 'runs': runs, 'mean': {}}
        for stage in ('stage0', 'stage1'):
            if stage not in runs[0]:
                continue
            for key, v in runs[0][stage].items():
                if isinstance(v, dict):
                    for part in ('val', 'test'):
                        vals = [r[stage][key][part] for r in runs]
                        summary['mean'][f'{stage}/{key}/{part}'] = [float(np.mean(vals)), float(np.std(vals))]
        with open(osp.join(args.out, f'{g.name}_coe{args.tag}.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        return summary

    runs = []
    for seed in range(args.seeds):
        ts = time.time()
        oof, full = train_experts(g, ctx, ens['experts'], cfg, seed, ens['folds'])
        res = combine(g, ctx, oof, full, ens['grid'])
        rec = {'seed': seed, 'stage0': evaluate(full, res)}
        print(f"seed {seed} stage0 done: coe {rec['stage0']['coe']['test']:.4f} | {time.time() - ts:.0f}s", flush=True)
        if ens.get('self_train', False):
            # One round of ensemble self-training: confident CoE predictions on non-validation
            # unlabelled nodes become pseudo-labels for every expert, then the experts are re-combined.
            conf_, pred = res['coe'].max(1)
            cand = torch.ones(g.num_nodes, dtype=torch.bool, device=dev)
            cand[split['train']] = False
            cand[split['valid']] = False
            pidx = (cand & (conf_ >= ens['tau'])).nonzero().view(-1)
            oof, full = train_experts(g, ctx, ens['experts'], cfg, seed, ens['folds'], pidx, pred[pidx])
            res = combine(g, ctx, oof, full, ens['grid'])
            rec['stage1'] = evaluate(full, res)
            rec['stage1'].update(n_pseudo=pidx.numel(),
                                 pseudo_precision=(pred[pidx] == y[pidx]).float().mean().item())  # diagnostic only
        rec['time_s'] = time.time() - ts
        runs.append(rec)
        save(runs, False)
        line = ' '.join(f"{k} {v['test']:.4f}" for k, v in rec['stage0'].items() if isinstance(v, dict))
        print(f'seed {seed} stage0: {line}', flush=True)
        if 'stage1' in rec:
            s1 = rec['stage1']
            print(f"seed {seed} stage1: coe {s1['coe']['test']:.4f} coe_cs {s1['coe_cs']['test']:.4f} "
                  f"n_pseudo {s1['n_pseudo']} prec {s1['pseudo_precision']:.3f} | {rec['time_s']:.0f}s", flush=True)

    summary = save(runs, True)
    for k, (m, sd) in summary['mean'].items():
        if k.endswith('/test'):
            print(f'{k:32s} {100 * m:.2f} ± {100 * sd:.2f}')


if __name__ == '__main__':
    main()
