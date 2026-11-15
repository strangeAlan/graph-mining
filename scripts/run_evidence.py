"""Paired, cached experiment for evidence-conditioned expert fusion.

Example: python scripts/run_evidence.py --dataset citeseer --out results/1115-evidence
Each expert cache is written atomically immediately after fitting, and fingerprinted
by its code/config/split. Gate selection is saved BEFORE any test evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from graphmine.data import accuracy, load_graph
from graphmine.ensemble import cooperate, fit_theta, stratified_folds
from graphmine.evidence import apply_gate, diffusion_evidence, fit_gate
from graphmine.prop import gcn_adj, hop_features, mean_adj
from graphmine.train import fit
from run import model_config


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def code_hash(names):
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}


def expert_cache(g, ctx, experts, cfg, spec, seed, cache):
    train, y = ctx['split']['train'], ctx['y']
    folds = spec['products_folds'] if g.name == 'ogbn-products' else spec['folds']
    fold = stratified_folds(y, train, folds, seed)
    split_hash = {k: hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k, v in ctx['split'].items()}
    raw_code = code_hash(['src/graphmine/models.py', 'src/graphmine/train.py',
                          'src/graphmine/data.py', 'src/graphmine/prop.py', 'src/graphmine/ensemble.py'])
    oofs, fulls, fingerprints, times = [], [], {}, {}
    all_stats, oof_stats = None, None
    for expert in experts:
        meta = {'dataset': g.name, 'seed': seed, 'folds': folds, 'split': split_hash,
                'expert': expert, 'config': cfg[expert], 'code': raw_code, 'adjacency_layout': 'csr',
                'shape': [g.num_nodes, g.x.size(1), g.edge_index.size(1)]}
        if expert == 'lp':
            meta.update(diffusion=spec['diffusion'], evidence_code=code_hash(['src/graphmine/evidence.py']))
        key = fingerprint(meta)
        fingerprints[expert] = key
        path = cache / f'{g.name}_{seed}_{expert}_{key}.pt'
        if path.exists():
            record = torch.load(path, map_location='cpu', weights_only=True)
            print(f'seed {seed} {expert}: cached', flush=True)
        else:
            start = time.time()
            oof = torch.empty(len(train), g.num_classes)
            st_oof = torch.empty(len(train), 3) if expert == 'lp' else None
            for f in range(folds + 1):
                source = train[fold != f] if f < folds else train
                if expert == 'lp':
                    pr, st = diffusion_evidence(ctx['adj'], y, source, g.num_classes,
                                                **spec['diffusion'], seed=seed + 9173)
                else:
                    fit_seed = seed + 100 * (f + 1) if f < folds else seed
                    pr = fit(g, ctx, expert, cfg[expert], source, y[source], fit_seed)[0]
                if f < folds:
                    oof[(fold == f).cpu()] = pr[train[fold == f]].cpu()
                    if expert == 'lp':
                        st_oof[(fold == f).cpu()] = st[train[fold == f]].cpu()
                else:
                    full = pr.cpu()
                    if expert == 'lp':
                        all_stats = st.cpu()
                del pr
                print(f'seed {seed} {expert}: fit {f + 1}/{folds + 1} ({time.time() - start:.0f}s)', flush=True)
            record = {'oof': oof, 'full': full, 'time_s': time.time() - start, 'meta': meta}
            if expert == 'lp':
                record.update(stats=all_stats, oof_stats=st_oof)
            tmp = path.with_suffix('.tmp')
            torch.save(record, tmp)
            tmp.replace(path)
        oofs.append(record['oof'])
        fulls.append(record['full'])
        times[expert] = record['time_s']
        if expert == 'lp':
            all_stats, oof_stats = record['stats'], record['oof_stats']
    return torch.stack(oofs), torch.stack(fulls), oof_stats, all_stats, fingerprints, times


def fit_legacy(oof, labels, val_full, val_labels, grid):
    acc = (oof.argmax(2) == labels[None]).float().mean(1)
    best = {'stack': None, 'coe': None}
    for gamma in [0.0] + grid['gamma']:
        name = 'stack' if gamma == 0 else 'coe'
        for reg in grid['reg']:
            theta, _ = fit_theta(oof, labels, acc, gamma, reg, grid['epochs'], grid['lr'])
            logits = cooperate(val_full, theta)
            val = (logits.argmax(1) == val_labels).float().mean().item()
            # Match the old implementation's first-on-tie selection.
            if best[name] is None or val > best[name]['val']:
                best[name] = {'val': val, 'theta': theta, 'gamma': gamma, 'reg': reg}
    return best


def diagnostics(g, pred, stats, weights, experts, full):
    """Post-selection test diagnostics; never input to fitting or selection."""
    test = g.split['test'].cpu()
    ess = stats[test, 1]
    groups = {'no_signal': ess == 0, 'ess_1_2': (ess > 0) & (ess <= 2),
              'ess_2_8': (ess > 2) & (ess <= 8), 'ess_gt8': ess > 8}
    out = {}
    for name, mask in groups.items():
        ids = test[mask]
        if not len(ids):
            continue
        item = {'n': len(ids), 'acc': accuracy(g, pred, ids),
                'expert_acc': {e: accuracy(g, full[j].argmax(1), ids) for j, e in enumerate(experts)}}
        if weights is not None:
            item['mean_weights'] = dict(zip(experts, weights[ids].mean(0).tolist()))
        out[name] = item
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['cora', 'citeseer', 'products'])
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--spec', default=str(ROOT / 'configs/evidence.yaml'))
    p.add_argument('--config', default=str(ROOT / 'configs/default.yaml'))
    p.add_argument('--skip-legacy', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    spec = yaml.safe_load(Path(args.spec).read_text())
    seeds = args.seeds if args.seeds is not None else spec['seeds']
    out = Path(args.out).resolve()
    cache = out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    conf = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    experts = spec['products_experts'] if args.dataset == 'products' else spec['experts']
    cfg = {e: model_config(args.config, args.dataset, e) if e != 'lp' else conf['lp'] for e in experts}
    g = load_graph(args.dataset, spec['split'])
    device = torch.device(args.device)
    ctx = {'x': g.x.to(device), 'y': g.y.to(device), 'edge_index': g.edge_index.to(device),
           'split': {k: v.to(device) for k, v in g.split.items()}}
    # CSR improves sparse propagation throughput; all paired controls share
    # these same tensors and cached predictions (no bitwise-determinism claim).
    ctx['adj'] = gcn_adj(ctx['edge_index'], g.num_nodes).to_sparse_csr()
    if 'sage' in experts:
        ctx['adj_mean'] = mean_adj(ctx['edge_index'], g.num_nodes).to_sparse_csr()
    if 'sign' in experts:
        ctx['feats'] = hop_features(ctx['adj'], ctx['x'], cfg['sign']['hops'])
    degree = torch.bincount(g.edge_index[0], minlength=g.num_nodes).float()
    manifest = {'base_commit': spec['base_commit'], 'head': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'spec': spec, 'seeds': seeds,
        'dataset': g.name, 'experts': experts, 'config': cfg, 'device': args.device,
        'code_hash': code_hash(['scripts/run_evidence.py', 'src/graphmine/evidence.py']),
        'torch': str(torch.__version__), 'adjacency_layout': 'csr',
        'split_hashes': {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in g.split.items()},
        'split_sizes': {k: len(v) for k, v in g.split.items()}, 'skip_legacy': args.skip_legacy}
    atomic_json(out / f'{g.name}_manifest.json', manifest)
    records = []
    for seed in seeds:
        start = time.time()
        oof, full, ts, fs, fingerprints, times = expert_cache(g, ctx, experts, cfg, spec, seed, cache)
        tr, va = g.split['train'], g.split['valid']
        oof_d, vp = oof.to(device), full[:, va].to(device)
        yt, yv = g.y[tr].to(device), g.y[va].to(device)
        acc = (oof.argmax(2) == g.y[tr][None]).float().mean(1)
        # All selection finishes before test predictions are scored below.
        selections, fitted = {}, {}
        for mode in ['global', 'confidence', 'evidence']:
            fitted[mode] = fit_gate(oof_d, yt, ts.to(device), degree[tr].to(device), vp, yv,
                                    fs[va].to(device), degree[va].to(device), experts.index('lp'), mode,
                                    **spec['gate'], seed=seed)
            selections[mode] = fitted[mode][3]
            print(f'seed {seed} {mode}: validation {selections[mode]}', flush=True)
        legacy = {} if args.skip_legacy else fit_legacy(oof_d, yt, vp, yv, conf['ensemble']['grid'])
        for key, value in legacy.items():
            selections[key] = {k: v for k, v in value.items() if k != 'theta'}
        expert_val = [(vp[j].argmax(1) == yv).float().mean().item() for j in range(len(experts))]
        best_expert = int(np.argmax(expert_val))
        selections['best_expert'] = experts[best_expert]
        atomic_json(out / f'{g.name}_seed{seed}_selection.json', selections)
        # Persist selected routers for replay and interpretation.
        torch.save({mode: {'state': fitted[mode][0].state_dict(), 'center': fitted[mode][1],
                          'scale': fitted[mode][2]} for mode in fitted},
                   cache / f'{g.name}_seed{seed}_routers.pt')
        record = {'seed': seed, 'selection': selections, 'cache_fingerprints': fingerprints,
                  'expert_fit_seconds': times, 'methods': {}}

        def evaluate(key, probs, weights=None, detailed=False):
            pred = probs.argmax(1)
            r = {'val': accuracy(g, pred, va), 'test': accuracy(g, pred, g.split['test'])}
            if detailed:
                r['groups'] = diagnostics(g, pred, fs, weights, experts, full)
            record['methods'][key] = r

        for j, expert in enumerate(experts):
            evaluate('expert_' + expert, full[j])
        evaluate('best_expert', full[best_expert])
        evaluate('mean_vote', full.mean(0))
        evaluate('acc_vote', (full * acc[:, None, None]).sum(0))
        for mode in fitted:
            gate, center, scale, _ = fitted[mode]
            probs, weights = apply_gate(gate, full, fs, degree, center, scale, mode, device)
            evaluate(mode, probs, weights, detailed=True)
        for name, value in legacy.items():
            pred = torch.cat([cooperate(full[:, i:i + 100000].to(device), value['theta']).argmax(1).cpu()
                              for i in range(0, g.num_nodes, 100000)])
            record['methods'][name] = {'val': accuracy(g, pred, va), 'test': accuracy(g, pred, g.split['test'])}
        record['wall_seconds'] = time.time() - start
        records.append(record)
        atomic_json(out / f'{g.name}_seed{seed}.json', record)
        means = {name: {part: [float(np.mean([r['methods'][name][part] for r in records])),
                                    float(np.std([r['methods'][name][part] for r in records]))]
                        for part in ['val', 'test']} for name in record['methods']}
        atomic_json(out / f'{g.name}_summary.json', {'manifest': manifest, 'complete': seed == seeds[-1],
                                                   'runs': records, 'mean': means})
        print(f"seed {seed} complete: " + ' '.join(f"{k}={100*v['test']:.2f}" for k, v in record['methods'].items()), flush=True)
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
