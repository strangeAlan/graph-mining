"""One targeted capacity-matched check, reusing the frozen evidence experiment.

The stacking class matrices are reconstructed at the already selected settings.
Only node-wise expert scales are fitted. Epoch zero is the unmodified stacking
baseline. 'context' removes ESS alone, retaining all other evidence features.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from graphmine.data import accuracy, load_graph
from graphmine.ensemble import cooperate, fit_theta
from graphmine.evidence import apply_gate, fit_gate
from run_evidence import atomic_json, code_hash, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'products'], required=True)
    parser.add_argument('--source', default='results/1115-evidence', type=Path)
    parser.add_argument('--out', default='results/1115-conditional-coe', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    name = 'ogbn-products' if args.dataset == 'products' else args.dataset
    original = json.loads((args.source / f'{name}_summary.json').read_text())
    spec, experts = original['manifest']['spec'], original['manifest']['experts']
    grid = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())[args.dataset]['ensemble']['grid']
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.source / 'cache'
    g = load_graph(args.dataset, spec['split'])
    tr, va, te = (g.split[k] for k in ['train', 'valid', 'test'])
    degree = torch.bincount(g.edge_index[0], minlength=g.num_nodes).float()
    manifest = {'source_manifest': original['manifest'], 'class_matrices': 'frozen stacking',
                'modes': ['global', 'confidence', 'context', 'evidence'], 'gate': spec['gate'],
                'code_hash': code_hash(['src/graphmine/evidence.py', 'scripts/run_conditional_coe.py']),
                'stop': 'One aligned check only; no further gate variants if validation does not improve.'}
    atomic_json(args.out / f'{name}_manifest.json', manifest)
    records = []
    for old in original['runs']:
        start, seed = time.time(), old['seed']
        oofs, fulls = [], []
        for expert in experts:
            key = old['cache_fingerprints'][expert]
            record = torch.load(cache / f'{name}_{seed}_{expert}_{key}.pt', map_location='cpu', weights_only=True)
            # Fingerprint identifies the frozen expert generator; use exactly
            # these cached predictions, not predictions from newly edited code.
            assert record['meta']['seed'] == seed and record['meta']['expert'] == expert
            for part, ids in g.split.items():
                assert record['meta']['split'][part] == hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            oofs.append(record['oof'])
            fulls.append(record['full'])
            if expert == 'lp':
                ts, fs = record['oof_stats'], record['stats']
        oof, full = torch.stack(oofs).to(args.device), torch.stack(fulls)
        vp, yt, yv = full[:, va].to(args.device), g.y[tr].to(args.device), g.y[va].to(args.device)
        acc = (oof.argmax(2) == yt[None]).float().mean(1)
        chosen = old['selection']['stack']
        theta, _ = fit_theta(oof, yt, acc, 0, chosen['reg'], grid['epochs'], grid['lr'])
        validation = (cooperate(vp, theta).argmax(1) == yv).float().mean().item()
        if abs(validation - chosen['val']) > 1e-7:
            raise RuntimeError(f'Stacking reconstruction changed validation: {validation} vs {chosen}')
        fitted, selection = {}, {'stack': chosen}
        for mode in manifest['modes']:
            fitted[mode] = fit_gate(oof, yt, ts.to(args.device), degree[tr].to(args.device), vp, yv,
                                    fs[va].to(args.device), degree[va].to(args.device), experts.index('lp'),
                                    mode, **spec['gate'], seed=seed, theta=theta)
            selection[mode] = fitted[mode][3]
            print(f'{name} seed {seed} {mode}: {selection[mode]}', flush=True)
        atomic_json(args.out / f'{name}_seed{seed}_selection.json', selection)
        methods = {'stack': old['methods']['stack']}
        for mode in fitted:
            gate, center, scale, _ = fitted[mode]
            probs, weights = apply_gate(gate, full, fs, degree, center, scale, mode, args.device)
            pred = probs.argmax(1)
            methods[mode] = {'val': accuracy(g, pred, va), 'test': accuracy(g, pred, te)}
            if mode == 'evidence':
                methods[mode]['groups'] = diagnostics(g, pred, fs, weights, experts, full)
        record = {'seed': seed, 'selection': selection, 'methods': methods, 'seconds': time.time() - start}
        records.append(record)
        means = {m: {part: [float(np.mean([r['methods'][m][part] for r in records])),
                            float(np.std([r['methods'][m][part] for r in records]))]
                     for part in ['val', 'test']} for m in methods}
        atomic_json(args.out / f'{name}_summary.json', {'manifest': manifest, 'complete': len(records) == len(original['runs']),
                                                       'runs': records, 'mean': means})
        print(f'{name} seed {seed} final: ' + ' '.join(f"{m}={100*v['test']:.3f}" for m,v in methods.items()), flush=True)


if __name__ == '__main__':
    main()
