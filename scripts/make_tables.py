"""Build the LaTeX tables of the report from result JSON files (missing entries print as --).

  python scripts/make_tables.py   -> report/tables/*.tex
"""
import json
import os
import os.path as osp

ROOT = osp.join(osp.dirname(osp.abspath(__file__)), '..')
OUT = osp.join(ROOT, 'report', 'tables')

# Which run directory holds the final number of each dataset (see notes/LOG.md).
PLANETOID = 'results/1114-planetoid'
PRODUCTS = ['results/1114-products', 'results/1113-products']  # first match wins
DATASETS = [('cora', 'Cora'), ('citeseer', 'Citeseer'), ('ogbn-products', 'ogbn-products')]


def load(dataset, name):
    dirs = PRODUCTS if dataset == 'ogbn-products' else [PLANETOID]
    for d in dirs:
        f = osp.join(ROOT, d, f'{dataset}_{name}.json')
        if osp.exists(f):
            with open(f) as fh:
                return json.load(fh)
    return None


def fmt(mean, std=None, bold=False):
    if mean is None:
        return '--'
    s = f'{100 * mean:.1f}' + (f'\\,$\\pm$\\,{100 * std:.1f}' if std is not None else '')
    return f'\\textbf{{{s}}}' if bold else s


def cell(dataset, name, key='test'):
    r = load(dataset, name)
    if r is None or f'{key}_mean' not in r:
        return None, None
    std = r[f'{key}_std'] if len(r['runs']) > 1 else None
    return r[f'{key}_mean'], std


def baseline_table():
    rows = [('MLP', 'mlp_base'), ('Label propagation', 'lp_lp'), ('GCN', 'gcn_base'), ('GAT', 'gat_base'),
            ('APPNP', 'appnp_base'), ('GraphSAGE', 'sage_base'), ('SIGN', 'sign_base')]
    lines = [r'\begin{tabular}{l cc cc cc}', r'\toprule',
             r' & \multicolumn{2}{c}{Cora} & \multicolumn{2}{c}{Citeseer} & \multicolumn{2}{c}{ogbn-products} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
             r'Model & plain & +C\&S & plain & +C\&S & plain & +C\&S \\', r'\midrule']
    best = {}
    for ds, _ in DATASETS:
        vals = [cell(ds, n)[0] or 0 for _, n in rows] + [cell(ds, n, 'cs_test')[0] or 0 for _, n in rows]
        best[ds] = max(vals)
    for label, name in rows:
        cells = []
        for ds, _ in DATASETS:
            if ds == 'ogbn-products' and name == 'gat_base':
                cells += ['--', '--']
                continue
            m, s = cell(ds, name)
            cells.append(fmt(m, s, m is not None and abs(m - best[ds]) < 1e-9))
            if name == 'lp_lp':
                cells.append('--')
            else:
                m, s = cell(ds, name, 'cs_test')
                cells.append(fmt(m, s, m is not None and abs(m - best[ds]) < 1e-9))
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(osp.join(OUT, 'tab_baselines.tex'), 'w') as f:
        f.write(baseline_table())
    print('wrote', OUT)


if __name__ == '__main__':
    main()
