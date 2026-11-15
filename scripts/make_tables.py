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


COE = {'cora': 'results/1114-planetoid-coe/cora_coe.json', 'citeseer': 'results/1114-planetoid-coe/citeseer_coe.json',
       'ogbn-products': 'results/1114-products-coe/ogbn-products_coe.json'}
SINGLE_ST = {'cora': 'results/1114-planetoid/cora_{m}_st.json', 'citeseer': 'results/1114-planetoid/citeseer_{m}_st.json',
             'ogbn-products': 'results/1115-products/ogbn-products_{m}_st.json'}
NAMES = {'mlp': 'MLP', 'lp': 'LP', 'gcn': 'GCN', 'gat': 'GAT', 'appnp': 'APPNP', 'sage': 'SAGE', 'sign': 'SIGN'}


def coe_table():
    """CoE vs single experts and voting, before and after ensemble self-training. The single expert is the one
    with the best mean validation accuracy (what one would deploy); the best on test is shown for reference."""
    cols = {}
    for ds, _ in DATASETS:
        with open(osp.join(ROOT, COE[ds])) as f:
            r = json.load(f)
        mean, experts = r['mean'], r['config']['ensemble']['experts']
        get = lambda st, key: (mean[f'{st}/{key}/test'][0], mean[f'{st}/{key}/test'][1] if len(r['runs']) > 1 else None)
        c = {}
        for st in ('stage0', 'stage1'):
            val_best = max(experts, key=lambda e: mean[f'{st}/expert_{e}/val'][0])
            test_best = max(experts, key=lambda e: mean[f'{st}/expert_{e}/test'][0])
            c[st + '/val_best'] = (*get(st, f'expert_{val_best}'), NAMES[val_best])
            c[st + '/test_best'] = (*get(st, f'expert_{test_best}'), NAMES[test_best])
            for key in ('mean_vote', 'acc_vote', 'stack', 'coe'):
                c[f'{st}/{key}'] = (*get(st, key), None)
        base = max(experts, key=lambda e: mean[f'stage0/expert_{e}/val'][0])
        f_st = osp.join(ROOT, SINGLE_ST[ds].format(m=base))
        if osp.exists(f_st):
            with open(f_st) as f:
                rs = json.load(f)
            if ds == 'ogbn-products':
                # One round, as in ensemble self-training: on this split validation accuracy does not track test
                # accuracy across self-training rounds, so the validation-chosen stage is always 0 (notes/LOG.md 11/15).
                c['single_st'] = (rs['runs'][0]['stages'][1]['test'], None, NAMES[base] + '$^\\dagger$')
            else:
                c['single_st'] = (rs['test_mean'], rs['test_std'] if len(rs['runs']) > 1 else None, NAMES[base])
        else:
            c['single_st'] = (None, None, NAMES[base])
        cols[ds] = c
    rows = [(r'\multicolumn{4}{l}{\textit{Without self-training}}', None),
            ('Best expert, chosen on validation', 'stage0/val_best'),
            ('Best expert on test (reference)', 'stage0/test_best'),
            ('Mean voting', 'stage0/mean_vote'), ('Accuracy-weighted voting', 'stage0/acc_vote'),
            (r'Stacking ($\gamma=0$)', 'stage0/stack'), ('CoE', 'stage0/coe'),
            (r'\midrule\multicolumn{4}{l}{\textit{With self-training}}', None),
            ('Single-model self-training', 'single_st'),
            ('Best expert, ensemble pseudo-labels', 'stage1/val_best'),
            ('CoE, ensemble pseudo-labels', 'stage1/coe')]
    lines = [r'\begin{tabular}{l ccc}', r'\toprule', r'Method & Cora & Citeseer & ogbn-products \\', r'\midrule']
    for label, key in rows:
        if key is None:
            lines.append(label + r' \\')
            continue
        cells = []
        for ds, _ in DATASETS:
            m, sd, name = cols[ds][key]
            txt = fmt(m, sd)
            cells.append(f'{txt} ({name})' if name and m is not None else txt)
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


RANDOM60 = {'cora': 'results/1115-planetoid-random60/cora_coe_random60.json',
            'citeseer': 'results/1115-planetoid-random60/citeseer_coe_random60.json'}


def random60_table():
    """CoE on Cora / Citeseer with a random 60/20/20 split (stage 0 only)."""
    rows = [('Best expert, chosen on validation', 'val_best'), ('Best expert on test (reference)', 'test_best'),
            ('Accuracy-weighted voting', 'acc_vote'), (r'Stacking ($\gamma=0$)', 'stack'), ('CoE', 'coe')]
    cols = {}
    for ds, path in RANDOM60.items():
        if not osp.exists(osp.join(ROOT, path)):
            cols[ds] = None
            continue
        with open(osp.join(ROOT, path)) as f:
            r = json.load(f)
        mean, experts = r['mean'], r['config']['ensemble']['experts']
        vb = max(experts, key=lambda e: mean[f'stage0/expert_{e}/val'][0])
        tb = max(experts, key=lambda e: mean[f'stage0/expert_{e}/test'][0])
        t = lambda key: mean[f'stage0/{key}/test']
        cols[ds] = {'val_best': (*t(f'expert_{vb}'), NAMES[vb]), 'test_best': (*t(f'expert_{tb}'), NAMES[tb]),
                    **{k: (*t(k), None) for k in ('acc_vote', 'stack', 'coe')}}
    lines = [r'\begin{tabular}{l cc}', r'\toprule', r'Method & Cora & Citeseer \\', r'\midrule']
    for label, key in rows:
        cells = []
        for ds in RANDOM60:
            if cols[ds] is None:
                cells.append('--')
                continue
            m, sd, name = cols[ds][key]
            cells.append(fmt(m, sd) + (f' ({name})' if name else ''))
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(osp.join(OUT, 'tab_baselines.tex'), 'w') as f:
        f.write(baseline_table())
    with open(osp.join(OUT, 'tab_coe.tex'), 'w') as f:
        f.write(coe_table())
    with open(osp.join(OUT, 'tab_random60.tex'), 'w') as f:
        f.write(random60_table())
    print('wrote', OUT)


if __name__ == '__main__':
    main()
