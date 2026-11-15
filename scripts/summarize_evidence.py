"""Export the paired accuracy table and an optional label-free support-shift plot."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob('*_summary.json')):
        data = json.loads(path.read_text())
        for method, value in data['mean'].items():
            manifest = data['manifest']
            dataset = manifest.get('dataset', manifest.get('source_manifest', {}).get('dataset'))
            rows.append({'dataset': dataset, 'seeds': len(data['runs']),
                         'complete': data['complete'], 'method': method,
                         'validation_mean_pct': 100 * value['val'][0],
                         'test_mean_pct': 100 * value['test'][0],
                         'test_std_pct': 100 * value['test'][1]})
    if not rows:
        raise SystemExit('No experiment summaries found')
    with (args.directory / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shift_path = args.directory / 'products_unlabelled_support_shift.json'
    if shift_path.exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        shift = json.loads(shift_path.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        labels = ['Training (OOF)', 'Validation', 'Test (labels unused)']
        for axis, key, title in zip(axes, ['coverage_p10_p50_p90', 'ess_p10_p50_p90'],
                                    ['Label-source coverage', 'Effective source count']):
            values = np.array([shift[s][key] for s in ['train', 'valid', 'test']])
            axis.errorbar(values[:, 1], np.arange(3),
                          xerr=np.stack([values[:, 1] - values[:, 0], values[:, 2] - values[:, 1]]),
                          fmt='o', capsize=4)
            axis.set_yticks(np.arange(3), labels)
            axis.set_xlabel(title + ' (median, 10–90% range)')
            axis.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.directory / 'products_support_quantiles.png', dpi=160)
        fig.savefig(args.directory / 'products_support_quantiles.pdf')
    print(f'Exported {len(rows)} rows to {args.directory / "comparison.csv"}')


if __name__ == '__main__':
    main()
