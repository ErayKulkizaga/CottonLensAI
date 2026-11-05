"""Export descriptive research figures from hash-bound audit CSV/JSON (no fitting)."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', required=True, type=Path)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit = json.loads((args.audit_root/'audit.json').read_text(encoding='utf-8'))
    tables = pd.read_csv(args.audit_root/'naive-summary.csv')
    rows = pd.read_csv(args.audit_root/'naive-origins.csv')
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for h, color in [(1, '#276b80'), (5, '#b66d2e')]:
        year = tables.loc[tables.horizon.eq(h) & tables.dimension.eq('year')]
        axes[0, 0].plot(year.group.astype(int), year.mae, marker='o', color=color, label=f'T+{h}')
        vol = tables.loc[tables.horizon.eq(h) & tables.dimension.eq('volatility_group')].set_index('group')
        axes[0, 1].plot(['low', 'middle', 'high'], vol.loc[['low', 'middle', 'high'], 'mae'], marker='o', color=color, label=f'T+{h}')
        errors = np.sort(rows.loc[rows.horizon.eq(h), 'naive_error'])
        axes[1, 0].plot(errors, 1-np.arange(1, len(errors)+1)/len(errors), color=color, label=f'T+{h}')
        acf = [v for v in audit['acf'] if v['horizon'] == h and v['field'] == 'signed_error']
        axes[1, 1].plot([r['lag'] for r in acf], [r['correlation'] for r in acf], marker='o', color=color, label=f'T+{h}')
    axes[0, 0].set(title='Yearly Naive MAE', ylabel='cent/lb')
    axes[0, 1].set(title='Past-defined volatility terciles', ylabel='cent/lb')
    axes[1, 0].set(title='Absolute-error tail', xlabel='cent/lb', ylabel='Empirical survival probability', yscale='log')
    axes[1, 1].set(title='Signed Naive error autocorrelation', xlabel='Recorded-observation lag', ylabel='Correlation')
    axes[1, 1].axhline(0, color='#888', linewidth=.7)
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=.15)
    fig.suptitle('CottonLensAI: reviewed 2016–2023 history; descriptive, not forward skill')
    fig.savefig(args.output/'naive-diagnostics.png', dpi=160)
    fig.savefig(args.output/'naive-diagnostics.svg')
    plt.close(fig)
    if args.result:
        result = json.loads(args.result.read_text(encoding='utf-8'))
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for group, color in [('ridge', '#b66d2e'), ('quantile', '#276b80')]:
            years = result['arms'][group]['selected']['years']
            ax.plot([r['year'] for r in years], [r['gain_pct'] for r in years], marker='o', label=group, color=color)
        ax.axhline(0, color='#888', linewidth=.8)
        ax.axhline(5, color='#666', linestyle='--', label='Fixed practical goal: 5%')
        ax.set(title='T+1 selected price-MAE gain versus Naive', xlabel='Year', ylabel='Gain (%)')
        ax.legend()
        ax.grid(alpha=.15)
        fig.savefig(args.output/'median-yearly-gains.png', dpi=160)
        fig.savefig(args.output/'median-yearly-gains.svg')
        plt.close(fig)


if __name__ == '__main__':
    main()
