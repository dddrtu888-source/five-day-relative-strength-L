"""Run the isolated five-day opportunity experiment on user-supplied daily bars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from relative_strength_l import core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asof', required=True, help='Last observable date, YYYY-MM-DD')
    parser.add_argument('--min-peers', type=int, default=500)
    parser.add_argument('--random-count', type=int, default=1000)
    parser.add_argument('--shortlist', type=int, default=15)
    parser.add_argument('--seed', type=int, default=20261112)
    args = parser.parse_args()
    if args.min_peers < 2 or args.random_count < 1 or not 1 <= args.shortlist <= args.random_count+2:
        parser.error('Invalid peer, candidate or shortlist count')
    data = pd.read_csv(args.csv, dtype={'code': str}, parse_dates=['date'])
    required = {'date', 'code', 'open', 'high', 'close', 'amount'}
    if not required <= set(data.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(data.columns))}')
    data = data.loc[data.date.le(pd.Timestamp(args.asof))].copy()
    if data.empty or data[['date', 'code']].isna().any().any() or data.duplicated(['date','code']).any():
        raise ValueError('Empty data, missing keys, or duplicate date/code')
    panels = {}
    for key in ['open', 'high', 'close', 'amount']:
        data[key] = pd.to_numeric(data[key], errors='raise')
        panels[key] = data.pivot(index='date', columns='code', values=key).sort_index().sort_index(axis=1)
    close, opens, high = (panels[key] for key in ['close','open','high'])
    returns = close.pct_change(fill_method=None)
    eligible = returns.notna() & np.isfinite(returns) & panels['amount'].gt(0) & close.gt(0)
    p, z, _ = core.distribution_scores(returns, eligible, args.min_peers)
    windows = core.score_windows(core.daily_score(p, z))
    split = core.temporal_split(close.index)
    complete = np.isfinite(windows).all(axis=2)
    if not (complete.sum(axis=1) >= 20).any():
        raise ValueError('No complete five-day universe containing at least 20 stocks')
    opportunities = core.training_opportunities(windows, opens, high, split)
    train_x = windows[split['train']]
    train_y = .7 * opportunities['peak5']
    reference = core._row_mean(np.where(np.isfinite(train_x).all(axis=2), train_y, np.nan))
    candidates = core.candidate_weights(args.random_count, args.seed)
    _, trials = core.search_training(train_x, train_y, reference, candidates, opportunities=opportunities)
    shortlist = core.training_shortlist(trials, candidates, args.shortlist)
    args.output.mkdir(parents=True, exist_ok=True)
    # Freeze training decisions before constructing any test labels.
    frozen = {'scope':'A-share price-based opportunity research', 'seed':args.seed,
              'top_n':20, 'min_peers':args.min_peers, 'candidate_count':len(candidates),
              'training_start':str(split['start'].date()),
              'last_training_signal':str(close.index[split['train']][-1].date()),
              'test_start':str(split['test_start'].date()), 'test_end':str(split['end'].date()),
              'shortlist':shortlist, 'uses_future_peak_for_labels':True,
              'executable_returns_validated':False}
    (args.output/'frozen_weights.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2), 'utf-8')
    trials.to_csv(args.output/'training_trials.csv', index=False, encoding='utf-8-sig')
    labels = core.outcome_labels(close, opens, high, peak_fraction=.7)
    metrics = ['peak5','peak10','hit3','next_day','entry_day']
    rows = []
    schemes = [(s['name'], s['weights_oldest_to_latest']) for s in shortlist]
    schemes += [('五日等权基准', [.2]*5), ('同日市场篮子', None)]
    for name, weights in schemes:
        picks = None if weights is None else core.top_indices(np.einsum('tnd,d->tn',windows,weights,optimize=False),20)
        for phase, mask in [('训练期',split['train']),('时间留出测试',split['test'])]:
            for day in np.flatnonzero(mask):
                if day+1 >= len(close):
                    continue
                ids = np.flatnonzero(complete[day]) if picks is None else picks[day][picks[day]>=0]
                if len(ids) == 0:
                    continue
                row = {'date':str(close.index[day].date()),'phase':phase,'model':name,'selected':len(ids)}
                for key in metrics:
                    values = labels[key].iloc[day].to_numpy()[ids]
                    values = values[np.isfinite(values)]
                    row[key] = float(np.mean(values>0) if key in ['next_day','entry_day'] else np.mean(values)) if len(values) else np.nan
                    row[key+'_n'] = len(values)
                rows.append(row)
    periods = pd.DataFrame(rows)
    periods.to_csv(args.output/'periods.csv', index=False, encoding='utf-8-sig')
    periods.groupby(['phase','model'], sort=False)[metrics].mean().to_csv(args.output/'summary.csv', encoding='utf-8-sig')
    print(f'Saved {len(shortlist)} frozen schemes and date-weighted opportunity summaries to {args.output}')


if __name__ == '__main__':
    main()
