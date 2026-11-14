"""Five-day relative-strength research primitives extracted from the original experiment.

Only market-relative scoring, temporal weights and opportunity labels are included.
Peak labels are hindsight diagnostics, not executable trading returns.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import pandas as pd

COST = 0.001

def distribution_scores(returns: pd.DataFrame, eligible: pd.DataFrame, min_peers: int = 500):
    """Leave-one-out midrank percentiles and sample-SD standardized deviations."""
    x = returns.where(eligible).to_numpy(float)
    count = np.isfinite(x).sum(axis=1)[:, None]
    peers = count - 1
    total = np.nansum(x, axis=1)[:, None]
    square = np.nansum(x*x, axis=1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (total-x)/peers
        variance = (square-x*x-peers*mean*mean)/(peers-1)
        sd = np.sqrt(np.maximum(variance, 0))
        z = (x-mean)/sd
        # rank - 1 = #lower + .5*(#tied - 1), exactly excluding self.
        ranks = pd.DataFrame(x).rank(axis=1, method="average").to_numpy()
        percentiles = (ranks-1)/peers
    invalid = (peers < min_peers) | ~np.isfinite(x)
    z[invalid | (sd < 1e-6)] = np.nan
    percentiles[invalid] = np.nan
    l = x-mean; l[invalid] = np.nan
    wrap = lambda v: pd.DataFrame(v, index=returns.index, columns=returns.columns)
    return wrap(percentiles), wrap(z), wrap(l)

def forward_returns(opens: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Next open to open H sessions later; missing intermediate opens stay missing."""
    label = opens.shift(-horizon-1)/opens.shift(-1)-1
    complete = opens.notna().iloc[::-1].rolling(horizon+1, min_periods=horizon+1).sum().iloc[::-1].shift(-1)
    return label.where(complete == horizon+1)

def daily_score(p: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """The fixed daily formula; only the five temporal weights are searched."""
    return 100*(.7*p+.3*(1+np.tanh(z/2))/2)

def score_windows(scores: pd.DataFrame) -> np.ndarray:
    """Day 1 is oldest, day 5 is the signal date. Missing days never get backfilled."""
    out = np.full((*scores.shape, 5), np.nan)
    if len(scores) >= 5:
        out[4:] = np.lib.stride_tricks.sliding_window_view(scores.to_numpy(float), 5, axis=0)
    return out

def candidate_weights(random_count: int = 1000, seed: int = 20261113) -> np.ndarray:
    """Five participating days, each at least 5%; no single-day fixed candidates."""
    rng = np.random.default_rng(seed)
    chunks = np.array_split(np.arange(random_count), 3)
    random = [.05+.75*rng.dirichlet(np.repeat(alpha, 5), len(chunk)) for alpha,chunk in zip((.5, 1., 2.), chunks)]
    return np.vstack([np.repeat(.2, 5), np.arange(1, 6)/15, *random])

def training_shortlist(trials: pd.DataFrame, candidates: np.ndarray, count: int = 15) -> list[dict]:
    """Choose the entire shortlist using training objectives only, never test ranks."""
    ranked = rank_trials(trials).head(count)
    if len(ranked) < count: raise ValueError(f"有效训练方案不足{count}组。")
    return [{"name": f"方案{rank:02d}", "rank": rank, "selected_id": int(row["编号"]),
             "weights_oldest_to_latest": candidates[int(row["编号"])].tolist(),
             "training_objective_value": float(row["训练目标"])}
            for rank,(_,row) in enumerate(ranked.iterrows(), 1)]

def top_indices(values: np.ndarray, top_n: int = 20) -> np.ndarray:
    """Partial selection with exact, code-order tie handling at the boundary."""
    safe = np.where(np.isfinite(values), values, -np.inf)
    count = min(top_n, safe.shape[1])
    ids = np.argpartition(-safe, count-1, axis=1)[:, :count]
    selected = np.take_along_axis(safe, ids, axis=1)
    boundary = selected.min(axis=1)
    tied = np.sum(safe == boundary[:, None], axis=1)
    tied_inside = np.sum(selected == boundary[:, None], axis=1)
    for row in np.flatnonzero(tied > tied_inside):
        above = np.flatnonzero(safe[row] > boundary[row])
        same = np.flatnonzero(safe[row] == boundary[row])
        ids[row] = np.r_[above, same[:count-len(above)]]
    selected = np.take_along_axis(safe, ids, axis=1)
    order = np.lexsort((ids, -selected), axis=1)
    ids = np.take_along_axis(ids, order, axis=1)
    ids[~np.isfinite(np.take_along_axis(safe, ids, axis=1))] = -1
    return ids

def temporal_split(dates: pd.DatetimeIndex) -> dict:
    """Latest five calendar years, conservatively purging eleven signal sessions before testing."""
    end = dates[-1]
    test_start = end-pd.DateOffset(years=1)
    start = end-pd.DateOffset(years=5)
    train = np.asarray((dates >= start) & (dates < test_start))
    # Require the furthest evaluated label to mature within training.
    for i in np.flatnonzero(train):
        if i+11 >= len(dates) or dates[i+11] >= test_start: train[i] = False
    test = np.asarray(dates >= test_start)
    if train.sum() < 200 or test.sum() < 100:
        raise ValueError("数据不足以完成前四年训练与最后一年隔离检验。")
    return {"start": start, "end": end, "test_start": test_start, "train": train, "test": test}

def _row_mean(values: np.ndarray) -> np.ndarray:
    count = np.isfinite(values).sum(axis=1)
    return np.divide(np.nansum(values, axis=1), count, out=np.full(len(values), np.nan), where=count > 0)

def training_inputs(windows: np.ndarray, opens: pd.DataFrame, split: dict) -> tuple:
    """Restrict outcome prices to the training calendar before forming any label."""
    test_position = int(np.flatnonzero(split["test"])[0])
    positions = np.flatnonzero(split["train"])
    y = forward_returns(opens.iloc[:test_position], 5).to_numpy()[positions]
    x = windows[positions]
    market = _row_mean(np.where(np.isfinite(x).all(axis=2), y, np.nan))
    return x, y, market

def rank_trials(trials: pd.DataFrame) -> pd.DataFrame:
    keys = ["训练目标"]
    if "训练三日触及超额" in trials:
        keys += ["训练三日触及超额", "训练辅助收益超额"]
    return trials[np.isfinite(trials["训练目标"])].sort_values(
        keys+["编号"], ascending=[False]*len(keys)+[True], kind="stable", na_position="last")

def training_opportunities(windows: np.ndarray, opens: pd.DataFrame, high: pd.DataFrame, split: dict) -> dict:
    """Construct peak/event labels using physically truncated training prices only."""
    stop = int(np.flatnonzero(split["test"])[0])
    positions = np.flatnonzero(split["train"])
    o, h = opens.iloc[:stop], high.iloc[:stop]
    p3 = peak_returns(h, o, 3)
    return {"peak5": peak_returns(h, o, 5).to_numpy()[positions],
            "hit3": np.where(p3.notna(), p3.round(12) > .05, np.nan)[positions]}

def search_training(x: np.ndarray, y: np.ndarray, market: np.ndarray, candidates: np.ndarray,
                    top_n: int = 20, progress: Callable[[str], None] = lambda _: None,
                    check: Callable[[], None] = lambda: None, *, opportunities: dict | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Rank training peak excess first, event excess second, auxiliary labelled-return excess last."""
    rows = []
    for i, weights in enumerate(candidates):
        check()
        score = np.einsum("tnd,d->tn", x, weights, optimize=False)
        ids = top_indices(score, top_n)
        outcomes = np.take_along_axis(y, np.maximum(ids, 0), axis=1)
        outcomes[ids < 0] = np.nan
        daily = _row_mean(outcomes)
        net = (1+daily)*(1-COST)**2-1
        reference = (1+market)*(1-COST)**2-1
        effect = net-reference
        objective = float(np.nanmean(effect)) if np.isfinite(effect).any() else -np.inf
        row = {"编号": i, "训练目标": objective, "训练平均净收益": float(np.nanmean(net)),
               "有效训练日期": int(np.isfinite(effect).sum())}
        if opportunities is not None:
            row["训练辅助收益超额"] = objective
            for key, column in (("peak5", "训练五日峰值超额"), ("hit3", "训练三日触及超额")):
                values = opportunities[key]
                selected = np.take_along_axis(values, np.maximum(ids, 0), axis=1).copy()
                selected[ids < 0] = np.nan
                reference = _row_mean(np.where(np.isfinite(x).all(axis=2), values, np.nan))
                delta = _row_mean(selected)-reference
                row[column] = float(np.nanmean(delta)) if np.isfinite(delta).any() else np.nan
            row["训练目标"] = row["训练五日峰值超额"]
        row.update({f"第{j+1}日": float(w) for j,w in enumerate(weights)})
        rows.append(row)
        if i % 20 == 0: progress(f"仅在前四年比较五日权重 {i+1}/{len(candidates)}")
    trials = pd.DataFrame(rows)
    valid = trials[np.isfinite(trials["训练目标"])]
    if valid.empty: raise ValueError("所有训练方案都缺少有效收益，未选择权重。")
    winner_id = int(rank_trials(valid).iloc[0]["编号"])
    return candidates[winner_id].copy(), trials

def peak_returns(high: pd.DataFrame, opens: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Intraday maximum over entry day through entry+H-1; require a complete window."""
    maximum = high.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1].shift(-1)
    return maximum/opens.shift(-1)-1

def outcome_labels(close: pd.DataFrame, opens: pd.DataFrame, high: pd.DataFrame, *, peak_fraction: float | None = None) -> dict:
    p3 = peak_returns(high, opens, 3)
    p5 = peak_returns(high, opens, 5)
    return {
        "return5": forward_returns(opens, 5) if peak_fraction is None else peak_fraction*p5,
        "peak5": p5,
        "peak10": peak_returns(high, opens, 10),
        "hit3": pd.DataFrame(np.where(p3.notna(), p3.round(12) > .05, np.nan), index=p3.index, columns=p3.columns),
        "next_day": close.shift(-1)/close-1,
        "entry_day": close.shift(-1)/opens.shift(-1)-1,
    }
