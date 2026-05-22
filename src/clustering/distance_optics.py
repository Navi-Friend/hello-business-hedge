from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


@dataclass
class DistanceOpticsConfig:
    method: str
    formation_months: int
    pca_components: int
    min_samples: int
    xi: float
    min_cluster_size: int


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame - frame.mean()
    scaled = centered / frame.std(ddof=0)
    return scaled.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")


def compute_ssd_distance(prices: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_frame(prices)
    matrix = normalized.to_numpy().T
    distances = pairwise_distances(matrix, metric="euclidean") ** 2
    return pd.DataFrame(distances, index=normalized.columns, columns=normalized.columns)


def compute_pca_distance(returns: pd.DataFrame, n_components: int) -> pd.DataFrame:
    normalized = _normalize_frame(returns)
    pca = PCA(n_components=min(n_components, normalized.shape[1]), random_state=42)
    scores = pca.fit_transform(normalized.to_numpy())
    reconstructed = scores @ pca.components_
    reconstructed_df = pd.DataFrame(reconstructed, index=normalized.index, columns=normalized.columns)
    return compute_ssd_distance(reconstructed_df)


def compute_pc_distance(returns: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
    aligned = returns.join(market_returns.rename("market"), how="inner")
    aligned = aligned.dropna(axis=0, how="any")
    if aligned.empty:
        raise ValueError("Not enough data to compute partial correlation distance.")

    market = aligned["market"]
    returns = aligned.drop(columns=["market"])
    corr = returns.corr()
    corr_m = returns.corrwith(market)

    rho_xy = corr.to_numpy()
    rho_xm = corr_m.to_numpy()
    denom = np.sqrt(1.0 - rho_xm**2)
    denom_matrix = np.outer(denom, denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_par = (rho_xy - np.outer(rho_xm, rho_xm)) / denom_matrix
    rho_par = np.clip(rho_par, -1.0, 1.0)
    distance = 1.0 - np.abs(rho_par)
    distance = np.nan_to_num(distance, nan=1.0, posinf=1.0, neginf=1.0)
    np.fill_diagonal(distance, 0.0)
    return pd.DataFrame(distance, index=returns.columns, columns=returns.columns)


def run_optics_distance_by_month(
    prices: pd.DataFrame,
    market_prices: pd.Series,
    tickers: Iterable[str],
    config: DistanceOpticsConfig,
) -> pd.DataFrame:
    prices_wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    tickers = [ticker for ticker in tickers if ticker in prices_wide.columns]
    prices_wide = prices_wide[tickers].dropna(axis=1, how="any")
    returns = prices_wide.pct_change().dropna(axis=0, how="any")
    market_series = market_prices.sort_index().dropna()
    market_returns = market_series.pct_change().dropna()

    months = sorted(returns.index.to_period("M").unique())
    outputs: List[pd.DataFrame] = []

    for month in months:
        month_start = month.to_timestamp()
        formation_end = month_start - pd.Timedelta(days=1)
        formation_start = formation_end - pd.DateOffset(months=config.formation_months)
        window_returns = returns.loc[
            (returns.index >= formation_start) & (returns.index <= formation_end)
        ].dropna(axis=1, how="any")
        if len(window_returns) < 60:
            continue

        if config.method == "pc_distance":
            window_market = market_returns.reindex(window_returns.index).dropna()
            window_returns = window_returns.loc[window_market.index]
            if window_returns.empty:
                continue
            distance = compute_pc_distance(window_returns, window_market)
        elif config.method == "pca_distance":
            distance = compute_pca_distance(window_returns, config.pca_components)
        else:
            distance = compute_ssd_distance(prices_wide.loc[window_returns.index])

        if distance.empty or distance.shape[0] < config.min_samples:
            continue

        optics = OPTICS(
            min_samples=config.min_samples,
            xi=config.xi,
            min_cluster_size=config.min_cluster_size,
            metric="precomputed",
        )
        labels = optics.fit_predict(distance.to_numpy())
        outputs.append(
            pd.DataFrame(
                {"month": month_start, "ticker": distance.index, "cluster": labels}
            )
        )

    if not outputs:
        return pd.DataFrame(columns=["month", "ticker", "cluster"])

    return pd.concat(outputs, ignore_index=True)
