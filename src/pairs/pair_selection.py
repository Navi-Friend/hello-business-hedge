from __future__ import annotations

from typing import List

import pandas as pd


def select_pairs(
    clustered: pd.DataFrame,
    features: pd.DataFrame,
    dispersion_threshold: float,
    max_pairs_per_cluster: int,
) -> pd.DataFrame:
    clustered = clustered.copy()
    features = features.copy()
    clustered["month"] = pd.to_datetime(clustered["month"])
    features["month"] = pd.to_datetime(features["month"])
    merged = clustered.merge(features[["month", "ticker", "mom1"]], on=["month", "ticker"], how="left")
    pairs: List[dict] = []
    diffs: List[float] = []

    for month, month_df in merged.groupby("month"):
        for cluster, cluster_df in month_df.groupby("cluster"):
            if cluster == -1:
                continue
            cluster_df = cluster_df.dropna(subset=["mom1"])
            cluster_df = cluster_df.sort_values("mom1")
            pair_count = min(max_pairs_per_cluster, len(cluster_df) // 2)
            for idx in range(pair_count):
                short_row = cluster_df.iloc[idx]
                long_row = cluster_df.iloc[-(idx + 1)]
                diff = float(long_row["mom1"] - short_row["mom1"])
                diffs.append(diff)
                pairs.append(
                    {
                        "month": month,
                        "long_ticker": long_row["ticker"],
                        "short_ticker": short_row["ticker"],
                        "cluster": cluster,
                        "mom1_diff": diff,
                    }
                )

    if not pairs:
        return pd.DataFrame(columns=["month", "long_ticker", "short_ticker", "cluster", "mom1_diff"])

    diff_std = pd.Series(diffs).std()
    if pd.isna(diff_std) or diff_std == 0:
        return pd.DataFrame(columns=["month", "long_ticker", "short_ticker", "cluster", "mom1_diff"])

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df[pairs_df["mom1_diff"] > max(diff_std, dispersion_threshold)]
    if pairs_df.empty:
        return pd.DataFrame(columns=["month", "long_ticker", "short_ticker", "cluster", "mom1_diff"])
    return pairs_df.reset_index(drop=True)
