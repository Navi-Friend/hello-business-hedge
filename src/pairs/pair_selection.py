from __future__ import annotations

from typing import List

import pandas as pd


def select_pairs(
    clustered: pd.DataFrame,
    features: pd.DataFrame,
    dispersion_threshold: float,
    max_pairs_per_cluster: int,
    selection_method: str = "cluster_all",
) -> pd.DataFrame:
    clustered = clustered.copy()
    features = features.copy()
    clustered["month"] = pd.to_datetime(clustered["month"])
    features["month"] = pd.to_datetime(features["month"])

    # Features are stamped at the month whose closing price was used. For a
    # trading month M, only features from the end of M-1 are known.
    selection_features = features[["month", "ticker", "mom1"]].copy()
    selection_features["month"] = selection_features["month"] + pd.DateOffset(months=1)
    merged = clustered.merge(selection_features, on=["month", "ticker"], how="left")
    pairs: List[dict] = []
    selection_method = selection_method.lower()

    for month, month_df in merged.groupby("month"):
        month_pairs: List[dict] = []
        month_diffs: List[float] = []
        for cluster, cluster_df in month_df.groupby("cluster"):
            if cluster == -1:
                continue
            cluster_df = cluster_df.dropna(subset=["mom1"])
            cluster_df = cluster_df.sort_values("mom1")
            if selection_method == "momentum_reversal":
                pair_count = min(max_pairs_per_cluster, len(cluster_df) // 2)
                for idx in range(pair_count):
                    long_row = cluster_df.iloc[idx]
                    short_row = cluster_df.iloc[-(idx + 1)]
                    diff = float(short_row["mom1"] - long_row["mom1"])
                    month_diffs.append(diff)
                    month_pairs.append(
                        {
                            "month": month,
                            "long_ticker": long_row["ticker"],
                            "short_ticker": short_row["ticker"],
                            "cluster": cluster,
                            "mom1_diff": diff,
                        }
                    )
            else:
                candidates: List[dict] = []
                rows = list(cluster_df.itertuples(index=False))
                for left_idx in range(len(rows)):
                    for right_idx in range(left_idx + 1, len(rows)):
                        left = rows[left_idx]
                        right = rows[right_idx]
                        diff = abs(float(left.mom1) - float(right.mom1))
                        candidates.append(
                            {
                                "month": month,
                                "long_ticker": left.ticker,
                                "short_ticker": right.ticker,
                                "cluster": cluster,
                                "mom1_diff": diff,
                            }
                        )
                candidates = sorted(candidates, key=lambda item: item["mom1_diff"], reverse=True)
                month_pairs.extend(candidates[:max_pairs_per_cluster])

        if not month_pairs:
            continue
        if selection_method == "momentum_reversal":
            diff_std = pd.Series(month_diffs).std()
            if pd.isna(diff_std) or diff_std == 0:
                continue
            threshold = max(diff_std, dispersion_threshold)
            pairs.extend(pair for pair in month_pairs if pair["mom1_diff"] > threshold)
        else:
            pairs.extend(month_pairs)

    if not pairs:
        return pd.DataFrame(columns=["month", "long_ticker", "short_ticker", "cluster", "mom1_diff"])
    return pd.DataFrame(pairs).reset_index(drop=True)
