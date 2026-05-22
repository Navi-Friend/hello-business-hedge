from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class OpticsConfig:
    min_samples: int
    xi: float
    min_cluster_size: float
    distance_metric: str
    pca_components: float
    standardize: bool


def _prepare_matrix(df: pd.DataFrame, feature_cols: List[str], config: OpticsConfig) -> np.ndarray:
    matrix = df[feature_cols].to_numpy(dtype=float)
    if config.standardize:
        matrix = StandardScaler().fit_transform(matrix)
    if config.pca_components:
        n_components = config.pca_components
        if isinstance(n_components, float) and n_components < 1:
            matrix = PCA(n_components=n_components, random_state=42).fit_transform(matrix)
        elif matrix.shape[1] > int(n_components):
            matrix = PCA(n_components=int(n_components), random_state=42).fit_transform(matrix)
    return matrix


def run_optics_by_month(features: pd.DataFrame, config: OpticsConfig) -> pd.DataFrame:
    feature_cols = [col for col in features.columns if col not in ("ticker", "month")]
    outputs = []

    for month, group in features.groupby("month"):
        group = group.dropna(subset=feature_cols)
        if len(group) < max(config.min_samples, 2):
            continue
        matrix = _prepare_matrix(group, feature_cols, config)
        optics = OPTICS(
            min_samples=config.min_samples,
            xi=config.xi,
            min_cluster_size=config.min_cluster_size,
            metric=config.distance_metric,
        )
        labels = optics.fit_predict(matrix)
        output = group[["month", "ticker"]].copy()
        output["cluster"] = labels
        outputs.append(output)

    if not outputs:
        return pd.DataFrame(columns=["month", "ticker", "cluster"])

    return pd.concat(outputs, ignore_index=True)
