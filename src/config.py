from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class DataConfig:
    tickers: List[str]
    start: str
    end: str
    prices_path: str
    fundamentals_path: str
    market_path: str
    market_ticker: str


@dataclass
class FeatureConfig:
    momentum_windows: List[int]
    pca_components: float
    standardize: bool


@dataclass
class ClusteringConfig:
    mode: str
    distance_method: str
    formation_months: int
    pca_components_distance: int
    optics_min_samples: int
    optics_xi: float
    optics_min_cluster_size: float
    dispersion_threshold: float
    distance_metric: str


@dataclass
class PairsConfig:
    selection_method: str
    signal_direction: str
    max_pairs_per_cluster: int
    max_portfolio_pairs: int
    min_formation_score: float
    hedge_lookback: int
    zscore_lookback: int
    entry_z: float
    exit_z: float


@dataclass
class RLConfig:
    enabled: bool
    algo: str
    total_timesteps: int
    transaction_cost_bps: float
    turnover_penalty: float
    drawdown_penalty: float
    action_reward_weight: float


@dataclass
class RiskConfig:
    max_leverage: float
    vol_target: float
    max_drawdown: float


@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig
    clustering: ClusteringConfig
    pairs: PairsConfig
    rl: RLConfig
    risk: RiskConfig


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    data = raw["data"]
    features = raw["features"]
    clustering = raw["clustering"]
    pairs = raw["pairs"]
    rl = raw["rl"]
    risk = raw["risk"]

    return Config(
        data=DataConfig(
            tickers=list(data["tickers"]),
            start=str(data["start"]),
            end=str(data["end"]),
            prices_path=str(data["prices_path"]),
            fundamentals_path=str(data["fundamentals_path"]),
            market_path=str(data["market_path"]),
            market_ticker=str(data["market_ticker"]),
        ),
        features=FeatureConfig(
            momentum_windows=list(features["momentum_windows"]),
            pca_components=float(features["pca_components"]),
            standardize=bool(features["standardize"]),
        ),
        clustering=ClusteringConfig(
            mode=str(clustering["mode"]),
            distance_method=str(clustering["distance_method"]),
            formation_months=int(clustering["formation_months"]),
            pca_components_distance=int(clustering["pca_components_distance"]),
            optics_min_samples=int(clustering["optics_min_samples"]),
            optics_xi=float(clustering["optics_xi"]),
            optics_min_cluster_size=float(clustering["optics_min_cluster_size"]),
            dispersion_threshold=float(clustering["dispersion_threshold"]),
            distance_metric=str(clustering["distance_metric"]),
        ),
        pairs=PairsConfig(
            selection_method=str(pairs.get("selection_method", "cluster_all")),
            signal_direction=str(pairs.get("signal_direction", "mean_reversion")),
            max_pairs_per_cluster=int(pairs["max_pairs_per_cluster"]),
            max_portfolio_pairs=int(pairs.get("max_portfolio_pairs", 20)),
            min_formation_score=float(pairs.get("min_formation_score", -1e9)),
            hedge_lookback=int(pairs["hedge_lookback"]),
            zscore_lookback=int(pairs["zscore_lookback"]),
            entry_z=float(pairs["entry_z"]),
            exit_z=float(pairs["exit_z"]),
        ),
        rl=RLConfig(
            enabled=bool(rl["enabled"]),
            algo=str(rl["algo"]),
            total_timesteps=int(rl["total_timesteps"]),
            transaction_cost_bps=float(rl["transaction_cost_bps"]),
            turnover_penalty=float(rl["turnover_penalty"]),
            drawdown_penalty=float(rl["drawdown_penalty"]),
            action_reward_weight=float(rl["action_reward_weight"]),
        ),
        risk=RiskConfig(
            max_leverage=float(risk["max_leverage"]),
            vol_target=float(risk["vol_target"]),
            max_drawdown=float(risk["max_drawdown"]),
        ),
    )
