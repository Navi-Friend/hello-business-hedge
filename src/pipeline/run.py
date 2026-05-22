from __future__ import annotations

from pathlib import Path
import logging
import os

import pandas as pd

from src.backtest.backtester import BacktestConfig, build_pair_signal, simulate_rule_based
from src.backtest.portfolio_backtest import simulate_portfolio
from src.clustering.distance_optics import DistanceOpticsConfig, run_optics_distance_by_month
from src.clustering.optics_cluster import OpticsConfig, run_optics_by_month
from src.config import Config, load_config
from src.data.spark_features import build_monthly_features
from src.data.stooq_loader import (
    build_market_proxy,
    ensure_parquet,
    fetch_fundamentals,
    fetch_prices_stooq,
)
from src.pairs.pair_selection import select_pairs
from src.rl.env import EnvConfig, PairTradingEnv
from src.rl.train import train_agent
from src.spark.session import build_spark

logger = logging.getLogger(__name__)


def _ensure_inputs(config: Config) -> None:
    prices_path = Path(config.data.prices_path)
    prices_for_proxy = pd.DataFrame()
    if prices_path.exists():
        prices_for_proxy = pd.read_parquet(prices_path)

    if prices_for_proxy.empty:
        logger.info("Prices parquet missing or empty; downloading from data source")
        prices = fetch_prices_stooq(config.data.tickers, config.data.start, config.data.end)
        if prices.empty:
            raise RuntimeError(
                "prices.parquet is empty; data download failed. "
                "Check network access, or provide data/prices.parquet manually."
            )
        ensure_parquet(config.data.prices_path, prices)
        prices_for_proxy = prices

    fundamentals_path = Path(config.data.fundamentals_path)
    if not fundamentals_path.exists():
        logger.info("Fundamentals parquet missing; creating placeholder fundamentals")
        fundamentals = fetch_fundamentals(config.data.tickers)
        ensure_parquet(config.data.fundamentals_path, fundamentals)

    market_path = Path(config.data.market_path)
    if not market_path.exists():
        logger.info("Market parquet missing; building proxy from prices")
        market_prices = build_market_proxy(prices_for_proxy, config.data.market_ticker)
        ensure_parquet(config.data.market_path, market_prices)


def run_pipeline(config: Config) -> None:
    logger.info("Starting pipeline run")
    _ensure_inputs(config)
    spark = build_spark()

    logger.info(
        "Inputs: prices=%s fundamentals=%s market=%s",
        config.data.prices_path,
        config.data.fundamentals_path,
        config.data.market_path,
    )

    prices_df = spark.read.parquet(config.data.prices_path)
    fundamentals_df = spark.read.parquet(config.data.fundamentals_path)

    features_df = build_monthly_features(
        prices_df, fundamentals_df, momentum_windows=config.features.momentum_windows
    )
    features_pd = features_df.toPandas()
    logger.info("Features built: rows=%d", len(features_pd))

    if config.clustering.mode == "distance_optics":
        logger.info("Clustering mode: distance_optics (%s)", config.clustering.distance_method)
        prices_pd = prices_df.select("date", "ticker", "close").toPandas()
        market_pd = spark.read.parquet(config.data.market_path).select("date", "close").toPandas()
        market_series = market_pd.set_index("date")["close"]
        distance_config = DistanceOpticsConfig(
            method=config.clustering.distance_method,
            formation_months=config.clustering.formation_months,
            pca_components=config.clustering.pca_components_distance,
            min_samples=config.clustering.optics_min_samples,
            xi=config.clustering.optics_xi,
            min_cluster_size=int(config.clustering.optics_min_cluster_size),
        )
        clusters_pd = run_optics_distance_by_month(
            prices_pd, market_series, config.data.tickers, distance_config
        )
    else:
        logger.info("Clustering mode: feature_optics")
        optics_config = OpticsConfig(
            min_samples=config.clustering.optics_min_samples,
            xi=config.clustering.optics_xi,
            min_cluster_size=config.clustering.optics_min_cluster_size,
            distance_metric=config.clustering.distance_metric,
            pca_components=config.features.pca_components,
            standardize=config.features.standardize,
        )
        clusters_pd = run_optics_by_month(features_pd, optics_config)
    logger.info("Clusters produced: rows=%d", len(clusters_pd))

    pairs_pd = select_pairs(
        clustered=clusters_pd,
        features=features_pd,
        dispersion_threshold=config.clustering.dispersion_threshold,
        max_pairs_per_cluster=config.pairs.max_pairs_per_cluster,
    )
    if pairs_pd.empty:
        logger.warning(
            "No pairs selected; stopping early. clusters=%d features=%d",
            len(clusters_pd),
            len(features_pd),
        )
        return
    logger.info("Pairs selected: rows=%d", len(pairs_pd))

    prices_pd = prices_df.select("date", "ticker", "close").toPandas()
    
    # Portfolio backtest (multiple pairs)
    backtest_config = BacktestConfig(
        entry_z=config.pairs.entry_z,
        exit_z=config.pairs.exit_z,
        transaction_cost_bps=config.rl.transaction_cost_bps,
        max_pairs=20,  # Top 20 pairs in portfolio
    )
    
    logger.info("Running portfolio backtest with up to %d pairs...", backtest_config.max_pairs)
    rule_results = simulate_portfolio(prices_pd, pairs_pd, backtest_config)
    
    # For RL training, use first pair only (for now)
    first_pair = pairs_pd.iloc[0]
    signal_df = build_pair_signal(
        prices_pd,
        long_ticker=first_pair["long_ticker"],
        short_ticker=first_pair["short_ticker"],
        hedge_lookback=config.pairs.hedge_lookback,
        zscore_lookback=config.pairs.zscore_lookback,
        entry_z=config.pairs.entry_z,
        exit_z=config.pairs.exit_z,
    )

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_pd.to_csv(output_dir / "pairs.csv", index=False)
    rule_results.to_csv(output_dir / "rule_backtest.csv", index=False)
    logger.info("Wrote outputs: %s", output_dir.resolve())

    if config.rl.enabled:
        logger.info("RL enabled: training %s", config.rl.algo)
        env_config = EnvConfig(
            transaction_cost_bps=config.rl.transaction_cost_bps,
            turnover_penalty=config.rl.turnover_penalty,
            drawdown_penalty=config.rl.drawdown_penalty,
            action_reward_weight=config.rl.action_reward_weight,
        )
        env = PairTradingEnv(signal_df, env_config)
        model = train_agent(env, config.rl.algo, config.rl.total_timesteps)
        model.save(str(output_dir / "rl_model"))
        logger.info("Saved RL model to %s", (output_dir / "rl_model").resolve())


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config("config/base.yaml")
    run_pipeline(config)


if __name__ == "__main__":
    main()
