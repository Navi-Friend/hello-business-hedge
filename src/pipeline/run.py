from __future__ import annotations

from pathlib import Path
import logging
import os

import numpy as np
import pandas as pd

from src.backtest.portfolio_backtest import (
    BacktestConfig,
    build_rolling_signal_dataset,
    simulate_portfolio,
)
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


def _compute_sharpe(nav: pd.Series) -> float:
    nav_values = nav.to_numpy(dtype=float)
    if len(nav_values) < 2:
        return 0.0
    returns = np.diff(nav_values) / (nav_values[:-1] + 1e-12)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))


def _clear_rl_outputs(output_dir: Path) -> None:
    pd.DataFrame(columns=["date", "nav", "reward", "pnl"]).to_csv(output_dir / "rl_test_nav.csv", index=False)
    for model_path in (output_dir / "rl_model.zip", output_dir / "rl_model"):
        model_path.unlink(missing_ok=True)


def _split_signal_dataset(signal_df: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(pd.to_datetime(signal_df["month"]).dt.to_period("M").unique())
    if len(months) < 2:
        return signal_df, pd.DataFrame(columns=signal_df.columns)
    split_idx = min(max(1, int(len(months) * train_fraction)), len(months) - 1)
    train_months = set(months[:split_idx])
    month_periods = pd.to_datetime(signal_df["month"]).dt.to_period("M")
    train_df = signal_df[month_periods.isin(train_months)].reset_index(drop=True)
    test_df = signal_df[~month_periods.isin(train_months)].reset_index(drop=True)
    return train_df, test_df


def _evaluate_agent(model, signal_df: pd.DataFrame, env_config: EnvConfig) -> pd.DataFrame:
    signal_df = signal_df.copy()
    signal_df["date"] = pd.to_datetime(signal_df["date"])
    nav = 1.0
    positions: dict[str, float] = {}
    records = []

    for _, month_df in signal_df.groupby("month"):
        for episode_id in month_df["episode_id"].unique():
            positions[episode_id] = 0.0

        for date, date_df in month_df.groupby("date"):
            portfolio_pnl = 0.0
            total_turnover = 0.0
            active_pairs = 0

            for row in date_df.itertuples(index=False):
                position = positions.get(row.episode_id, 0.0)
                zscore = float(row.zscore) if np.isfinite(float(row.zscore)) else 0.0
                zone = float(row.zone) if np.isfinite(float(row.zone)) else 0.0
                signal_direction = str(getattr(row, "signal_direction", env_config.signal_direction))
                if signal_direction == "adaptive":
                    signal_direction = "mean_reversion"
                direction_value = 1.0 if signal_direction == "mean_reversion" else -1.0
                obs = np.array([position, zscore, zone, direction_value], dtype=np.float32)
                action, _ = model.predict(obs, deterministic=True)
                target = float(np.clip(action[0], -1.0, 1.0))
                if bool(row.episode_end):
                    target = 0.0

                spread_ret = float(row.spread_return)
                if not np.isfinite(spread_ret):
                    spread_ret = 0.0
                spread_ret = np.clip(spread_ret, -0.05, 0.05)

                turnover = abs(target - position)
                cost = turnover * (env_config.transaction_cost_bps / 10000.0)
                pnl = position * spread_ret - cost
                pnl = float(np.clip(pnl, -0.05, 0.05))
                if not np.isfinite(pnl):
                    pnl = 0.0

                portfolio_pnl += pnl
                total_turnover += turnover
                active_pairs += 1
                positions[row.episode_id] = target

            avg_pnl = portfolio_pnl / max(active_pairs, 1)
            nav *= 1.0 + avg_pnl
            records.append(
                {
                    "date": date,
                    "nav": nav,
                    "reward": avg_pnl,
                    "pnl": avg_pnl,
                    "turnover": total_turnover / max(active_pairs, 1),
                    "num_pairs": active_pairs,
                }
            )

    return pd.DataFrame(records)


def _rule_sharpe_for_rl_period(rule_results: pd.DataFrame, rl_nav: pd.DataFrame) -> float:
    if rl_nav.empty:
        return 0.0
    rule_copy = rule_results.copy()
    rule_copy["date"] = pd.to_datetime(rule_copy["date"])
    rl_dates = pd.to_datetime(rl_nav["date"])
    period_rule = rule_copy[(rule_copy["date"] >= rl_dates.min()) & (rule_copy["date"] <= rl_dates.max())]
    if period_rule.empty:
        return 0.0
    return _compute_sharpe(period_rule["nav"])


def _ensure_inputs(config: Config) -> None:
    prices_path = Path(config.data.prices_path)
    prices_for_proxy = pd.DataFrame()
    prices_updated = False
    if prices_path.exists():
        prices_for_proxy = pd.read_parquet(prices_path)

    required_tickers = set(config.data.tickers)
    available_tickers = set(prices_for_proxy["ticker"].unique()) if not prices_for_proxy.empty else set()
    missing_tickers = sorted(required_tickers - available_tickers)
    coverage = len(available_tickers & required_tickers) / max(len(required_tickers), 1)
    needs_refresh = prices_for_proxy.empty or coverage < 0.95

    if needs_refresh:
        if prices_for_proxy.empty:
            logger.info("Prices parquet missing or empty; downloading from data source")
        else:
            logger.warning(
                "Prices parquet covers only %d/%d configured tickers; refreshing. Missing sample=%s",
                len(available_tickers & required_tickers),
                len(required_tickers),
                missing_tickers[:10],
            )
        prices = fetch_prices_stooq(config.data.tickers, config.data.start, config.data.end)
        if prices.empty:
            if prices_for_proxy.empty:
                raise RuntimeError(
                    "prices.parquet is empty; data download failed. "
                    "Check network access, or provide data/prices.parquet manually."
                )
            logger.warning("Data refresh failed; keeping existing incomplete prices parquet")
        else:
            fetched_tickers = set(prices["ticker"].unique())
            if len(fetched_tickers & required_tickers) <= len(available_tickers & required_tickers):
                logger.warning(
                    "Data refresh did not improve ticker coverage (%d fetched vs %d cached); keeping cache",
                    len(fetched_tickers & required_tickers),
                    len(available_tickers & required_tickers),
                )
            else:
                ensure_parquet(config.data.prices_path, prices)
                prices_for_proxy = prices
                prices_updated = True
                logger.info(
                    "Prices refreshed: %d/%d configured tickers available",
                    len(fetched_tickers & required_tickers),
                    len(required_tickers),
                )

    fundamentals_path = Path(config.data.fundamentals_path)
    if not fundamentals_path.exists():
        logger.info("Fundamentals parquet missing; creating placeholder fundamentals")
        fundamentals = fetch_fundamentals(config.data.tickers)
        ensure_parquet(config.data.fundamentals_path, fundamentals)

    market_path = Path(config.data.market_path)
    if prices_updated or not market_path.exists():
        logger.info("Building market proxy from current prices")
        market_prices = build_market_proxy(prices_for_proxy, config.data.market_ticker)
        ensure_parquet(config.data.market_path, market_prices)

    final_tickers = set(prices_for_proxy["ticker"].unique()) if not prices_for_proxy.empty else set()
    final_coverage = len(final_tickers & required_tickers) / max(len(required_tickers), 1)
    if final_coverage < 0.5:
        raise RuntimeError(
            f"Insufficient price universe: {len(final_tickers & required_tickers)}/"
            f"{len(required_tickers)} configured tickers available. "
            "Provide a fuller prices.parquet or set a working STOOQ_API_KEY."
        )


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
        selection_method=config.pairs.selection_method,
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
        max_pairs=config.pairs.max_portfolio_pairs,
        formation_months=config.clustering.formation_months,
        hedge_lookback=config.pairs.hedge_lookback,
        zscore_lookback=config.pairs.zscore_lookback,
        signal_direction=config.pairs.signal_direction,
        min_formation_score=config.pairs.min_formation_score,
    )
    
    logger.info(
        "Running rolling portfolio backtest: %d formation months, up to %d pairs/month",
        backtest_config.formation_months,
        backtest_config.max_pairs,
    )
    rule_results = simulate_portfolio(prices_pd, pairs_pd, backtest_config)
    rule_sharpe = _compute_sharpe(rule_results["nav"])
    logger.info("Rule-based rolling Sharpe: %.3f", rule_sharpe)

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_pd.to_csv(output_dir / "pairs.csv", index=False)
    rule_results.to_csv(output_dir / "rule_backtest.csv", index=False)
    _clear_rl_outputs(output_dir)
    logger.info("Wrote outputs: %s", output_dir.resolve())

    if config.rl.enabled:
        if rule_sharpe < 0.0:
            logger.warning(
                "Skipping RL training because rule-based Sharpe is negative (%.3f). "
                "Fix pair selection/baseline before optimizing RL sizing.",
                rule_sharpe,
            )
            return

        logger.info("RL enabled: training %s", config.rl.algo)
        env_config = EnvConfig(
            transaction_cost_bps=config.rl.transaction_cost_bps,
            turnover_penalty=config.rl.turnover_penalty,
            drawdown_penalty=config.rl.drawdown_penalty,
            action_reward_weight=config.rl.action_reward_weight,
            signal_direction=config.pairs.signal_direction,
        )
        signal_df = build_rolling_signal_dataset(prices_pd, pairs_pd, backtest_config)
        if signal_df.empty:
            logger.warning("Skipping RL training: no rolling signal dataset could be built")
            return

        train_signal, test_signal = _split_signal_dataset(signal_df)
        if train_signal.empty or test_signal.empty:
            logger.warning(
                "Skipping RL training: insufficient train/test signals (train=%d, test=%d)",
                len(train_signal),
                len(test_signal),
            )
            return

        logger.info(
            "RL dataset: train rows=%d, test rows=%d, train months=%d, test months=%d",
            len(train_signal),
            len(test_signal),
            pd.to_datetime(train_signal["month"]).dt.to_period("M").nunique(),
            pd.to_datetime(test_signal["month"]).dt.to_period("M").nunique(),
        )
        env = PairTradingEnv(train_signal, env_config)
        model = train_agent(env, config.rl.algo, config.rl.total_timesteps)

        rl_test_nav = _evaluate_agent(model, test_signal, env_config)
        rl_test_nav.to_csv(output_dir / "rl_test_nav.csv", index=False)
        rl_sharpe = _compute_sharpe(rl_test_nav["nav"])
        logger.info("RL out-of-sample Sharpe: %.3f", rl_sharpe)
        rule_oos_sharpe = _rule_sharpe_for_rl_period(rule_results, rl_test_nav)
        if rl_sharpe > rule_oos_sharpe:
            model.save(str(output_dir / "rl_model"))
            logger.info("Saved RL model to %s", (output_dir / "rl_model").resolve())
        else:
            for model_path in (output_dir / "rl_model.zip", output_dir / "rl_model"):
                model_path.unlink(missing_ok=True)
            logger.warning(
                "Discarded RL model because RL Sharpe %.3f did not beat rule OOS Sharpe %.3f",
                rl_sharpe,
                rule_oos_sharpe,
            )


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config("config/base.yaml")
    if os.getenv("VERIFY_ONLY", "").lower() in {"1", "true", "yes"}:
        logger.info("VERIFY_ONLY is set; running data/pairs/rule-backtest checks without RL training")
        config.rl.enabled = False
    run_pipeline(config)


if __name__ == "__main__":
    main()
