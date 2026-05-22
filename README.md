# StatArb Pipeline (Spark + OPTICS + RL)

This project implements a minimal, library-first research pipeline for statistical arbitrage:
cluster assets, form pairs, compute spreads and z-scores, backtest rule-based trading, and
optionally train an RL agent to scale position sizes.

## What it does

1. **Data ingest** (Stooq): daily prices for tickers + market proxy series (may require `STOOQ_API_KEY`).
2. **Feature build** (Spark): monthly momentum features (mom1..mom48) + fundamentals.
3. **Clustering** (OPTICS):
    - **Feature OPTICS**: cluster by fundamentals + momentum (Han et al. style).
    - **Distance OPTICS**: cluster by distance matrix (SSD/PCA/partial correlation).
4. **Pair selection**: inside each cluster, sort by mom1, pair top vs bottom, keep pairs
   whose mom1 spread exceeds a cross-sectional threshold.
5. **Spread + z-score**: OLS spread, rolling z-score, trading zones.
6. **Backtest**: rule-based open/close logic with transaction costs.
7. **RL sizing** (optional): continuous action in [-1, 1], shaped reward, position scaling.

## Modes (clustering)

Set in `config/base.yaml`:

### 1) distance_optics

Uses OPTICS on a distance matrix computed over a rolling formation window.
`distance_method` options:

- `pc_distance` (partial correlation vs market index, Kenett style)
- `pca_distance` (PCA of returns, Sarmento & Horta style)
- `ssd_distance` (normalized price SSD, Gatev style)

Key controls:

- `formation_months`: rolling formation window length (e.g., 36 months)
- `pca_components_distance`: PCA components used for PCA distance
- `optics_min_samples`, `optics_xi`, `optics_min_cluster_size`

### 2) feature_optics

Clusters directly on monthly features (momentum + fundamentals).
Uses PCA (variance threshold) and standardization before OPTICS.

## Configuration highlights

`config/base.yaml`:

- **data**: tickers, date range, `prices_path`, `fundamentals_path`, `market_path`, `market_ticker`
- **features**: `momentum_windows` (1..48), `pca_components` (e.g., 0.99 variance)
- **pairs**: `entry_z`, `exit_z`, `hedge_lookback`, `zscore_lookback`
- **rl**: `algo` (A2C/PPO), `transaction_cost_bps`, `action_reward_weight`

## How to run

```bash
# Build and start
docker compose up -d --build

# Execute pipeline inside container (Spark cluster is already running)
docker compose exec app python -m src.pipeline.run

# Check logs
docker compose logs app
```

### Quick tests

```bash
# Test all modules without full pipeline
python test_local.py

# Test specific module
python -c "from src.clustering.distance_optics import DistanceOPTICS; print('✓ OPTICS imported')"
```

## Outputs

Generated in `data/`:

- `pairs.csv`: selected pairs per month (columns: ticker1, ticker2, cluster_id, mom1_spread, zscore, signal)
- `rule_backtest.csv`: rule-based NAV, equity, signals, transaction costs
- `rl_model.zip`: trained RL agent (if enabled in config)
- `clustering_debug.csv`: cluster purity, size per month

## How to verify the result

After running the pipeline, check these files in `data/`:

- `pairs.csv` and `rule_backtest.csv` mean the pipeline found pairs and completed the rule-based backtest.
- `rl_model.zip` appears only when `rl.enabled: true` and at least one pair was found.
- `pipeline.log` captures the full Spark run, and `pipeline.exit` contains the final exit code from the last captured run.

If the command exits `0` but `pairs.csv` is missing, the pipeline stopped early because pair selection returned no eligible pairs for that run.

## Notes

- Data is fetched from Stooq (daily data). If Stooq returns “Get your apikey”, set `STOOQ_API_KEY` via env.
- If your network returns HTML instead of CSV, set `STOOQ_BASE_URLS=https://stooq.pl`.
- The distance-based mode needs the market index series (`market_path`).
- RL agent is a sizing layer on top of pairs; it does not generate signals directly.
- PySpark connects to the Apache Spark master via `SPARK_MASTER_URL` (spark://spark-master:7077).
- Transaction costs are hardcoded at `cost_bps` in config; adjust for your broker's fees.
