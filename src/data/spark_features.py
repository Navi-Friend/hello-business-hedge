from __future__ import annotations

from typing import Iterable

from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F


def build_monthly_features(
    prices_df: DataFrame, fundamentals_df: DataFrame, momentum_windows: Iterable[int]
) -> DataFrame:
    w = Window.partitionBy("ticker").orderBy("date")
    df = prices_df.withColumn("ret", F.col("close") / F.lag("close").over(w) - F.lit(1.0))
    w_vol = w.rowsBetween(-20, 0)
    df = df.withColumn("vol_21", F.stddev_samp("ret").over(w_vol))
    df = df.withColumn("month", F.date_trunc("month", "date"))

    monthly = df.groupBy("ticker", "month").agg(
        F.last("close", ignorenulls=True).alias("close"),
        F.last("vol_21", ignorenulls=True).alias("vol_21"),
    )

    w_month = Window.partitionBy("ticker").orderBy("month")
    for window in momentum_windows:
        monthly = monthly.withColumn(
            f"mom{window}",
            F.col("close") / F.lag("close", window).over(w_month) - F.lit(1.0),
        )

    features = monthly.join(F.broadcast(fundamentals_df), on="ticker", how="left")
    return features
