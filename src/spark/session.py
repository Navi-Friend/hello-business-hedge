import os

from pyspark.sql import SparkSession


def build_spark(app_name: str = "stat_arb") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
    )

    master = os.getenv("SPARK_MASTER_URL")
    if master:
        builder = builder.master(master)

    return builder.getOrCreate()
