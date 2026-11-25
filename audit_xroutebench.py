
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data/external/xroutebench")

train = pd.read_parquet(DATA / "train.parquet")
test = pd.read_parquet(DATA / "test.parquet")
pricing = pd.read_parquet(DATA / "llm_candidates.parquet")

print("\n========== 1. UNIQUE PROMPTS ==========")

for name, df in [("TRAIN", train), ("TEST", test)]:
    print(f"\n{name}")
    print("Rows:", len(df))
    print("Unique queries:", df["query"].nunique())
    print("Unique tasks:", df["task_name"].nunique())
    print("Unique models:", df["model_name"].nunique())

print("\n========== 2. TASK DISTRIBUTION ==========")

print("\nTRAIN:")
print(train.groupby("task_name").agg(
    rows=("query", "size"),
    unique_prompts=("query", "nunique"),
    models=("model_name", "nunique")
).sort_values("unique_prompts", ascending=False))

print("\nTEST:")
print(test.groupby("task_name").agg(
    rows=("query", "size"),
    unique_prompts=("query", "nunique"),
    models=("model_name", "nunique")
).sort_values("unique_prompts", ascending=False))

print("\n========== 3. MODEL COVERAGE ==========")

print("\nTRAIN:")
print(train["model_name"].value_counts())

print("\nTEST:")
print(test["model_name"].value_counts())

print("\n========== 4. MODEL COVERAGE PER PROMPT ==========")

for name, df in [("TRAIN", train), ("TEST", test)]:
    coverage = df.groupby("query")["model_name"].nunique()

    print(f"\n{name}")
    print(coverage.describe())
    print("Prompts with all 18 models:",
          (coverage == 18).sum())
    print("Prompts with fewer than 18 models:",
          (coverage < 18).sum())

print("\n========== 5. PERFORMANCE DISTRIBUTION ==========")

print(train.groupby("task_name")["performance"].agg(
    ["count", "mean", "std", "min", "max"]
))

print("\nPerformance values:")
print(sorted(train["performance"].dropna().unique())[:50])

print("\n========== 6. METRIC TYPES ==========")

print(train.groupby(["task_name", "metric"]).size())

print("\n========== 7. DUPLICATE RECORDS ==========")

for name, df in [("TRAIN", train), ("TEST", test)]:
    duplicates = df.duplicated(
        subset=["query", "model_name"],
        keep=False
    )

    print(name, "duplicate query-model rows:", duplicates.sum())

print("\n========== 8. TRAIN / TEST LEAKAGE ==========")

train_queries = set(train["query"])
test_queries = set(test["query"])

overlap = train_queries.intersection(test_queries)

print("Train unique prompts:", len(train_queries))
print("Test unique prompts:", len(test_queries))
print("Overlapping prompts:", len(overlap))

print("\n========== 9. TOKEN AND LATENCY ==========")

print(train[[
    "input_tokens",
    "output_tokens",
    "token_num",
    "response_time"
]].describe())

print("\nNegative token counts:",
      (train[["input_tokens", "output_tokens"]] < 0).sum())

print("Non-positive latency:",
      (train["response_time"] <= 0).sum())

print("\n========== 10. PRICING REGISTRY ==========")

print(pricing[[
    "model_name",
    "input_price_per_1m",
    "output_price_per_1m",
    "service"
]].to_string(index=False))

train_models = set(train["model_name"])
pricing_models = set(pricing["model_name"])

print("\nModels without pricing:",
      sorted(train_models - pricing_models))

print("Pricing models without observations:",
      sorted(pricing_models - train_models))

print("\n========== AUDIT COMPLETE ==========")
