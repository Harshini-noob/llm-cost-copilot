
import pandas as pd
from pathlib import Path

DATA = Path("data/external/xroutebench")

train = pd.read_parquet(DATA / "train.parquet")

print("\n========== 1. DUPLICATE QUERY-MODEL RECORDS ==========")

mask = train.duplicated(
    subset=["query", "model_name"],
    keep=False
)

duplicates = train[mask].sort_values(
    ["query", "model_name"]
)

print("Duplicate rows:", len(duplicates))
print("Affected query-model pairs:",
      duplicates.groupby(["query", "model_name"]).ngroups)

print("\nFirst duplicate groups:")

for (query, model), group in list(
    duplicates.groupby(["query", "model_name"])
)[:5]:
    print("\nMODEL:", model)
    print("QUERY:", query[:250].replace("\n", " "))
    print("Rows:", len(group))

    columns = [
        "task_name", "performance", "input_tokens",
        "output_tokens", "response_time", "response"
    ]

    for col in columns:
        print(f"\n{col}:")
        print(group[col].tolist()[:5])


print("\n========== 2. EXACT DUPLICATE RECORDS ==========")

print(
    "Exact duplicate rows:",
    train.duplicated(keep=False).sum()
)

print("\n========== 3. TASK-METRIC SCORE RANGES ==========")

print(
    train.groupby(["task_name", "metric"])["performance"]
    .agg(["count", "nunique", "mean", "std", "min", "max"])
    .to_string()
)

print("\n========== 4. PERFORMANCE VALUES PER TASK ==========")

for task, group in train.groupby("task_name"):
    values = sorted(group["performance"].dropna().unique())

    print(f"\n{task}:")
    print("Unique score count:", len(values))
    print("First 20 values:", values[:20])
    print("Last 10 values:", values[-10:])


print("\n========== 5. TOKEN CONSISTENCY ==========")

print(
    "Rows where token_num != input_tokens + output_tokens:",
    (
        train["token_num"]
        != train["input_tokens"] + train["output_tokens"]
    ).sum()
)

print("\n========== INVESTIGATION COMPLETE ==========")
