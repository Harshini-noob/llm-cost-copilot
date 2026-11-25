
import pandas as pd
from pathlib import Path

DATA = Path("data/external/xroutebench")

train = pd.read_parquet(DATA / "train.parquet")
test = pd.read_parquet(DATA / "test.parquet")

for name, df in [("TRAIN", train), ("TEST", test)]:
    print(f"\n========== {name}: CROSS-TASK PROMPTS ==========")

    task_counts = (
        df.groupby("query")["task_name"]
        .nunique()
    )

    cross_task_queries = task_counts[task_counts > 1]

    print("Queries appearing in multiple tasks:",
          len(cross_task_queries))

    if len(cross_task_queries):
        details = (
            df[df["query"].isin(cross_task_queries.index)]
            .groupby("query")
            .agg(
                tasks=("task_name", lambda x: sorted(set(x))),
                models=("model_name", "nunique"),
                rows=("model_name", "size"),
            )
        )

        print(details.to_string())

        print("\nTask-pair distribution:")

        pairs = (
            df[df["query"].isin(cross_task_queries.index)]
            [["query", "task_name"]]
            .drop_duplicates()
        )

        print(
            pairs.groupby("task_name").size()
        )

    print("\n========== TASK PROMPT WEIGHTS ==========")

    print(
        df.groupby("task_name")
        .agg(
            unique_prompts=("query", "nunique"),
            rows=("query", "size"),
        )
        .to_string()
    )

print("\n========== COMPLETE ==========")
