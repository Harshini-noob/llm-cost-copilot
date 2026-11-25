
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import json

OUTPUT_DIR = Path("data/external/xroutebench")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("M4.1 - XROUTEBENCH DATASET DOWNLOAD")
print("=" * 60)

# 1. Download the generic benchmark
print("\n[1/2] Downloading generic benchmark...")

dataset = load_dataset(
    "ulab-ai/xRouteBench",
    "llmrouter_generic"
)

print("\nAvailable splits:")
print(dataset)

# Save every split independently
for split_name, split_data in dataset.items():

    output_path = OUTPUT_DIR / f"{split_name}.parquet"

    split_data.to_parquet(output_path)

    print(
        f"Saved {split_name}: "
        f"{len(split_data)} rows -> {output_path}"
    )

# 2. Download candidate model pricing
print("\n[2/2] Downloading candidate model registry...")

pricing = load_dataset(
    "ulab-ai/xRouteBench",
    "llm_candidates"
)

print("\nCandidate dataset:")
print(pricing)

pricing_train = pricing["train"]

pricing_path = OUTPUT_DIR / "llm_candidates.parquet"

pricing_train.to_parquet(pricing_path)

print(
    f"Saved candidate registry: "
    f"{len(pricing_train)} rows -> {pricing_path}"
)

print("\nDOWNLOAD COMPLETE")
