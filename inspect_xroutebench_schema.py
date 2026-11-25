
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/external/xroutebench")

train = pd.read_parquet(DATA_DIR / "train.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")
pricing = pd.read_parquet(
    DATA_DIR / "llm_candidates.parquet"
)

def inspect_dataframe(name, df):

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst record:")
    print(df.iloc[0].to_dict())

inspect_dataframe("TRAIN DATA", train)
inspect_dataframe("TEST DATA", test)
inspect_dataframe("MODEL PRICING", pricing)

print("\n" + "=" * 65)
print("TRAIN / TEST PROMPT COUNTS")
print("=" * 65)

for name, df in [("Train", train), ("Test", test)]:

    possible_id_columns = [
        c for c in df.columns
        if "id" in c.lower() or "query" in c.lower()
    ]

    print(name, "rows:", len(df))
    print("Possible ID columns:", possible_id_columns)

print("\nSchema inspection complete.")
