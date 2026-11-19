
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "pilot_prompts.jsonl"

SUPPORTED_TASKS = {
    "math",
    "code",
    "knowledge",
    "extraction",
    "summarization",
    "chit_chat",
}


def load_dataset(path=DEFAULT_DATASET):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            required = {"prompt_id", "task_type", "prompt", "evaluation_method"}
            missing = required - record.keys()

            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {missing}"
                )

            if record["task_type"] not in SUPPORTED_TASKS:
                raise ValueError(
                    f"Unsupported task type on line {line_number}"
                )

            if not record["prompt"].strip():
                raise ValueError(f"Empty prompt on line {line_number}")

            records.append(record)

    prompt_ids = [record["prompt_id"] for record in records]

    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Duplicate prompt_id values found")

    return records


if __name__ == "__main__":
    dataset = load_dataset()

    print(f"Loaded {len(dataset)} prompts")

    for task in sorted(SUPPORTED_TASKS):
        count = sum(row["task_type"] == task for row in dataset)
        print(f"{task}: {count}")
