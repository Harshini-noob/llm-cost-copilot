"""
scripts/demo_m5.py

50-prompt offline demonstration of M5 runtime routing using genuine held-out xRouteBench
outcomes and cached local MiniLM embeddings (ZERO live LLM API calls).

Critical integrity rules:
1. The validation-selected Always-Best Single Model is chosen strictly on a prompt-grouped
   validation split from train_processed.parquet — NEVER on test outcomes.
2. Live provider registry (model_registry.py: openai/gpt-oss-20b, openai/gpt-oss-120b,
   gemini-2.5-flash) lacks a genuine 50-prompt held-out calibration dataset; this script
   explicitly discloses that gap and replays xRouteBench's 18-model pool offline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when executed as `python scripts/demo_m5.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.m4_data_adapter import create_prompt_grouped_validation_split
from eval.routers.data_prep import ModelOutcomeRecord, PreparedDataset, PromptRecord
from eval.routers.encoders import SentenceTransformerEncoder
from eval.routers.learned_router import LearnedRouter
from routers.lr_runtime import LearnedLRRuntimeRouter


def run_demo(
    *,
    lam: str | float = "auto:balanced",
    n_prompts: int = 50,
    seed: int = 42,
) -> dict:
    print("=" * 88)
    print("M5 Runtime Routing — 50-Prompt Offline Replay Demonstration (Zero Live API Calls)")
    print("=" * 88)
    print(
        "[Disclosure] Live provider registry (openai/gpt-oss-20b, openai/gpt-oss-120b, "
        "gemini-2.5-flash) does not have a genuine 50-prompt calibration dataset.\n"
        "             Replaying genuine held-out xRouteBench observations (18 models, 13 tasks) "
        "with cached local MiniLM embeddings."
    )

    train_df = pd.read_parquet("data/processed/xroutebench/train_processed.parquet")
    test_df = pd.read_parquet("data/processed/xroutebench/test_processed.parquet")
    tr_fit_df, val_df = create_prompt_grouped_validation_split(
        train_df, val_ratio=0.15, seed=seed
    )

    # Select Always-Best Single Model strictly on validation data (NOT test outcomes)
    val_model_perf = val_df.groupby("model_name")["performance"].mean()
    val_best_model = str(val_model_perf.idxmax())
    val_best_score = float(val_model_perf[val_best_model])
    print(
        f"[Validation] Always-Best Single Model selected on validation split: "
        f"{val_best_model} (val quality={val_best_score:.4f})"
    )

    # Load cached MiniLM encoder locally (never downloads)
    encoder = SentenceTransformerEncoder(model_name_or_path="all-MiniLM-L6-v2", local_files_only=True)

    # Load precomputed xRouteBench embeddings if available, else encode train fit prompts
    emb_path = Path("data/processed/xroutebench/embeddings_minilm.npz")
    tr_queries = sorted(tr_fit_df["query"].unique())
    if emb_path.exists():
        with np.load(emb_path, allow_pickle=False) as z:
            q_list = [str(q) for q in z["train_queries"].tolist()]
            q_embs = z["train_embeddings"].astype(np.float32)
            q_to_vec = {q: q_embs[i] for i, q in enumerate(q_list)}
        emb_map = {q: q_to_vec[q] for q in tr_queries if q in q_to_vec}
        missing = [q for q in tr_queries if q not in emb_map]
        if missing:
            m_vecs = encoder.encode(missing)
            for q, v in zip(missing, m_vecs):
                emb_map[q] = v
    else:
        vecs = encoder.encode(tr_queries)
        emb_map = {q: vecs[i] for i, q in enumerate(tr_queries)}

    # Build PreparedDataset strictly from tr_fit_df
    prompts_dict: dict[str, PromptRecord] = {}
    train_records: list[ModelOutcomeRecord] = []
    for row in tr_fit_df.itertuples(index=False):
        q = str(row.query)
        vec = emb_map[q]
        if q not in prompts_dict:
            prompts_dict[q] = PromptRecord(
                prompt_id=q,
                task_category=str(row.task_name),
                prompt_text=q,
                prompt_embedding=vec,
                split="train",
            )
        train_records.append(
            ModelOutcomeRecord(
                prompt_id=q,
                task_category=str(row.task_name),
                prompt_text=q,
                model_id=str(row.model_name),
                provider="xroutebench",
                label=float(row.performance),
                cost=float(row.reconstructed_cost_usd),
                token_usage=None,
                execution_status="success",
                evaluation_status="scored",
                split="train",
                run_count=1,
                latency_ms=float(row.response_time) * 1000.0,
                prompt_embedding=vec,
                correct_runs=float(row.performance),
                evaluated_runs=1,
                empirical_quality=float(row.performance),
            )
        )

    model_costs = tr_fit_df.groupby("model_name")["reconstructed_cost_usd"].mean().to_dict()

    dataset = PreparedDataset(
        records=train_records,
        prompts=prompts_dict,
        split_info={"train_prompt_count": len(tr_queries)},
        sufficiency_report={"status": "xroutebench_demo"},
    )

    lr = LearnedRouter(encoder=encoder, C=0.5, max_iter=1000, random_state=seed)
    lr.fit(dataset)
    runtime_router = LearnedLRRuntimeRouter(
        lr,
        model_costs=model_costs,
        router_version="m5-demo-lr-v1",
        encoder_name="all-MiniLM-L6-v2",
    )

    # Sample 50 mixed held-out test prompts stratified across tasks
    unique_test = (
        test_df[["task_name", "query"]]
        .drop_duplicates()
        .sample(n=n_prompts, random_state=seed)
        .reset_index(drop=True)
    )
    test_lookup = test_df.set_index(["query", "model_name"])

    print(
        f"\n{'#':>2} | {'Task':<16} | {'Chosen Model':<28} | {'Pred Q':>6} | "
        f"{'Obs Q':>6} | {'Obs Cost':>10} | {'FB':>3} | {'Version'}"
    )
    print("-" * 105)

    chosen_counts: Counter[str] = Counter()
    router_qualities: list[float] = []
    router_costs: list[float] = []
    best_single_qualities: list[float] = []
    best_single_costs: list[float] = []
    overheads_ms: list[float] = []

    for idx, row in enumerate(unique_test.itertuples(index=False), start=1):
        q = str(row.query)
        task = str(row.task_name)
        dec = runtime_router.route(q, lam=lam, task_category=task)

        obs = test_lookup.loc[(q, dec.chosen_model)]
        obs_q = float(obs["performance"])
        obs_c = float(obs["reconstructed_cost_usd"])

        best_obs = test_lookup.loc[(q, val_best_model)]
        best_single_qualities.append(float(best_obs["performance"]))
        best_single_costs.append(float(best_obs["reconstructed_cost_usd"]))

        chosen_counts[dec.chosen_model] += 1
        router_qualities.append(obs_q)
        router_costs.append(obs_c)
        overheads_ms.append(dec.router_latency_ms)

        pred_q = dec.predicted_scores.get(dec.chosen_model)
        pred_str = f"{pred_q:.3f}" if pred_q is not None else "  N/A "
        print(
            f"{idx:>2} | {task:<16} | {dec.chosen_model:<28} | {pred_str:>6} | "
            f"{obs_q:>6.3f} | ${obs_c:>9.6f} | {'No':>3} | {dec.router_version}"
        )

    total_cost = float(np.sum(router_costs))
    cost_per_1k = float(np.mean(router_costs) * 1000.0)
    best_cost_per_1k = float(np.mean(best_single_costs) * 1000.0)
    router_q = float(np.mean(router_qualities))
    best_q = float(np.mean(best_single_qualities))
    p50_ms = float(np.percentile(overheads_ms, 50))
    p95_ms = float(np.percentile(overheads_ms, 95))
    savings_pct = ((best_cost_per_1k - cost_per_1k) / best_cost_per_1k * 100.0) if best_cost_per_1k > 0 else 0.0

    print("\n" + "=" * 88)
    print("50-Prompt Offline Replay Summary")
    print("=" * 88)
    print(f"Total prompts:               {len(unique_test)}")
    print(f"Lambda / preset:             {lam} (resolved={dec.lambda_value})")
    print(f"Model selection distribution:{dict(chosen_counts)}")
    print(f"Total actual historical cost:${total_cost:.6f}")
    print(f"Router cost / 1k requests:   ${cost_per_1k:.4f}")
    print(f"Router p50 / p95 overhead:   {p50_ms:.2f} ms / {p95_ms:.2f} ms")
    print(f"Router quality (mean):       {router_q:.4f}")
    print(f"Val-Best Single ({val_best_model}): {best_q:.4f} (cost/1k=${best_cost_per_1k:.4f})")
    print(f"Cost savings vs Val-Best:    {savings_pct:.2f}% (quality delta={router_q - best_q:+.4f})")

    return {
        "total_prompts": len(unique_test),
        "router_quality": router_q,
        "val_best_model": val_best_model,
        "val_best_quality": best_q,
        "router_cost_per_1k": cost_per_1k,
        "val_best_cost_per_1k": best_cost_per_1k,
        "savings_pct": savings_pct,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 50-prompt M5 offline routing demo.")
    parser.add_argument("--lam", default="auto:balanced", help="Lambda preset or float")
    parser.add_argument("--prompts", type=int, default=50, help="Number of test prompts")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    args = parser.parse_args()
    run_demo(lam=args.lam, n_prompts=args.prompts, seed=args.seed)
