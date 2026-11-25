"""
telemetry/decision_log.py

SQLite-backed auditable decision log and outcome feedback store for M5 runtime routing.
Does NOT store raw prompt text by default (stores SHA-256 prompt_hash, derived features,
and optional prompt embedding bytes for offline retraining).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid

import numpy as np

from eval.routers.learned_router import CODE_PATTERN

VALID_FEEDBACK_TYPES = {"judge_score", "user_feedback", "downstream_task_success"}


def compute_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def extract_derived_features(prompt: str) -> dict[str, Any]:
    return {
        "prompt_length": len(prompt),
        "has_code": bool(CODE_PATTERN.search(prompt)),
    }


def normalize_feedback_value(
    feedback_type: str,
    raw_value: Any,
) -> tuple[str, float]:
    """Validate and normalize an outcome feedback value into [0.0, 1.0].

    Conventions:
    - judge_score: numeric in [1, 5], normalized linearly via (score - 1.0) / 4.0
    - user_feedback: 'thumbs_up'/True -> 1.0, 'thumbs_down'/False -> 0.0, or float in [0.0, 1.0]
    - downstream_task_success: bool/int in {0, 1} or float in [0.0, 1.0]
    """
    if feedback_type not in VALID_FEEDBACK_TYPES:
        raise ValueError(
            f"Invalid feedback_type '{feedback_type}'. Must be one of {sorted(VALID_FEEDBACK_TYPES)}"
        )

    if feedback_type == "judge_score":
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"judge_score must be numeric in [1, 5], got {raw_value!r}")
        val = float(raw_value)
        if not (1.0 <= val <= 5.0):
            raise ValueError(f"judge_score must be between 1 and 5, got {val}")
        return str(raw_value), (val - 1.0) / 4.0

    if feedback_type == "user_feedback":
        if isinstance(raw_value, bool):
            return ("thumbs_up" if raw_value else "thumbs_down"), (1.0 if raw_value else 0.0)
        if isinstance(raw_value, str):
            norm_s = raw_value.strip().lower()
            if norm_s in ("thumbs_up", "up", "positive", "good", "1"):
                return norm_s, 1.0
            if norm_s in ("thumbs_down", "down", "negative", "bad", "0"):
                return norm_s, 0.0
            raise ValueError(f"Unrecognized string user_feedback: {raw_value!r}")
        if isinstance(raw_value, (int, float)):
            val = float(raw_value)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Numeric user_feedback must be in [0.0, 1.0], got {val}")
            return str(raw_value), val
        raise ValueError(f"Invalid user_feedback type: {type(raw_value).__name__}")

    # downstream_task_success
    if isinstance(raw_value, bool):
        return str(raw_value).lower(), (1.0 if raw_value else 0.0)
    if isinstance(raw_value, (int, float)):
        val = float(raw_value)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"downstream_task_success must be in [0.0, 1.0], got {val}")
        return str(raw_value), val
    raise ValueError(f"Invalid downstream_task_success value: {raw_value!r}")


class DecisionLogger:
    """SQLite persistence for M5 routing decisions and distinct outcome feedback."""

    def __init__(self, db_path: str | Path = "data/runtime/decision_log.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS routing_decisions (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    prompt_length INTEGER NOT NULL,
                    has_code INTEGER NOT NULL,
                    task_category TEXT NOT NULL,
                    prompt_embedding BLOB,
                    embedding_dim INTEGER,
                    predicted_scores_json TEXT NOT NULL,
                    ranked_candidates_json TEXT NOT NULL,
                    chosen_model TEXT NOT NULL,
                    lambda_value REAL NOT NULL,
                    router_version TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    actual_cost_usd REAL,
                    expected_cost_usd REAL NOT NULL,
                    budget_accounted_cost_usd REAL,
                    latency_ms REAL NOT NULL,
                    router_latency_ms REAL NOT NULL,
                    fallback_attempts_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    quality_status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS outcome_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    raw_value TEXT NOT NULL,
                    normalized_score REAL NOT NULL,
                    notes TEXT,
                    FOREIGN KEY (request_id) REFERENCES routing_decisions(request_id)
                );

                CREATE INDEX IF NOT EXISTS idx_decisions_prompt_hash
                    ON routing_decisions(prompt_hash);
                CREATE INDEX IF NOT EXISTS idx_feedback_request_id
                    ON outcome_feedback(request_id);
                """
            )

    def log_decision(
        self,
        *,
        prompt: str,
        chosen_model: str,
        predicted_scores: dict[str, float | None],
        ranked_candidates: list[str],
        lambda_value: float,
        router_version: str,
        expected_cost_usd: float,
        latency_ms: float,
        router_latency_ms: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        actual_cost_usd: float | None = None,
        budget_accounted_cost_usd: float | None = None,
        fallback_attempts: list[dict[str, Any]] | None = None,
        errors: list[dict[str, Any]] | None = None,
        quality_status: str = "unverified",
        task_category: str = "unknown",
        prompt_embedding: np.ndarray | None = None,
        request_id: str | None = None,
    ) -> str:
        req_id = request_id or f"req_{uuid.uuid4().hex[:16]}"
        ts = datetime.now(timezone.utc).isoformat()
        p_hash = compute_prompt_hash(prompt)
        feats = extract_derived_features(prompt)

        emb_blob = None
        emb_dim = None
        if prompt_embedding is not None:
            arr = np.asarray(prompt_embedding, dtype=np.float32).ravel()
            emb_blob = arr.tobytes()
            emb_dim = int(arr.size)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO routing_decisions (
                    request_id, timestamp, prompt_hash, prompt_length, has_code,
                    task_category, prompt_embedding, embedding_dim,
                    predicted_scores_json, ranked_candidates_json, chosen_model,
                    lambda_value, router_version, input_tokens, output_tokens,
                    actual_cost_usd, expected_cost_usd, budget_accounted_cost_usd,
                    latency_ms, router_latency_ms, fallback_attempts_json,
                    errors_json, quality_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    req_id,
                    ts,
                    p_hash,
                    int(feats["prompt_length"]),
                    int(feats["has_code"]),
                    str(task_category),
                    emb_blob,
                    emb_dim,
                    json.dumps(predicted_scores),
                    json.dumps(ranked_candidates),
                    str(chosen_model),
                    float(lambda_value),
                    str(router_version),
                    input_tokens,
                    output_tokens,
                    actual_cost_usd,
                    float(expected_cost_usd),
                    budget_accounted_cost_usd,
                    float(latency_ms),
                    float(router_latency_ms),
                    json.dumps(fallback_attempts or []),
                    json.dumps(errors or []),
                    str(quality_status),
                ),
            )
        return req_id

    def record_feedback(
        self,
        request_id: str,
        *,
        feedback_type: str,
        raw_value: Any,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Record an outcome feedback label for an existing decision without manufacturing missing labels."""
        raw_str, norm_score = normalize_feedback_value(feedback_type, raw_value)
        fb_id = f"fb_{uuid.uuid4().hex[:16]}"
        ts = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT request_id FROM routing_decisions WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown request_id '{request_id}'")

            conn.execute(
                """
                INSERT INTO outcome_feedback (
                    feedback_id, request_id, timestamp, feedback_type,
                    raw_value, normalized_score, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (fb_id, request_id, ts, feedback_type, raw_str, float(norm_score), notes),
            )
            conn.execute(
                "UPDATE routing_decisions SET quality_status = ? WHERE request_id = ?",
                ("verified", request_id),
            )

        return {
            "feedback_id": fb_id,
            "request_id": request_id,
            "feedback_type": feedback_type,
            "raw_value": raw_str,
            "normalized_score": norm_score,
            "timestamp": ts,
        }

    def get_decision(self, request_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM routing_decisions WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["predicted_scores"] = json.loads(d.pop("predicted_scores_json"))
            d["ranked_candidates"] = json.loads(d.pop("ranked_candidates_json"))
            d["fallback_attempts"] = json.loads(d.pop("fallback_attempts_json"))
            d["errors"] = json.loads(d.pop("errors_json"))
            emb = d.pop("prompt_embedding")
            dim = d.pop("embedding_dim")
            if emb is not None and dim is not None:
                d["prompt_embedding"] = np.frombuffer(emb, dtype=np.float32).copy()
            else:
                d["prompt_embedding"] = None
            fb_rows = conn.execute(
                "SELECT feedback_id, timestamp, feedback_type, raw_value, normalized_score, notes "
                "FROM outcome_feedback WHERE request_id = ? ORDER BY timestamp ASC",
                (request_id,),
            ).fetchall()
            d["feedback"] = [dict(r) for r in fb_rows]
            return d
