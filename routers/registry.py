"""
routers/registry.py

Versioned router bundle store with atomic active-router pointer updates and rollback.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from eval.routers.encoders import BaseEncoder
from routers.base import RuntimeRouter
from routers.knn_runtime import KNNRuntimeRouter
from routers.lr_runtime import LearnedLRRuntimeRouter


class RouterVersionManager:
    """Manages versioned router artifacts, atomic promotion, and rollback."""

    def __init__(self, store_dir: str | Path = "artifacts/routers") -> None:
        self.store_dir = Path(store_dir)
        self.pointer_path = self.store_dir / "active_router.json"

    def version_dir(self, version: str) -> Path:
        return self.store_dir / version

    def list_versions(self) -> list[str]:
        if not self.store_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.store_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").exists()
        )

    def get_pointer(self) -> dict[str, Any] | None:
        if not self.pointer_path.exists():
            return None
        return json.loads(self.pointer_path.read_text(encoding="utf-8"))

    def get_active_version(self) -> str | None:
        ptr = self.get_pointer()
        return ptr.get("active_version") if ptr else None

    def load_version(
        self,
        version: str,
        *,
        encoder: BaseEncoder | None = None,
    ) -> RuntimeRouter:
        v_dir = self.version_dir(version)
        manifest_path = v_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Router version '{version}' not found at '{v_dir}'")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rtype = manifest.get("router_type")
        if rtype == "knn":
            return KNNRuntimeRouter.load_bundle(v_dir, encoder=encoder)
        if rtype == "learned_lr":
            return LearnedLRRuntimeRouter.load_bundle(v_dir, encoder=encoder)
        raise ValueError(f"Unsupported router_type '{rtype}' in '{manifest_path}'")

    def load_active(self, *, encoder: BaseEncoder | None = None) -> RuntimeRouter:
        active = self.get_active_version()
        if not active:
            raise FileNotFoundError(
                f"No active router pointer found at '{self.pointer_path}'"
            )
        return self.load_version(active, encoder=encoder)

    def _write_pointer_atomic(self, payload: dict[str, Any]) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.store_dir), prefix=".active_router_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.pointer_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

    def set_active(self, version: str) -> dict[str, Any]:
        v_dir = self.version_dir(version)
        if not (v_dir / "manifest.json").exists():
            raise FileNotFoundError(f"Cannot activate missing router version '{version}'")
        current_ptr = self.get_pointer() or {}
        prev = current_ptr.get("active_version")
        payload = {
            "active_version": version,
            "previous_version": prev if prev != version else current_ptr.get("previous_version"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_pointer_atomic(payload)
        return payload

    def rollback(self) -> dict[str, Any]:
        ptr = self.get_pointer()
        if not ptr or not ptr.get("previous_version"):
            raise ValueError("No previous_version available for rollback")
        prev = str(ptr["previous_version"])
        if not (self.version_dir(prev) / "manifest.json").exists():
            raise FileNotFoundError(f"Previous version '{prev}' bundle no longer exists")
        payload = {
            "active_version": prev,
            "previous_version": ptr.get("active_version"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "rolled_back": True,
        }
        self._write_pointer_atomic(payload)
        return payload
