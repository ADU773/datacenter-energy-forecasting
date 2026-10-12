"""Model registry: versioned model artefacts plus aliases such as ``production``.

Models live under ``runs/registry/<name>/v<version>/`` (paths are stored relative to the
runs directory, so the folder can be moved) and are indexed in the same SQLite file
as the experiment tracker, so a version always points back to the run that produced it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from dcforecast.models import Forecaster, load_model
from dcforecast.tracking.tracker import connect, utcnow
from dcforecast.utils import get_logger, relpath, to_jsonable

log = get_logger(__name__)

PRODUCTION = "production"


class ModelNotFoundError(LookupError):
    """Raised when a model, version or alias does not exist."""


class ModelRegistry:
    """Register, look up, promote and load models."""

    def __init__(self, runs_dir: Path | str) -> None:
        self.runs_dir = Path(runs_dir)
        self.db_path = self.runs_dir / "tracker.db"
        self.root = self.runs_dir / "registry"

    def register(
        self, model: Forecaster, run_id: str | None = None, metrics: dict[str, Any] | None = None
    ) -> int:
        """Save ``model`` as the next version of ``model.name`` and return the version."""
        with connect(self.db_path) as con:
            row = con.execute(
                "SELECT COALESCE(MAX(version), 0) FROM models WHERE name = ?", (model.name,)
            ).fetchone()
            version = int(row[0]) + 1
            path = self.root / model.name / f"v{version}"
            if path.exists():
                shutil.rmtree(path)
            model.save(path)
            con.execute(
                "INSERT INTO models (name, version, run_id, path, created_at, metrics)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    model.name,
                    version,
                    run_id,
                    path.relative_to(self.runs_dir).as_posix(),
                    utcnow(),
                    json.dumps(to_jsonable(metrics or {})),
                ),
            )
        log.info("registered %s v%d -> %s", model.name, version, relpath(path))
        return version

    def list_models(self) -> pd.DataFrame:
        """All registered versions with the aliases pointing at them."""
        with connect(self.db_path) as con:
            rows = [dict(r) for r in con.execute("SELECT * FROM models ORDER BY name, version")]
            aliases = [dict(r) for r in con.execute("SELECT * FROM aliases")]
        for r in rows:
            r["metrics"] = json.loads(r["metrics"]) if r["metrics"] else {}
            r["aliases"] = [
                a["alias"]
                for a in aliases
                if a["name"] == r["name"] and a["version"] == r["version"]
            ]
        return pd.DataFrame(rows)

    def latest_version(self, name: str) -> int:
        with connect(self.db_path) as con:
            row = con.execute("SELECT MAX(version) FROM models WHERE name = ?", (name,)).fetchone()
        if row is None or row[0] is None:
            raise ModelNotFoundError(f"No registered versions of model '{name}'")
        return int(row[0])

    def get(self, name: str, version: int | None = None) -> dict[str, Any]:
        """Registry entry for ``name`` (latest version by default)."""
        version = version or self.latest_version(name)
        with connect(self.db_path) as con:
            row = con.execute(
                "SELECT * FROM models WHERE name = ? AND version = ?", (name, version)
            ).fetchone()
        if row is None:
            raise ModelNotFoundError(f"Model '{name}' v{version} is not registered")
        entry = dict(row)
        entry["metrics"] = json.loads(entry["metrics"]) if entry["metrics"] else {}
        entry["abs_path"] = self.runs_dir / entry["path"]
        return entry

    def set_alias(
        self, alias: str, name: str, version: int | None = None, reason: str = ""
    ) -> dict:
        """Point ``alias`` (e.g. ``production``) at a model version."""
        entry = self.get(name, version)
        with connect(self.db_path) as con:
            con.execute(
                "INSERT INTO aliases (alias, name, version, updated_at, reason) VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(alias) DO UPDATE SET name=excluded.name, version=excluded.version,"
                " updated_at=excluded.updated_at, reason=excluded.reason",
                (alias, name, entry["version"], utcnow(), reason),
            )
        log.info("alias '%s' -> %s v%d (%s)", alias, name, entry["version"], reason)
        return entry

    def resolve(self, ref: str) -> dict[str, Any]:
        """Resolve an alias, ``name`` or ``name:version`` to a registry entry."""
        with connect(self.db_path) as con:
            row = con.execute("SELECT * FROM aliases WHERE alias = ?", (ref,)).fetchone()
        if row is not None:
            return {**self.get(row["name"], int(row["version"])), "alias": ref}
        if ":" in ref:
            name, ver = ref.split(":", 1)
            return self.get(name, int(ver.lstrip("v")))
        try:
            return self.get(ref)
        except ModelNotFoundError:
            raise ModelNotFoundError(
                f"'{ref}' is neither an alias nor a registered model -- run `dcforecast train` "
                "and `dcforecast promote` first"
            ) from None

    def aliases(self) -> dict[str, dict[str, Any]]:
        with connect(self.db_path) as con:
            rows = [dict(r) for r in con.execute("SELECT * FROM aliases")]
        return {r["alias"]: r for r in rows}

    def load(self, ref: str = PRODUCTION, device: str | None = "cpu") -> tuple[Forecaster, dict]:
        """Load the model behind ``ref`` (alias / name / name:version)."""
        entry = self.resolve(ref)
        model = load_model(Path(entry["abs_path"]), device=device)
        return model, entry
