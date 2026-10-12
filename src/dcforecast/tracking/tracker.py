"""A small, dependency-free experiment tracker backed by SQLite (``runs/tracker.db``).

Each run stores: kind (train / backtest / benchmark / ...), model, params, metrics, duration,
device, git commit (if the project is a git checkout), profile, artifact paths and tags.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dcforecast.utils import get_logger, git_commit, relpath, to_jsonable

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    model       TEXT,
    status      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    duration_s  REAL,
    device      TEXT,
    git_commit  TEXT,
    profile     TEXT,
    params      TEXT,
    metrics     TEXT,
    artifacts   TEXT,
    tags        TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_kind ON runs(kind, started_at);
CREATE TABLE IF NOT EXISTS models (
    name        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    run_id      TEXT,
    path        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    metrics     TEXT,
    PRIMARY KEY (name, version)
);
CREATE TABLE IF NOT EXISTS aliases (
    alias       TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    updated_at  TEXT NOT NULL,
    reason      TEXT
);
"""


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open (and initialise) the tracker database; commit on success and always close.

    Closing explicitly matters on Windows, where an open handle keeps the file locked.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30)
    try:
        con.row_factory = sqlite3.Row
        con.executescript(SCHEMA)
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


@dataclass
class Run:
    """Handle for an active run (collects metrics / artifacts until it is closed)."""

    run_id: str
    kind: str
    model: str | None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    device: str | None = None

    def log_params(self, **params: Any) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, Any] | None = None, **kw: Any) -> None:
        self.metrics.update(metrics or {})
        self.metrics.update(kw)

    def log_artifact(self, name: str, path: Path | str) -> None:
        self.artifacts[name] = relpath(Path(path))

    def set_device(self, device: str) -> None:
        self.device = str(device)


class Tracker:
    """Create, list and compare runs."""

    def __init__(self, runs_dir: Path | str, profile: str | None = None) -> None:
        self.runs_dir = Path(runs_dir)
        self.db_path = self.runs_dir / "tracker.db"
        self.profile = profile

    def _con(self) -> AbstractContextManager[sqlite3.Connection]:
        return connect(self.db_path)

    @staticmethod
    def new_run_id() -> str:
        return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    @contextmanager
    def start_run(
        self,
        kind: str,
        model: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Iterator[Run]:
        """Context manager: the run is marked finished/failed with its duration on exit."""
        run = Run(self.new_run_id(), kind, model, dict(params or {}), tags=dict(tags or {}))
        started = utcnow()
        t0 = time.perf_counter()
        commit = git_commit()
        with self._con() as con:
            con.execute(
                "INSERT INTO runs (run_id, kind, model, status, started_at, git_commit, profile)"
                " VALUES (?, ?, ?, 'running', ?, ?, ?)",
                (run.run_id, kind, model, started, commit, self.profile),
            )
        status = "failed"
        try:
            yield run
            status = "finished"
        finally:
            with self._con() as con:
                con.execute(
                    "UPDATE runs SET status=?, finished_at=?, duration_s=?, device=?, params=?,"
                    " metrics=?, artifacts=?, tags=? WHERE run_id=?",
                    (
                        status,
                        utcnow(),
                        round(time.perf_counter() - t0, 3),
                        run.device,
                        json.dumps(to_jsonable(run.params)),
                        json.dumps(to_jsonable(run.metrics)),
                        json.dumps(to_jsonable(run.artifacts)),
                        json.dumps(to_jsonable(run.tags)),
                        run.run_id,
                    ),
                )

    # ------------------------------------------------------------------ queries
    def list_runs(
        self, kind: str | None = None, model: str | None = None, limit: int = 50
    ) -> pd.DataFrame:
        """Most recent runs first."""
        sql = "SELECT * FROM runs WHERE 1=1"
        args: list[Any] = []
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if model:
            sql += " AND model = ?"
            args.append(model)
        sql += " ORDER BY started_at DESC, rowid DESC LIMIT ?"
        args.append(int(limit))
        with self._con() as con:
            rows = [dict(r) for r in con.execute(sql, args).fetchall()]
        for r in rows:
            for key in ("params", "metrics", "artifacts", "tags"):
                r[key] = json.loads(r[key]) if r.get(key) else {}
        return pd.DataFrame(rows)

    def get_run(self, run_id: str) -> dict[str, Any]:
        """A single run as a dict (prefix match on the run id is allowed)."""
        with self._con() as con:
            rows = con.execute(
                "SELECT * FROM runs WHERE run_id LIKE ? ORDER BY started_at DESC", (run_id + "%",)
            ).fetchall()
        if not rows:
            raise KeyError(f"No run matches '{run_id}'")
        if len(rows) > 1:
            raise KeyError(f"Run id prefix '{run_id}' is ambiguous ({len(rows)} matches)")
        r = dict(rows[0])
        for key in ("params", "metrics", "artifacts", "tags"):
            r[key] = json.loads(r[key]) if r.get(key) else {}
        return r

    def compare(self, run_ids: list[str], metrics: list[str] | None = None) -> pd.DataFrame:
        """Side-by-side metrics of several runs (one column per run)."""
        runs = [self.get_run(r) for r in run_ids]
        keys = metrics or sorted(
            {k for r in runs for k, v in r["metrics"].items() if isinstance(v, (int, float))}
        )
        data = {
            f"{r['run_id'][:22]} ({r['model'] or r['kind']})": {
                **{"kind": r["kind"], "device": r["device"], "duration_s": r["duration_s"]},
                **{k: r["metrics"].get(k) for k in keys},
            }
            for r in runs
        }
        return pd.DataFrame(data)
