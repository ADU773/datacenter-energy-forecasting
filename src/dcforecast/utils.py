"""Small shared helpers: logging, seeding, timing, JSON I/O and git metadata."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logging once (ASCII-only, stderr)."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if not any(getattr(h, "_dcforecast", False) for h in root.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        handler._dcforecast = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("matplotlib", "PIL", "urllib3", "httpx", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger under the ``dcforecast`` namespace."""
    return logging.getLogger(name if name.startswith("dcforecast") else f"dcforecast.{name}")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and (if importable) torch; optionally force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 - seeds third-party code using the global RNG
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels make runs bit-reproducible but are slower; otherwise let
    # cuDNN benchmark and pick the fastest convolution algorithms.
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


@dataclass
class Timer:
    """Wall-clock timer usable as a context manager."""

    start: float = field(default_factory=time.perf_counter)
    elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start


@contextmanager
def log_duration(logger: logging.Logger, what: str) -> Iterator[Timer]:
    """Log how long a pipeline step took."""
    timer = Timer()
    logger.info("%s ...", what)
    with timer:
        yield timer
    logger.info("%s done in %.1fs", what, timer.elapsed)


def to_jsonable(obj: Any) -> Any:
    """Convert numpy / pandas / Path objects into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def write_json(path: Path, payload: Any) -> Path:
    """Write pretty JSON (UTF-8), creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(payload), fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def git_commit(cwd: Path | None = None) -> str | None:
    """Return the short git commit hash of the working tree, if any."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


def relpath(path: Path) -> str:
    """Path relative to the CWD when possible (keeps absolute local paths out of artifacts)."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path.cwd().resolve())).as_posix()
    except ValueError:  # different drive on Windows
        return Path(path).name


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process in MB (peak working set on Windows)."""
    try:
        import psutil

        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
        if peak:
            return round(peak / 2**20, 1)
    except Exception:  # pragma: no cover - psutil is a hard dependency
        pass
    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(maxrss / (2**20 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # pragma: no cover
        return None


def dir_size_mb(path: Path) -> float:
    """Total size of all files below ``path`` in MB (0 if it does not exist)."""
    path = Path(path)
    if not path.exists():
        return 0.0
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20, 2)
