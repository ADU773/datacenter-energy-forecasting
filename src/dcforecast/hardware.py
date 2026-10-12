"""Hardware discovery and resource sampling (CPU, RAM, GPU via torch + nvidia-smi).

Everything here is privacy-aware: the host name is never collected (published artifacts call
the machine ``host-01``), and only generic hardware model names are reported.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

import psutil

HOST_ALIAS = "host-01"


def cpu_model_name() -> str:
    """Best-effort marketing name of the CPU (e.g. 'AMD Ryzen 7 6800HS ...')."""
    try:
        if sys.platform.startswith("win"):
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return str(value).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.stdout.strip():
                return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown CPU"


def nvidia_smi_query(fields: str) -> list[list[str]]:
    """Run ``nvidia-smi --query-gpu=<fields>`` and return parsed CSV rows (empty if absent)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [[c.strip() for c in line.split(",")] for line in out.stdout.strip().splitlines()]


@dataclass
class GPUInfo:
    """One CUDA/MPS device."""

    name: str
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    compute_capability: str | None = None


@dataclass
class HardwareInfo:
    """Snapshot of the machine the pipeline runs on (no host name, no paths)."""

    host: str = HOST_ALIAS
    os: str = ""
    python: str = ""
    torch: str | None = None
    cpu_model: str = ""
    cpu_physical_cores: int | None = None
    cpu_logical_cores: int | None = None
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpus: list[GPUInfo] = field(default_factory=list)
    accelerator: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_hardware() -> HardwareInfo:
    """Collect CPU / RAM / GPU information."""
    vm = psutil.virtual_memory()
    info = HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        cpu_model=cpu_model_name(),
        cpu_physical_cores=psutil.cpu_count(logical=False),
        cpu_logical_cores=psutil.cpu_count(logical=True),
        ram_total_gb=round(vm.total / 1e9, 2),
        ram_available_gb=round(vm.available / 1e9, 2),
    )
    try:
        import torch

        info.torch = torch.__version__
        if torch.cuda.is_available():
            info.accelerator = "cuda"
            driver = None
            rows = nvidia_smi_query("driver_version")
            if rows:
                driver = rows[0][0]
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                free, total = torch.cuda.mem_get_info(idx)
                info.gpus.append(
                    GPUInfo(
                        name=props.name,
                        vram_total_gb=round(total / 1024**3, 2),
                        vram_free_gb=round(free / 1024**3, 2),
                        cuda_version=torch.version.cuda,
                        driver_version=driver,
                        compute_capability=f"{props.major}.{props.minor}",
                    )
                )
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info.accelerator = "mps"
            info.gpus.append(GPUInfo(name="Apple MPS"))
    except ImportError:  # pragma: no cover
        pass
    return info


def format_hardware(info: HardwareInfo) -> str:
    """Human-readable, ASCII-only hardware summary."""
    lines = [
        f"Host           : {info.host} (anonymised)",
        f"OS / Python    : {info.os} / Python {info.python} / torch {info.torch}",
        f"CPU            : {info.cpu_model}",
        f"CPU cores      : {info.cpu_physical_cores} physical / {info.cpu_logical_cores} logical",
        f"RAM            : {info.ram_total_gb:.1f} GB total, {info.ram_available_gb:.1f} GB available",
        f"Accelerator    : {info.accelerator}",
    ]
    if not info.gpus:
        lines.append("GPU            : none detected (training will use the CPU)")
    for i, gpu in enumerate(info.gpus):
        lines.append(f"GPU {i}          : {gpu.name}")
        if gpu.vram_total_gb is not None:
            lines.append(
                f"  VRAM         : {gpu.vram_total_gb:.2f} GB total, {gpu.vram_free_gb:.2f} GB free"
            )
        if gpu.cuda_version:
            lines.append(f"  CUDA (torch) : {gpu.cuda_version}")
        if gpu.driver_version:
            lines.append(f"  Driver       : {gpu.driver_version}")
        if gpu.compute_capability:
            lines.append(f"  Compute cap. : {gpu.compute_capability}")
    return "\n".join(lines)


class ResourceSampler:
    """Background sampler for process CPU% / RSS (psutil) and GPU utilisation (nvidia-smi).

    GPU utilisation is device-wide as reported by the driver, so it also includes any other
    process using the same GPU.
    """

    def __init__(self, interval_s: float = 0.25, gpu: bool = True) -> None:
        self.interval_s = max(0.05, float(interval_s))
        self.gpu = gpu and shutil.which("nvidia-smi") is not None
        self.cpu_pct: list[float] = []
        self.rss_mb: list[float] = []
        self.gpu_util: list[float] = []
        self.gpu_mem_mb: list[float] = []
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._smi: subprocess.Popen[str] | None = None
        self._proc = psutil.Process()

    def _sample_process(self) -> None:
        self._proc.cpu_percent(None)
        while not self._stop.wait(self.interval_s):
            try:
                self.cpu_pct.append(self._proc.cpu_percent(None))
                self.rss_mb.append(self._proc.memory_info().rss / 1024**2)
            except psutil.Error:  # pragma: no cover
                break

    def _sample_gpu(self) -> None:
        exe = shutil.which("nvidia-smi")
        if exe is None:  # pragma: no cover
            return
        ms = max(100, int(self.interval_s * 1000))
        try:
            self._smi = subprocess.Popen(
                [
                    exe,
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    f"-lms={ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:  # pragma: no cover
            return
        assert self._smi.stdout is not None
        for line in self._smi.stdout:
            if self._stop.is_set():
                break
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    self.gpu_util.append(float(parts[0]))
                    self.gpu_mem_mb.append(float(parts[1]))
                except ValueError:
                    continue

    def __enter__(self) -> ResourceSampler:
        self._stop.clear()
        t = threading.Thread(target=self._sample_process, daemon=True)
        t.start()
        self._threads.append(t)
        if self.gpu:
            g = threading.Thread(target=self._sample_gpu, daemon=True)
            g.start()
            self._threads.append(g)
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._smi is not None:
            self._smi.terminate()
            try:
                self._smi.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._smi.kill()
        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()

    def summary(self) -> dict[str, float | None]:
        """Mean / max of everything that was sampled."""

        def _stat(values: list[float], fn: Any) -> float | None:
            return round(float(fn(values)), 2) if values else None

        def _mean(v: list[float]) -> float:
            return sum(v) / len(v)

        return {
            "cpu_percent_mean": _stat(self.cpu_pct, _mean),
            "cpu_percent_max": _stat(self.cpu_pct, max),
            "rss_mb_max": _stat(self.rss_mb, max),
            "gpu_util_percent_mean": _stat(self.gpu_util, _mean),
            "gpu_util_percent_max": _stat(self.gpu_util, max),
            "gpu_mem_used_mb_max": _stat(self.gpu_mem_mb, max),
            "n_samples": float(len(self.cpu_pct)),
        }
