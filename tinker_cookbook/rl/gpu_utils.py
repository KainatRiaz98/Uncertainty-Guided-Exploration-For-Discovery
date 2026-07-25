"""
Multi-GPU aware memory helpers.

torch.cuda.memory_allocated() / empty_cache() act on the CURRENT device only.
On the single-GPU path that is the whole story; once a run is sharded across
several GPUs those calls silently report and free only a fraction of the run,
which makes the gpu/* metrics misleading exactly when they matter most (they
are what shows whether a big-model run was near OOM).
"""

from typing import Dict, List

import torch


def visible_devices() -> List[int]:
    """Indices of every CUDA device visible to this process."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def empty_cache_all() -> None:
    """Release the cached allocator pool on every visible device."""
    if not torch.cuda.is_available():
        return
    for idx in visible_devices():
        with torch.cuda.device(idx):
            torch.cuda.empty_cache()


def reset_peak_stats_all() -> None:
    for idx in visible_devices():
        torch.cuda.reset_peak_memory_stats(idx)


def memory_metrics(prefix: str = "gpu") -> Dict[str, float]:
    """
    Per-device and total memory metrics, in GB.

    Emits `<prefix>/memory_allocated_gb` etc. as the SUM across devices (so the
    existing single-GPU dashboards keep working), plus `<prefix>/dev<i>/...`
    per device so an imbalanced shard layout is visible.
    """
    devices = visible_devices()
    if not devices:
        return {}

    out: Dict[str, float] = {}
    total_alloc = total_reserved = total_peak = 0.0

    for idx in devices:
        alloc = torch.cuda.memory_allocated(idx) / 1e9
        reserved = torch.cuda.memory_reserved(idx) / 1e9
        peak = torch.cuda.max_memory_allocated(idx) / 1e9
        total_alloc += alloc
        total_reserved += reserved
        total_peak += peak
        if len(devices) > 1:
            out[f"{prefix}/dev{idx}/memory_allocated_gb"] = alloc
            out[f"{prefix}/dev{idx}/memory_reserved_gb"] = reserved
            out[f"{prefix}/dev{idx}/memory_peak_gb"] = peak

    out[f"{prefix}/memory_allocated_gb"] = total_alloc
    out[f"{prefix}/memory_reserved_gb"] = total_reserved
    out[f"{prefix}/memory_peak_gb"] = total_peak
    return out


def format_memory_line() -> str:
    """One-line human-readable per-device summary for the training log."""
    devices = visible_devices()
    if not devices:
        return "no CUDA devices"
    parts = []
    for idx in devices:
        alloc = torch.cuda.memory_allocated(idx) / 1e9
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        parts.append(f"cuda:{idx} {alloc:.1f}/{total:.0f}GB")
    return " | ".join(parts)
