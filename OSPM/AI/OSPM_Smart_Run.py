# OSPM_Smart_Run.py
# ========================================================================================================================
# Lightweight runtime allocation and shared-worker-pool planner for OSPM.
# Smart Run sits upstream of the physics engine. It detects how many CPUs the
# current process can actually use, chooses a compact per-model worker cap, and
# defines the dynamic admission policy used by the Spherical scheduler.
#
# The important distinction is that THREADS_PER_MODEL is a per-model maximum,
# not a permanent reservation. Active models share one Julia worker pool. Smart
# Run keeps one model-width of headroom when admitting new models so existing
# models can temporarily expand back toward their full worker cap during highly
# parallel phases such as finalization. It does not change science settings.
# ========================================================================================================================

from dataclasses import dataclass
import os
from typing import Optional

@dataclass(frozen=True)
class ResourceSnapshot:
    available_cpus: int
    affinity_cpus: Optional[int]
    scheduler_cpus: Optional[int]
    source: str
    julia_threads: Optional[int] = None
    system_cpus: Optional[int] = None

@dataclass(frozen=True)
class AllocationPlan:
    available_cpus: int
    threads_per_model: int
    model_owner_limit: int
    used_cpus: int
    idle_cpus: int
    min_cpus_per_model: int
    max_cpus_per_model: int
    preferred_cpus_per_model: int

@dataclass(frozen=True)
class SharedPoolPlan:
    total_workers: int
    workers_per_model: int
    initial_model_owners: int
    reserve_workers: int
    admission_worker_limit: int
    dynamic_admission: bool
    whole_model_admission: bool
    finishing_priority: bool


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _scheduler_cpu_limit():
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "PBS_NCPUS", "NSLOTS"):
        value = _positive_int(os.environ.get(name))
        if value is not None:
            return value
    return None


def detect_available_cpus(julia_threads=None):
    scheduler_cpus = _scheduler_cpu_limit()
    system_cpus = _positive_int(os.cpu_count()) or 1
    affinity_cpus = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_cpus = len(os.sched_getaffinity(0))
        except OSError:
            affinity_cpus = None
    julia_threads = _positive_int(julia_threads)
    limits = []
    if scheduler_cpus is not None:
        limits.append((scheduler_cpus, "scheduler_cpus"))
    if affinity_cpus is not None and affinity_cpus > 0:
        limits.append((affinity_cpus, "affinity_cpus"))
    if julia_threads is not None:
        limits.append((julia_threads, "julia_threads"))
    if not limits:
        limits.append((system_cpus, "system_cpus"))
    available_cpus, source = min(limits, key=lambda item: item[0])
    return ResourceSnapshot(available_cpus=available_cpus, affinity_cpus=affinity_cpus, scheduler_cpus=scheduler_cpus, source=source, julia_threads=julia_threads, system_cpus=system_cpus)


def plan_smart_run(snapshot, preferred_cpus_per_model, min_cpus_per_model=5, max_cpus_per_model=20, max_model_owners=None):
    available = int(snapshot.available_cpus)
    minimum = int(min_cpus_per_model)
    maximum = int(max_cpus_per_model)
    preferred = int(preferred_cpus_per_model)
    minimum > 0 or _raise_value("min_cpus_per_model must be positive")
    maximum >= minimum or _raise_value("max_cpus_per_model must be >= min_cpus_per_model")
    if available < minimum:
        raise RuntimeError(f"Smart Run requires at least {minimum} CPUs, but only {available} are available")
    maximum = min(maximum, available)
    preferred = max(minimum, min(preferred, maximum))
    owner_limit_value = None if max_model_owners is None else int(max_model_owners)
    owner_limit = None if owner_limit_value is None or owner_limit_value <= 0 else owner_limit_value
    candidates = []
    for threads_per_model in range(minimum, maximum + 1):
        owners = max(1, available // threads_per_model)
        if owner_limit is not None:
            owners = min(owners, owner_limit)
        used = owners * threads_per_model
        idle = available - used
        candidates.append((threads_per_model, owners, used, idle))
    if owner_limit is not None:
        target_owners = min(owner_limit, available // minimum)
        candidates = [candidate for candidate in candidates if candidate[1] == target_owners]
    threads_per_model, owners, used, _ = min(candidates, key=lambda candidate: (available - candidate[2], abs(candidate[0] - preferred), -candidate[0]))
    return AllocationPlan(available_cpus=available, threads_per_model=threads_per_model, model_owner_limit=owners, used_cpus=used, idle_cpus=available - used, min_cpus_per_model=minimum, max_cpus_per_model=maximum, preferred_cpus_per_model=preferred)


def plan_shared_pool(allocation):
    total_workers = int(allocation.available_cpus)
    workers_per_model = int(allocation.threads_per_model)
    initial_model_owners = int(allocation.model_owner_limit)
    total_workers > 0 or _raise_value("available worker count must be positive")
    workers_per_model > 0 or _raise_value("workers_per_model must be positive")
    initial_model_owners > 0 or _raise_value("initial_model_owners must be positive")
    reserve_workers = min(workers_per_model, max(0, total_workers - workers_per_model))
    admission_worker_limit = total_workers - reserve_workers
    return SharedPoolPlan(total_workers=total_workers, workers_per_model=workers_per_model, initial_model_owners=initial_model_owners, reserve_workers=reserve_workers, admission_worker_limit=admission_worker_limit, dynamic_admission=True, whole_model_admission=True, finishing_priority=True)


def should_admit_model(pool, active_workers):
    active = max(0, int(active_workers))
    return active + int(pool.workers_per_model) <= int(pool.admission_worker_limit)


def _raise_value(message):
    raise ValueError(message)
