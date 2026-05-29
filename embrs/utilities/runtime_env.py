"""Runtime environment detection for EMBRS.

Centralizes the environment-driven switches needed to run EMBRS on headless
HPC clusters (e.g. Georgia Tech's PACE) where there is no display, no
outbound internet on compute nodes, and CPU allocation is managed by a
scheduler (Slurm) rather than the physical node's core count.

Environment Variables:
    EMBRS_PACE: Set to '1' as a convenience switch on PACE. Flips the
        defaults for headless mode (on) and the weather solar source
        ('offline'). Each can still be overridden individually.
    EMBRS_HEADLESS: Set to '1' to force headless mode (no Tk windows)
        regardless of display availability.

Notes:
    CPU detection is always scheduler-aware; it is a correctness fix rather
    than a PACE-only behavior and is not gated behind any flag.
"""

import os
import sys


def _env_flag(name: str) -> bool:
    """Return True if the named environment variable is set to '1'."""
    return os.environ.get(name, "0") == "1"


def is_pace() -> bool:
    """Whether the EMBRS_PACE convenience switch is enabled."""
    return _env_flag("EMBRS_PACE")


def is_headless() -> bool:
    """Whether EMBRS should avoid opening Tk windows.

    True when ``EMBRS_HEADLESS=1`` or ``EMBRS_PACE=1`` is set, or — on
    Linux — when no X display is available. macOS/Windows are only
    considered headless when explicitly requested, since Tk does not
    require ``$DISPLAY`` there.
    """
    if _env_flag("EMBRS_HEADLESS") or is_pace():
        return True
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return True
    return False


def default_solar_source() -> str:
    """Default weather solar source when the .cfg does not specify one.

    'offline' under EMBRS_PACE (compute nodes have no internet), otherwise
    'openmeteo' to preserve legacy behavior.
    """
    return "offline" if is_pace() else "openmeteo"


def available_cpus() -> int:
    """Number of CPUs actually usable by this process.

    Prefers the most conservative of the scheduler's CPU pinning
    (``os.sched_getaffinity`` on Linux) and ``SLURM_CPUS_PER_TASK`` so a
    job does not oversubscribe a shared node. Falls back to
    ``os.cpu_count()`` off-cluster. Always at least 1.
    """
    counts = []
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except AttributeError:
        pass  # not available on macOS/Windows

    slurm = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if slurm.isdigit():
        counts.append(int(slurm))

    if not counts:
        counts.append(os.cpu_count() or 1)

    return max(1, min(counts))
