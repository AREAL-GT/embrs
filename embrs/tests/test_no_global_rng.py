"""Lint: no global-RNG / UUID4 / wall-clock-time usage in production code.

Hard rule (this test fails on a violation):
    Production code in EMBRS must not call ``np.random.<X>``, ``numpy.random.<X>``,
    stdlib ``random.<X>``, ``uuid.uuid4()``, or ``time.time()``. Randomness must
    flow through a ``numpy.random.Generator`` obtained from the FireSim's seed
    sequence (see ``BaseFireSim.child_generator`` / ``child_rng_seed``). IDs must
    come from a deterministic factory. Wall-clock timestamps belong only in
    metadata that is masked from determinism comparisons.

    Construction APIs are allowed:
        np.random.default_rng, np.random.Generator, np.random.SeedSequence,
        np.random.BitGenerator, np.random.{MT19937,PCG64,PCG64DXSM,Philox,SFC64}.

Advisory rule (warn-only, separate test):
    Iterating over a ``set`` (or set-derived view like ``dict.keys() & ...``) in
    a loop body that draws from RNG can make output depend on PYTHONHASHSEED.
    Wrap such loops with ``sorted(...)``. The advisory regex is heuristic — it
    is meant to surface candidates for review, not to fail CI.

Marked xfail until Phases 2–5 close out the existing violations. Remove the
xfail marker once the inventory is empty (or only contains ALLOWLIST entries).
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTS = [REPO_ROOT / "embrs"]

EXCLUDE_DIR_NAMES = {
    "tests",
    "test_code",
    "__pycache__",
    ".git",
    "embrs.egg-info",
    "site",
    "docs_raw",
    "config_files",
    "validation",
    "examples",
}
EXCLUDE_FILE_NAMES = {Path(__file__).name}

# Construction APIs allowed even though they live under np.random.* / numpy.random.*
NP_RANDOM_OK_NAMES = {
    "default_rng", "Generator", "SeedSequence", "BitGenerator",
    "MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64",
}

# Each rule: (compiled regex, human description, optional callable that
# decides whether a regex match is actually a violation).
def _np_random_filter(m: re.Match) -> bool:
    return m.group(1) not in NP_RANDOM_OK_NAMES

NP_RANDOM_RE = re.compile(r"\bnp\.random\.(\w+)")
NUMPY_RANDOM_RE = re.compile(r"\bnumpy\.random\.(\w+)")
STDLIB_RANDOM_CALL_RE = re.compile(
    r"(?:^|[^a-zA-Z_.])random\."
    r"(random|randint|choice|uniform|gauss|sample|shuffle|seed|"
    r"randrange|getrandbits|triangular|betavariate|expovariate|"
    r"gammavariate|lognormvariate|normalvariate|paretovariate|"
    r"vonmisesvariate|weibullvariate)\b"
)
STDLIB_RANDOM_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+random(?:\s+as\s+\w+)?|from\s+random\s+import\s+)",
    re.MULTILINE,
)
UUID4_RE = re.compile(r"\buuid\.uuid4\b")
TIME_TIME_RE = re.compile(r"\btime\.time\(\)")

# Allowlist for narrow, justified exemptions. Format:
# "<repo-relative path>:<line>:<rule>". Each entry should carry a one-line
# justification comment immediately above it.
ALLOWLIST: set[str] = {
    # Numba @njit kernel: threading our owned Generator into the JIT-compiled
    # hot path is non-trivial. Per the seed-determinism plan, this O(1e-4)
    # perturbation is accepted as known small nondeterminism. Phase 2.
    "embrs/models/dead_fuel_moisture.py:454:np.random.*",
}


def _iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
                continue
            if path.name in EXCLUDE_FILE_NAMES:
                continue
            yield path


def _scan_file_for_violations(path: Path) -> List[Tuple[int, str, str]]:
    """Return (lineno, rule_id, line_text) violations in file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    has_stdlib_random_import = bool(STDLIB_RANDOM_IMPORT_RE.search(text))
    out: List[Tuple[int, str, str]] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        # Strip line-level comments to reduce false positives in docs/comments.
        line = raw.split("#", 1)[0]
        if not line.strip():
            continue

        for m in NP_RANDOM_RE.finditer(line):
            if _np_random_filter(m):
                out.append((lineno, "np.random.*", raw.rstrip()))

        for m in NUMPY_RANDOM_RE.finditer(line):
            if m.group(1) not in NP_RANDOM_OK_NAMES:
                out.append((lineno, "numpy.random.*", raw.rstrip()))

        if has_stdlib_random_import and STDLIB_RANDOM_CALL_RE.search(line):
            out.append((lineno, "stdlib random.*", raw.rstrip()))

        if UUID4_RE.search(line):
            out.append((lineno, "uuid.uuid4", raw.rstrip()))

        if TIME_TIME_RE.search(line):
            out.append((lineno, "time.time()", raw.rstrip()))

    # File-level: report `import random` / `from random import` separately.
    for m in STDLIB_RANDOM_IMPORT_RE.finditer(text):
        lineno = text.count("\n", 0, m.start()) + 1
        line_text = text.splitlines()[lineno - 1]
        out.append((lineno, "stdlib random import", line_text.rstrip()))

    return out


def _format_violations(rows: List[Tuple[Path, int, str, str]]) -> str:
    lines = [f"{len(rows)} global-randomness violation(s) found:\n"]
    for path, lineno, rule, src in rows:
        rel = path.relative_to(REPO_ROOT)
        lines.append(f"  {rel}:{lineno}: [{rule}]")
        lines.append(f"      {src}")
    lines.append("")
    lines.append("Each call must be replaced with a Generator obtained via")
    lines.append("BaseFireSim.child_generator(name) or a deterministic ID factory.")
    lines.append("If a line is a justified exception, add it to ALLOWLIST in")
    lines.append("this test file with a written rationale.")
    return "\n".join(lines)


def _collect_violations() -> List[Tuple[Path, int, str, str]]:
    rows: List[Tuple[Path, int, str, str]] = []
    for path in _iter_python_files(ROOTS):
        for lineno, rule, src in _scan_file_for_violations(path):
            rel_key = f"{path.relative_to(REPO_ROOT)}:{lineno}:{rule}"
            if rel_key in ALLOWLIST:
                continue
            rows.append((path, lineno, rule, src))
    rows.sort(key=lambda r: (str(r[0]), r[1]))
    return rows


def test_no_global_rng_in_production_code():
    """Hard enforcement: fail on any non-allowlisted global-RNG/UUID4/time.time call.

    EMBRS reached zero violations at the end of Phase 3 (predictor + forecast
    pool RNG plumbing). The single ALLOWLIST entry is the Numba JIT kernel in
    dead_fuel_moisture.py — see the ALLOWLIST comment for the rationale.
    """
    rows = _collect_violations()
    if rows:
        pytest.fail(_format_violations(rows), pytrace=False)


# -----------------------------------------------------------------------------
# Advisory: iteration-order hazards. Warn-only.
# -----------------------------------------------------------------------------

ITER_OVER_NAMED_SET_RE = re.compile(r"\bfor\s+\w+\s+in\s+\w*[Ss]et\w*\b")
SET_OPS_RE = re.compile(
    r"\.(?:intersection|union|difference|symmetric_difference)\(|"
    r"\.keys\(\)\s*[&|^\-]\s*\w+\.keys\(\)"
)


def _scan_file_for_iter_advisory(path: Path) -> List[Tuple[int, str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: List[Tuple[int, str, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0]
        if ITER_OVER_NAMED_SET_RE.search(line):
            out.append((lineno, "iter-named-set", raw.rstrip()))
        if SET_OPS_RE.search(line):
            out.append((lineno, "set-op", raw.rstrip()))
    return out


def test_iteration_order_advisory():
    """Warn-only: list set-iteration patterns that may affect determinism.

    Heuristic and false-positive prone. Never fails. Hits show up in pytest's
    captured stdout / warnings section so reviewers can spot new occurrences.
    """
    hits: List[Tuple[Path, int, str, str]] = []
    for path in _iter_python_files(ROOTS):
        for lineno, rule, src in _scan_file_for_iter_advisory(path):
            hits.append((path, lineno, rule, src))
    hits.sort(key=lambda r: (str(r[0]), r[1]))

    if not hits:
        return

    msg_lines = [f"\n[iteration-order advisory] {len(hits)} candidate hit(s):"]
    for path, lineno, rule, src in hits:
        rel = path.relative_to(REPO_ROOT)
        msg_lines.append(f"  {rel}:{lineno}: [{rule}]  {src.strip()}")
    msg_lines.append(
        "If iteration order can affect outputs, wrap the iterable in sorted(...)."
    )
    msg = "\n".join(msg_lines)
    print(msg)
    warnings.warn(msg, stacklevel=2)
