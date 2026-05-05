"""Seed-determinism regression tests for EMBRS (no firefighting controller).

Two tests, both xfailed until later phases land:
  - test_same_seed_same_outcome:        unblocks after Phase 2 (core RNG plumbed).
  - test_different_seed_different_outcome: unblocks after Phase 2 as well.

These tests run a small EMBRS scenario twice and hash the final fire grid via
``hash_fire_grid`` from the firefighting test helper module. They do NOT
exercise the firefighting ControlClass; the parallel test in
applications/firefighting/tests/test_determinism.py covers the integrated path.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="unblocks at Phase 2 (EMBRS core RNG plumbing)", strict=False)
def test_same_seed_same_outcome(tmp_path):
    """Same seed → byte-identical final fire grid hash across two runs."""
    pytest.skip("Phase 1: helpers stubbed; reactivate after Phase 2 lands.")


@pytest.mark.xfail(reason="unblocks at Phase 2 (EMBRS core RNG plumbing)", strict=False)
def test_different_seed_different_outcome(tmp_path):
    """Different seeds → different final fire grid hash (sanity check)."""
    pytest.skip("Phase 1: helpers stubbed; reactivate after Phase 2 lands.")
