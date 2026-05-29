"""Tests for WindNinja invocation configuration.

Covers the per-invocation thread count, which combines with the worker pool
size to determine total load on a (Slurm-managed) PACE allocation.
"""

import pytest

from embrs.models.wind_forecast import windninja_num_threads


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("EMBRS_PACE", "WINDNINJA_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    yield


def test_default_off_cluster():
    assert windninja_num_threads() == 4


def test_defaults_to_one_under_pace(monkeypatch):
    monkeypatch.setenv("EMBRS_PACE", "1")
    assert windninja_num_threads() == 1


def test_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("EMBRS_PACE", "1")
    monkeypatch.setenv("WINDNINJA_NUM_THREADS", "8")
    assert windninja_num_threads() == 8


def test_invalid_override_ignored(monkeypatch):
    monkeypatch.setenv("WINDNINJA_NUM_THREADS", "garbage")
    assert windninja_num_threads() == 4
