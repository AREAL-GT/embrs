"""Tests for runtime environment detection.

Validates the PACE/headless/offline switches and scheduler-aware CPU
counting in ``embrs.utilities.runtime_env``.
"""

import os

import pytest

from embrs.utilities import runtime_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start each test from a known-clean environment."""
    for var in ("EMBRS_PACE", "EMBRS_HEADLESS", "DISPLAY", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(var, raising=False)
    yield


class TestPaceAndHeadless:
    def test_pace_flag(self, monkeypatch):
        assert runtime_env.is_pace() is False
        monkeypatch.setenv("EMBRS_PACE", "1")
        assert runtime_env.is_pace() is True

    def test_pace_implies_headless(self, monkeypatch):
        monkeypatch.setenv("EMBRS_PACE", "1")
        assert runtime_env.is_headless() is True

    def test_explicit_headless_flag(self, monkeypatch):
        monkeypatch.setenv("EMBRS_HEADLESS", "1")
        assert runtime_env.is_headless() is True

    def test_linux_no_display_is_headless(self, monkeypatch):
        monkeypatch.setattr(runtime_env.sys, "platform", "linux")
        # DISPLAY already cleared by fixture
        assert runtime_env.is_headless() is True

    def test_linux_with_display_not_headless(self, monkeypatch):
        monkeypatch.setattr(runtime_env.sys, "platform", "linux")
        monkeypatch.setenv("DISPLAY", ":0")
        assert runtime_env.is_headless() is False

    def test_macos_without_display_not_headless(self, monkeypatch):
        monkeypatch.setattr(runtime_env.sys, "platform", "darwin")
        assert runtime_env.is_headless() is False


class TestSolarSource:
    def test_default_off_cluster(self):
        assert runtime_env.default_solar_source() == "openmeteo"

    def test_default_on_pace(self, monkeypatch):
        monkeypatch.setenv("EMBRS_PACE", "1")
        assert runtime_env.default_solar_source() == "offline"


class TestAvailableCpus:
    def test_returns_at_least_one(self):
        assert runtime_env.available_cpus() >= 1

    def test_honors_slurm_cpus_per_task(self, monkeypatch):
        # Force the sched_getaffinity branch to look large so SLURM wins.
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)),
                            raising=False)
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
        assert runtime_env.available_cpus() == 4

    def test_takes_min_of_affinity_and_slurm(self, monkeypatch):
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1},
                            raising=False)
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        assert runtime_env.available_cpus() == 2

    def test_falls_back_to_cpu_count(self, monkeypatch):
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 6)
        assert runtime_env.available_cpus() == 6
