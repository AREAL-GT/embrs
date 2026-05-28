"""Tests for embrs.fire_danger.weather_loader."""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from embrs.fire_danger.weather_loader import load_wxs


@pytest.fixture
def english_wxs(tmp_path):
    p = tmp_path / "tiny.wxs"
    p.write_text(textwrap.dedent("""\
        RAWS_UNITS: English
        RAWS_ELEVATION: 4200
        RAWS: 1
        Year  Mth  Day   Time    Temp     RH  HrlyPcp  WindSpd WindDir CloudCov
        2025  7    22    0000    57.7     39    0.00      4.9     175       4
        2025  7    22    0100    58.1     38    0.10      5.0     180      10
        2025  7    22    0200    60.0     35    0.00      6.0     185      20
    """))
    return p


def test_english_parse_basic_shape(english_wxs):
    w = load_wxs(str(english_wxs))
    assert w.time_step_min == 60
    assert len(w.df) == 3
    assert list(w.df.columns) == [
        "temp_F", "temp_C", "rh_pct", "rh_frac", "wind_mph", "wind_dir_deg",
        "precip_in_hr", "precip_cm_hr", "cloud_cover",
    ]


def test_english_unit_conversions(english_wxs):
    w = load_wxs(str(english_wxs))
    # Temp: 57.7F -> 14.2778C
    assert w.df["temp_F"].iloc[0] == pytest.approx(57.7)
    assert w.df["temp_C"].iloc[0] == pytest.approx((57.7 - 32) * 5 / 9, rel=1e-6)
    # RH: 39% -> 0.39
    assert w.df["rh_frac"].iloc[0] == pytest.approx(0.39)
    # Precip: 0.10 in/h -> 0.254 cm/h
    assert w.df["precip_cm_hr"].iloc[1] == pytest.approx(0.10 * 2.54, rel=1e-9)
    # Wind preserved in mph
    assert w.df["wind_mph"].iloc[0] == pytest.approx(4.9)
    # Elevation: 4200 ft -> ~1280.16 m
    assert w.ref_elev_m == pytest.approx(4200 * 0.3048, rel=1e-6)


def test_non_hourly_rejected(tmp_path):
    p = tmp_path / "halfhour.wxs"
    p.write_text(textwrap.dedent("""\
        RAWS_UNITS: English
        RAWS_ELEVATION: 4200
        RAWS: 1
        Year  Mth  Day   Time    Temp     RH  HrlyPcp  WindSpd WindDir CloudCov
        2025  7    22    0000    57.7     39    0.00      4.9     175       4
        2025  7    22    0030    58.1     38    0.10      5.0     180      10
    """))
    with pytest.raises(ValueError, match="hourly"):
        load_wxs(str(p))


def test_too_few_rows(tmp_path):
    p = tmp_path / "single.wxs"
    p.write_text(textwrap.dedent("""\
        RAWS_UNITS: English
        RAWS_ELEVATION: 4200
        RAWS: 1
        Year  Mth  Day   Time    Temp     RH  HrlyPcp  WindSpd WindDir CloudCov
        2025  7    22    0000    57.7     39    0.00      4.9     175       4
    """))
    with pytest.raises(ValueError, match="fewer than 2"):
        load_wxs(str(p))


def test_missing_elevation(tmp_path):
    p = tmp_path / "noelev.wxs"
    p.write_text(textwrap.dedent("""\
        RAWS_UNITS: English
        RAWS: 1
        Year  Mth  Day   Time    Temp     RH  HrlyPcp  WindSpd WindDir CloudCov
        2025  7    22    0000    57.7     39    0.00      4.9     175       4
        2025  7    22    0100    58.1     38    0.10      5.0     180      10
    """))
    with pytest.raises(ValueError, match="RAWS_ELEVATION"):
        load_wxs(str(p))


def test_malformed_row_skipped_but_file_still_parses(tmp_path):
    p = tmp_path / "withbadrow.wxs"
    p.write_text(textwrap.dedent("""\
        RAWS_UNITS: English
        RAWS_ELEVATION: 4200
        RAWS: 1
        Year  Mth  Day   Time    Temp     RH  HrlyPcp  WindSpd WindDir CloudCov
        2025  7    22    0000    57.7     39    0.00      4.9     175       4
        garbage line with wrong column count
        2025  7    22    0100    58.1     38    0.10      5.0     180      10
        2025  7    22    0200    60.0     35    0.00      6.0     185      20
    """))
    w = load_wxs(str(p))
    assert len(w.df) == 3


def test_real_long_sample_loads():
    """Smoke test against the canonical long sample (62.6 days)."""
    w = load_wxs("/Users/rjdp3/Documents/Research/embrs_weather/long_weather_example.wxs")
    assert w.time_step_min == 60
    assert len(w.df) >= 1500
    # CloudCov should hit the full percent range somewhere in 62 days
    assert w.df["cloud_cover"].max() == pytest.approx(100.0)
    assert w.df["cloud_cover"].min() == pytest.approx(0.0)
    # No NaNs in canonical columns
    assert not w.df.isna().any().any()
