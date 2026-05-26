"""Component 3 — hourly Nelson dead fuel moisture for NFDRS size classes.

Drives EMBRS's :class:`embrs.models.dead_fuel_moisture.DeadFuelMoisture`
through ``update_internal(et=1.0, ...)`` — NOT through ``update(...)``, which
derives elapsed time from ``datetime.toordinal()`` and is therefore broken for
sub-day timesteps (plan M0-1). The driver mirrors the EMBRS pattern in
``embrs/fire_simulator/cell.py:608-625``: one ``initializeEnvironment`` call
followed by ``update_internal`` per hour.

Stick parameters are configured to match NFDRS4 (``nfdrs4.cpp:96-117``,
plan M0-2): NFDRS radii, NFDRS adsorption rates, and
``setMaximumLocalMoisture(0.35)``. Initial moisture 0.2 is set via the ``wi``
argument of :meth:`initializeEnvironment`.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from embrs.fire_danger.config import DeadMoistureHourly, HourlyWeather
from embrs.models.dead_fuel_moisture import DeadFuelMoisture


# (radius_cm, adsorption_rate) per NFDRS stick. All four sticks share
# setMaximumLocalMoisture(0.35) and initial moisture 0.2.
_NFDRS_STICK_PARAMS: Dict[str, tuple[float, float]] = {
    "1h":    (0.20, 0.462252733),
    "10h":   (0.64, 0.079548303),
    "100h":  (2.00, 0.06),
    "1000h": (3.81, 0.06),
}

_NFDRS_MAX_LOCAL_MOISTURE = 0.35
_NFDRS_INIT_MOISTURE = 0.20
_BAROMETRIC_PRESSURE = 0.0218  # cal/cm^3 (EMBRS convention; NFDRS4 uses ~0.02180)


def make_nfdrs_sticks() -> Dict[str, DeadFuelMoisture]:
    """Construct four NFDRS-configured dead-fuel-moisture sticks.

    Each stick is built directly (not via the EMBRS factory) so we get
    explicit NFDRS radii including the 1000-hr radius of **3.81 cm**, not the
    EMBRS factory's 6.40 cm. The ``stv`` / ``wmx`` / ``wfilmk`` constructor
    arguments keep EMBRS factory values; ``wmx`` is then overridden to 0.35
    via :meth:`setMaximumLocalMoisture` (the C++ Init's behaviour).
    """
    # Per-stick (stv, wmx, wfilmk) from EMBRS factory bodies (kept as-is for
    # the parameters NFDRS4 does not override). Only wmx is then overridden.
    embrs_stv_wmx_wfilmk = {
        "1h":    (0.006, 0.85, 0.10),
        "10h":   (0.05,  0.60, 0.05),
        "100h":  (5.0,   0.40, 0.005),
        "1000h": (7.5,   0.32, 0.003),
    }
    sticks: Dict[str, DeadFuelMoisture] = {}
    for key, (radius, ads_rate) in _NFDRS_STICK_PARAMS.items():
        stv, wmx, wfilmk = embrs_stv_wmx_wfilmk[key]
        stick = DeadFuelMoisture(radius=radius, stv=stv, wmx=wmx, wfilmk=wfilmk)
        stick.setAdsorptionRate(ads_rate)
        stick.setMaximumLocalMoisture(_NFDRS_MAX_LOCAL_MOISTURE)
        sticks[key] = stick
    return sticks


def compute_dead_moisture(weather: HourlyWeather) -> DeadMoistureHourly:
    """Drive the four NFDRS sticks hourly across the weather table.

    Snow handling per scope §7.1 / nfdrs4.cpp:389-397: on snow hours feed
    ``at=0``, ``rh=0.999``, ``sW=0`` to the stick **and** hold ``rcum`` flat
    (do not add that hour's precip to the cumulative total). When
    ``weather.df['snow']`` is missing every hour is treated as not-snow.

    Output columns are in **percent moisture content**
    (``meanWtdMoisture() * 100``).
    """
    df = weather.df
    n = len(df)

    temp_C = df["temp_C"].to_numpy()
    rh_frac = df["rh_frac"].to_numpy()
    solar = df["solar_wm2"].to_numpy() if "solar_wm2" in df.columns else np.zeros(n)
    precip_cm_hr = df["precip_cm_hr"].to_numpy()
    snow = df["snow"].to_numpy() if "snow" in df.columns else np.zeros(n, dtype=bool)

    sticks = make_nfdrs_sticks()
    stick_keys = ("1h", "10h", "100h", "1000h")
    out_cols = {f"MC{k[:-1] if k.endswith('h') else k}": np.empty(n) for k in stick_keys}
    # Map "1h" -> "MC1", "10h" -> "MC10", "100h" -> "MC100", "1000h" -> "MC1000"
    out_cols = {
        "MC1": np.empty(n),
        "MC10": np.empty(n),
        "MC100": np.empty(n),
        "MC1000": np.empty(n),
    }

    rcum = 0.0
    for i in range(n):
        if snow[i]:
            at_i = 0.0
            rh_i = 0.999
            sW_i = 0.0
            # rcum unchanged
        else:
            at_i = float(temp_C[i])
            rh_i = float(rh_frac[i])
            sW_i = float(solar[i])
            rcum += float(precip_cm_hr[i])

        for key, col in zip(stick_keys, ("MC1", "MC10", "MC100", "MC1000")):
            stick = sticks[key]
            if not stick.initialized():
                stick.initializeEnvironment(
                    at_i, rh_i, sW_i, rcum,                     # env (ta, ha, sr, rc)
                    at_i, rh_i, _NFDRS_INIT_MOISTURE,            # stick (ti, hi, wi)
                    _BAROMETRIC_PRESSURE,
                )
            stick.update_internal(
                1.0,                  # et: 1 hour
                at_i, rh_i, sW_i, rcum, _BAROMETRIC_PRESSURE,
            )
            out_cols[col][i] = stick.meanWtdMoisture() * 100.0

    out_df = pd.DataFrame(out_cols, index=df.index.copy())
    out_df.index.name = df.index.name or "datetime"
    return DeadMoistureHourly(df=out_df)
