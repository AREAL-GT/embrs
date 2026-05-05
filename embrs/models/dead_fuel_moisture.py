"""Dead Fuel Moisture Model.

This module implements the Nelson dead fuel moisture model, which simulates
moisture diffusion through cylindrical fuel sticks using implicit finite
difference methods.

The model tracks moisture content, temperature, and saturation at discrete
nodes along the radius of the fuel stick, updating based on weather conditions.

Performance Note:
    The inner loop calculations are JIT-compiled using Numba when available.
    Set EMBRS_DISABLE_JIT=1 to disable JIT compilation for debugging.
"""

import math
import numpy as np
import datetime
from collections import OrderedDict
from typing import Any, Dict

from embrs.utilities.numba_utils import njit_if_enabled, NUMBA_AVAILABLE

# Physical constants
Aks = 2.0e-13
Alb = 0.6
Alpha = 0.25
Ap = 0.000772
Aw = 0.8
Eps = 0.85
Hfs = 0.99
Kelvin = 273.2
Pi = 3.141592654
Pr = 0.7
Sbc = 1.37e-12
Sc = 0.58
Smv = 94.743
St = 72.8
Tcd = 6.0
Tcn = 3.0
Thdiff = 8.0
Wl = 0.0023
Srf = 14.82052
Wsf = 4.60517
Hrd = 0.116171
Hrn = 0.112467
Sir = 0.0714285
Scr = 0.285714


# =============================================================================
# JIT-Compiled Kernels
# =============================================================================

@njit_if_enabled(cache=True)
def _prepare_coefficients(m_nodes, m_w, m_s, m_t, m_d, m_x,
                          m_Twold, m_Tsold, m_Ttold, m_Tv, m_To):
    """Prepare coefficient arrays for finite difference solve.

    This kernel copies current values to "old" arrays and computes
    thermal and moisture diffusivity coefficients.

    Uses vectorized array operations for efficiency.

    Args:
        m_nodes: Number of nodes
        m_w, m_s, m_t, m_d, m_x: Current state arrays
        m_Twold, m_Tsold, m_Ttold, m_Tv, m_To: Output coefficient arrays
    """
    # Vectorized array copies - Numba optimizes these efficiently
    m_Twold[:] = m_w
    m_Tsold[:] = m_s
    m_Ttold[:] = m_t
    # Vectorized coefficient computations
    for i in range(m_nodes):
        m_Tv[i] = Thdiff * m_x[i]
        m_To[i] = m_d[i] * m_x[i]


@njit_if_enabled(cache=True)
def _compute_gravity_drainage(m_nodes, m_w, m_wsa, wdiff, gnu, m_x, m_vf, m_Tg):
    """Compute gravity drainage coefficients.

    Args:
        m_nodes: Number of nodes
        m_w: Moisture content array
        m_wsa: Saturation moisture content
        wdiff: Difference between max and saturation moisture
        gnu: Kinematic viscosity
        m_x: Radial position array
        m_vf: Volume fraction factor
        m_Tg: Output gravity drainage coefficient array
    """
    for i in range(m_nodes):
        m_Tg[i] = 0.0
        svp = (m_w[i] - m_wsa) / wdiff
        if Sir <= svp <= Scr:
            ak = Aks * (2.0 * np.sqrt(svp / Scr) - 1.0)
            m_Tg[i] = (ak / (gnu * wdiff)) * m_x[i] * m_vf * (Scr / svp) ** 1.5


@njit_if_enabled(cache=True)
def _solve_saturation(m_nodes, m_dx, m_mdt, m_x, m_Tg, m_Tsold, m_s):
    """Solve saturation diffusion using vectorized finite difference.

    Uses numpy array slicing for interior nodes (1 to m_nodes-2).

    Args:
        m_nodes: Number of nodes
        m_dx: Internodal distance
        m_mdt: Moisture time step
        m_x: Radial position array
        m_Tg: Gravity drainage coefficient array
        m_Tsold: Old saturation values
        m_s: Saturation array (updated in place)
    """
    # Interior nodes only (indices 1 to m_nodes-2)
    # East coefficients: m_Tg[i+1] / m_dx for i in 1..m_nodes-2
    ae = m_Tg[2:m_nodes] / m_dx
    # West coefficients: m_Tg[i-1] / m_dx for i in 1..m_nodes-2
    aw = m_Tg[0:m_nodes-2] / m_dx
    # Radial coefficients: m_x[i] * m_dx / m_mdt for i in 1..m_nodes-2
    ar = m_x[1:m_nodes-1] * m_dx / m_mdt
    # Diagonal coefficients
    ap = ae + aw + ar

    # Compute new saturation values for interior nodes
    m_s[1:m_nodes-1] = (ae * m_Tsold[2:m_nodes] +
                        aw * m_Tsold[0:m_nodes-2] +
                        ar * m_Tsold[1:m_nodes-1]) / ap

    # Clamp to [0, 1]
    for i in range(1, m_nodes - 1):
        if m_s[i] < 0.0:
            m_s[i] = 0.0
        elif m_s[i] > 1.0:
            m_s[i] = 1.0

    # Boundary condition: center node equals adjacent
    m_s[m_nodes - 1] = m_s[m_nodes - 2]


@njit_if_enabled(cache=True)
def _check_continuous_liquid(m_nodes, m_s):
    """Check if liquid is continuous across all interior nodes.

    Args:
        m_nodes: Number of nodes
        m_s: Saturation array

    Returns:
        True if all interior nodes have saturation >= Sir
    """
    for i in range(1, m_nodes - 1):
        if m_s[i] < Sir:
            return False
    return True


@njit_if_enabled(cache=True)
def _update_moisture_continuous(m_nodes, m_wsa, wdiff, m_wmx, m_s, m_w, random_vals):
    """Update moisture when liquid is continuous (saturation-based).

    Args:
        m_nodes: Number of nodes
        m_wsa: Saturation moisture content
        wdiff: Difference between max and saturation moisture
        m_wmx: Maximum moisture content
        m_s: Saturation array
        m_w: Moisture array (updated in place)
        random_vals: Pre-generated random perturbation values (or None)
    """
    for i in range(1, m_nodes - 1):
        m_w[i] = m_wsa + m_s[i] * wdiff
        # Apply perturbation if random values provided
        if random_vals is not None:
            m_w[i] += random_vals[i]
        # Clamp to valid range
        if m_w[i] < 0.0:
            m_w[i] = 0.0
        elif m_w[i] > m_wmx:
            m_w[i] = m_wmx

    # Boundary condition
    m_w[m_nodes - 1] = m_w[m_nodes - 2]


@njit_if_enabled(cache=True)
def _update_moisture_diffusion(m_nodes, m_dx, m_mdt, m_wmx, m_x, m_To, m_Twold, m_w):
    """Update moisture using vectorized diffusion equation.

    Args:
        m_nodes: Number of nodes
        m_dx: Internodal distance
        m_mdt: Moisture time step
        m_wmx: Maximum moisture content
        m_x: Radial position array
        m_To: Moisture diffusivity coefficient array
        m_Twold: Old moisture values
        m_w: Moisture array (updated in place)
    """
    # Interior nodes (indices 1 to m_nodes-2) - vectorized computation
    ae = m_To[2:m_nodes] / m_dx
    aw = m_To[0:m_nodes-2] / m_dx
    ar = m_x[1:m_nodes-1] * m_dx / m_mdt
    ap = ae + aw + ar

    # Compute new moisture values
    m_w[1:m_nodes-1] = (ae * m_Twold[2:m_nodes] +
                        aw * m_Twold[0:m_nodes-2] +
                        ar * m_Twold[1:m_nodes-1]) / ap

    # Clamp to valid range
    for i in range(1, m_nodes - 1):
        if m_w[i] < 0.0:
            m_w[i] = 0.0
        elif m_w[i] > m_wmx:
            m_w[i] = m_wmx

    # Boundary condition
    m_w[m_nodes - 1] = m_w[m_nodes - 2]


@njit_if_enabled(cache=True)
def _update_temperature(m_nodes, m_dx, m_mdt, m_x, m_Tv, m_Ttold, m_t):
    """Update temperature using vectorized thermal diffusion.

    Args:
        m_nodes: Number of nodes
        m_dx: Internodal distance
        m_mdt: Moisture time step
        m_x: Radial position array
        m_Tv: Thermal diffusivity coefficient array
        m_Ttold: Old temperature values
        m_t: Temperature array (updated in place)
    """
    # Interior nodes (indices 1 to m_nodes-2) - vectorized computation
    ae = m_Tv[2:m_nodes] / m_dx
    aw = m_Tv[0:m_nodes-2] / m_dx
    ar = m_x[1:m_nodes-1] * m_dx / m_mdt
    ap = ae + aw + ar

    # Compute new temperature values
    m_t[1:m_nodes-1] = (ae * m_Ttold[2:m_nodes] +
                        aw * m_Ttold[0:m_nodes-2] +
                        ar * m_Ttold[1:m_nodes-1]) / ap

    # Cap temperature at physical maximum
    for i in range(1, m_nodes - 1):
        if m_t[i] > 71.0:
            m_t[i] = 71.0

    # Boundary condition
    m_t[m_nodes - 1] = m_t[m_nodes - 2]


@njit_if_enabled(cache=True)
def _update_internal_loop(
    num_steps, et, mdt, mdt_2,
    ta0, at, ha0, rh, sv0, sv1, bp0_old, bpr,
    m_w, m_s, m_t, m_d, m_x,
    m_Twold, m_Tsold, m_Ttold, m_Tv, m_To, m_Tg,
    m_nodes, m_dx, m_density,
    m_wmax, m_wmx, m_wfilmk, m_vf, m_hc, m_hwf,
    m_stca, m_stcd, m_stv,
    m_allowRainstorm, m_allowRainfall2, m_amlf, m_capf,
    ra1, m_rdur_init, pptrate,
    rai0, rai1,
    ddt, m_wsa_init, m_hf_init,
    perturbate,
    tstate_arr,
    tt_init, ddtNext_init
):
    """JIT-compiled main time-stepping loop for dead fuel moisture model.

    Encapsulates the entire inner loop of update_internal() in a single
    JIT-compiled function. This eliminates ~115 Python interpreter iterations
    and ~690 JIT kernel call transitions per update_internal() call.

    All state arrays (m_w, m_s, m_t, m_d, etc.) are mutated in-place.
    Scalar state values are returned as a tuple for the caller to write
    back to ``self``.

    Returns:
        Tuple of (m_rdur, wsa, hf, wfilm, state, sem).
    """
    Ap_over_024_over_18 = Ap / 0.24 / 18.0

    tt = tt_init
    ddtNext = ddtNext_init
    m_rdur = m_rdur_init
    wsa = m_wsa_init
    hf = m_hf_init
    wfilm = 0.0
    state = 0
    sem = 0.0

    # Pre-allocate random perturbation array
    random_vals = np.empty(m_nodes)

    for nstep in range(1, num_steps):
        # Weather interpolation
        tfract = tt / et
        ta = ta0 + (at - ta0) * tfract
        ha = ha0 + (rh - ha0) * tfract
        sv = sv0 + (sv1 - sv0) * tfract
        bp = bp0_old + (bpr - bp0_old) * tfract
        fsc = sv / Srf
        tka = ta + Kelvin
        tdw = 5205.0 / ((5205.0 / tka) - math.log(ha))
        tdp = tdw - Kelvin

        if fsc < 0.000001:
            tsk = Tcn + Kelvin
            hr = Hrn
            sr = 0.0
        else:
            tsk = Tcd + Kelvin
            hr = Hrd
            sr = Srf * fsc

        psa = 0.0000239 * math.exp(20.58 - (5205.0 / tka))
        pa = ha * psa
        psd = 0.0000239 * math.exp(20.58 - (5205.0 / tdw))

        if ra1 < 0.0001:
            m_rdur = 0.0
        else:
            m_rdur += mdt

        # Surface node calculations
        hr_plus_hc = hr + m_hc
        tfd = ta + (sr - hr * (ta - tsk + Kelvin)) / hr_plus_hc
        qv = 13550.0 - 10.22 * (tfd + Kelvin)
        qw = 5040.0 * math.exp(-14.0 * m_w[0])
        hw = (m_hwf * Ap_over_024_over_18) * qv
        m_t[0] = tfd - (hw * (tfd - ta) / (hr_plus_hc + hw))
        tkf = m_t[0] + Kelvin
        gnu = 0.00439 + 0.00000177 * (338.76 - tkf) ** 2.1237

        c1 = 0.1617 - 0.001419 * m_t[0]
        c2 = 0.4657 + 0.003578 * m_t[0]
        wsa = c1 * Wsf ** c2
        wdiff = m_wmax - wsa
        if wdiff < 0.000001:
            wdiff = 0.000001
        ps1 = 0.0000239 * math.exp(20.58 - (5205.0 / tkf))
        p1 = pa + Ap * bp * (qv / (qv + qw)) * (tka - tkf)
        if p1 < 0.000001:
            p1 = 0.000001
        hf = p1 / ps1
        if hf > Hfs:
            hf = Hfs
        hf_log = -math.log(1.0 - hf)
        sem = c1 * hf_log ** c2

        # State machine for surface moisture
        state = 0
        wfilm = 0.0
        w_old = m_w[0]
        w_new = w_old
        s_new = m_s[0]

        if ra1 > 0.0:
            if m_allowRainstorm and pptrate >= m_stv:
                state = 8
                wfilm = m_wfilmk
                w_new = m_wmx
            else:
                if m_rdur < 1.0 or not m_allowRainfall2:
                    state = 6
                    w_new = w_old + rai0
                else:
                    state = 7
                    w_new = w_old + rai1
            wfilm = m_wfilmk
            s_new = (w_new - wsa) / wdiff
            m_t[0] = tfd
            hf = Hfs
        else:
            if w_old > wsa:
                p1 = ps1
                hf = Hfs
                aml = m_amlf * (ps1 - psd) / bp
                if m_t[0] <= tdp and p1 > psd:
                    aml = 0.0
                w_new = w_old - aml * mdt_2
                if aml > 0.0:
                    w_new -= (mdt * m_capf / gnu)
                if w_new > m_wmx:
                    w_new = m_wmx
                s_new = (w_new - wsa) / wdiff
                if w_new > w_old:
                    state = 3
                elif w_new == w_old:
                    state = 9
                else:
                    state = 5
            elif m_t[0] <= tdp:
                state = 4
                aml = 0.0 if p1 > psd else m_amlf * (p1 - psd) / bp
                w_new = w_old - aml * mdt_2
                s_new = (w_new - wsa) / wdiff
            else:
                if w_old >= sem:
                    state = 2
                    bi = m_stcd * m_dx / m_d[0]
                else:
                    state = 1
                    bi = m_stca * m_dx / m_d[0]
                w_new = (m_w[1] + bi * sem) / (1.0 + bi)
                s_new = 0.0

        if w_new > m_wmx:
            m_w[0] = m_wmx
        else:
            m_w[0] = w_new
        if s_new > 0.0:
            m_s[0] = s_new
        else:
            m_s[0] = 0.0
        tstate_arr[state] += 1

        # Prepare coefficients
        _prepare_coefficients(
            m_nodes, m_w, m_s, m_t, m_d, m_x,
            m_Twold, m_Tsold, m_Ttold, m_Tv, m_To
        )

        if state != 9:
            # Compute gravity drainage
            _compute_gravity_drainage(
                m_nodes, m_w, wsa, wdiff, gnu,
                m_x, m_vf, m_Tg
            )

            # Solve saturation
            _solve_saturation(
                m_nodes, m_dx, mdt, m_x,
                m_Tg, m_Tsold, m_s
            )

            # Update moisture based on liquid continuity
            continuousLiquid = _check_continuous_liquid(m_nodes, m_s)

            if continuousLiquid:
                if perturbate:
                    # Known O(1e-4) nondeterminism: this np.random.uniform call
                    # lives inside a Numba @njit kernel where threading our
                    # owned Generator through is non-trivial. Per the
                    # seed-determinism plan, this is accepted as a small
                    # nondeterminism source and ALLOWLISTed in
                    # tests/test_no_global_rng.py.
                    for i in range(m_nodes):
                        random_vals[i] = np.random.uniform(-0.0001, 0.0001)
                    _update_moisture_continuous(
                        m_nodes, wsa, wdiff, m_wmx,
                        m_s, m_w, random_vals
                    )
                else:
                    # Inline moisture update without perturbation
                    for i in range(1, m_nodes - 1):
                        m_w[i] = wsa + m_s[i] * wdiff
                        if m_w[i] < 0.0:
                            m_w[i] = 0.0
                        elif m_w[i] > m_wmx:
                            m_w[i] = m_wmx
                    m_w[m_nodes - 1] = m_w[m_nodes - 2]
            else:
                _update_moisture_diffusion(
                    m_nodes, m_dx, mdt, m_wmx,
                    m_x, m_To, m_Twold, m_w
                )

        # Update temperature
        _update_temperature(
            m_nodes, m_dx, mdt,
            m_x, m_Tv, m_Ttold, m_t
        )

        # Periodically recalculate diffusivity
        if (ddtNext - tt) < (0.5 * mdt):
            _compute_diffusivity(
                m_nodes, m_t, m_w, wsa, hf,
                m_density, bp, m_d
            )
            ddtNext += ddt

    return (m_rdur, wsa, hf, wfilm, state, sem)


@njit_if_enabled(cache=True)
def _compute_diffusivity(m_nodes, m_t, m_w, m_wsa, m_hf, m_density, bp, m_d):
    """Compute moisture diffusivity at each node.

    Args:
        m_nodes: Number of nodes
        m_t: Temperature array
        m_w: Moisture array
        m_wsa: Saturation moisture content
        m_hf: Surface humidity
        m_density: Fuel density
        bp: Barometric pressure
        m_d: Diffusivity array (updated in place)
    """
    for i in range(m_nodes):
        tk = m_t[i] + 273.2
        qv = 13550.0 - 10.22 * tk
        cpv = 7.22 + 0.002374 * tk + 2.67e-07 * tk * tk
        dv = 0.22 * 3600.0 * (0.0242 / bp) * ((tk / 273.2) ** 1.75)
        ps1 = 0.0000239 * np.exp(20.58 - (5205.0 / tk))
        c1 = 0.1617 - 0.001419 * m_t[i]
        c2 = 0.4657 + 0.003578 * m_t[i]

        wc = m_w[i] if m_w[i] < m_wsa else m_wsa

        dhdm = 0.0
        if m_w[i] < m_wsa:
            if c2 != 1.0 and m_hf < 1.0 and c1 != 0.0 and c2 != 0.0:
                dhdm = (1.0 - m_hf) * ((-np.log(1.0 - m_hf)) ** (1.0 - c2)) / (c1 * c2)
        else:
            if c2 != 1.0 and Hfs < 1.0 and c1 != 0.0 and c2 != 0.0:
                dhdm = (1.0 - Hfs) * (Wsf ** (1.0 - c2)) / (c1 * c2)

        daw = 1.3 - 0.64 * wc
        svaw = 1.0 / daw
        vfaw = svaw * wc / (0.685 + svaw * wc)
        vfcw = (0.685 + svaw * wc) / ((1.0 / m_density) + svaw * wc)
        rfcw = 1.0 - np.sqrt(1.0 - vfcw)
        fac = 1.0 / (rfcw * vfcw)
        con = 1.0 / (2.0 - vfaw)
        qw = 5040.0 * np.exp(-14.0 * wc)
        e = (qv + qw - cpv * tk) / 1.2
        dvpr = 18.0 * 0.016 * (1.0 - vfcw) * dv * ps1 * dhdm / (m_density * 1.987 * tk)
        m_d[i] = dvpr + 3600.0 * 0.0985 * con * fac * np.exp(-e / (1.987 * tk))


# =============================================================================
# DeadFuelMoisture Class
# =============================================================================

class DeadFuelMoisture:
    """Nelson dead fuel moisture model for cylindrical fuel sticks.

    This class implements a 1D implicit finite difference solver that simulates
    moisture diffusion through fuel sticks based on weather conditions.

    Attributes:
        m_radius: Fuel stick radius (cm)
        m_nodes: Number of radial nodes
        m_w: Moisture content at each node (g/g)
        m_t: Temperature at each node (°C)
        m_s: Saturation at each node (fraction)
    """

    def __init__(self, radius: float, stv: float, wmx: float, wfilmk: float):
        """Initialize the dead fuel moisture model.

        Args:
            radius (float): Fuel stick radius (cm).
            stv (float): Storm threshold (cm/h).
            wmx (float): Maximum fiber saturation (g/g).
            wfilmk (float): Water film constant.
        """
        self.m_semTime = None
        self.initializeParameters(radius, stv, wmx, wfilmk)

    def initializeParameters(self, radius, stv, wmx, wfilmk):
        """Initialize model parameters based on fuel geometry."""
        self.m_radius = radius
        self.m_density = 0.4
        self.m_length = 41.0
        self.m_dSteps = self.deriveDiffusivitySteps(radius)
        self.m_hc = self.derivePlanarHeatTransferRate(radius)
        self.m_nodes = self.deriveStickNodes(radius)
        self.m_rai0 = self.deriveRainfallRunoffFactor(radius)
        self.m_rai1 = 0.5
        self.m_stca = self.deriveAdsorptionRate(radius)
        self.m_stcd = 0.06
        self.m_mSteps = self.deriveMoistureSteps(radius)
        self.m_stv = stv
        self.m_wmx = wmx
        self.m_wfilmk = wfilmk
        self.m_allowRainfall2 = True
        self.m_allowRainstorm = True
        self.m_pertubateColumn = True
        self.m_rampRai0 = True
        self.m_rdur = 0.0
        self.initializeStick()

    def deriveDiffusivitySteps(self, radius: float) -> int:
        """Derive number of diffusivity sub-steps from stick radius."""
        return int(4.777 + 2.496 / radius ** 1.3)

    def derivePlanarHeatTransferRate(self, radius: float) -> float:
        """Derive planar heat transfer rate from stick radius."""
        return 0.2195 + 0.05260 / radius ** 2.5

    def deriveStickNodes(self, radius: float) -> int:
        """Derive number of radial nodes from stick radius (always odd)."""
        nodes = int(10.727 + 0.1746 / radius)
        if nodes % 2 == 0:
            nodes += 1
        return nodes

    def deriveRainfallRunoffFactor(self, radius: float) -> float:
        """Derive rainfall runoff factor from stick radius."""
        return 0.02822 + 0.1056 / radius ** 2.2

    def deriveAdsorptionRate(self, radius: float) -> float:
        """Derive adsorption rate constant from stick radius."""
        return 0.0004509 + 0.006126 / radius ** 2.6

    def deriveMoistureSteps(self, radius: float) -> int:
        """Derive number of moisture sub-steps from stick radius."""
        return int(9.8202 + 26.865 / radius ** 1.4)

    def initializeStick(self):
        """Initialize the fuel stick state arrays.

        Arrays are stored as numpy arrays for compatibility with JIT kernels.
        """
        # Internodal distance (cm)
        self.m_dx = self.m_radius / (self.m_nodes - 1)
        self.m_dx_2 = self.m_dx * 2

        # Maximum possible stick moisture content (g/g)
        self.m_wmax = (1.0 / self.m_density) - (1.0 / 1.53)

        # Initialize arrays as numpy arrays for JIT compatibility
        # Temperature array - initialized to 20°C (ambient)
        self.m_t = np.full(self.m_nodes, 20.0, dtype=np.float64)

        # Saturation array - initialized to 0
        self.m_s = np.zeros(self.m_nodes, dtype=np.float64)

        # Diffusivity array - initialized to 0
        self.m_d = np.zeros(self.m_nodes, dtype=np.float64)

        # Moisture array - initialized to half the maximum
        self.m_w = np.full(self.m_nodes, 0.5 * self.m_wmx, dtype=np.float64)

        # Nodal radial distances
        self.m_x = np.zeros(self.m_nodes, dtype=np.float64)
        for i in range(self.m_nodes - 1):
            self.m_x[i] = self.m_radius - (self.m_dx * i)
        self.m_x[self.m_nodes - 1] = 0.0

        # Nodal volume fractions
        self.m_v = np.zeros(self.m_nodes, dtype=np.float64)
        ro = self.m_radius
        ri = ro - 0.5 * self.m_dx
        a2 = self.m_radius * self.m_radius
        self.m_v[0] = (ro * ro - ri * ri) / a2
        vwt = self.m_v[0]
        for i in range(1, self.m_nodes - 1):
            ro = ri
            ri = ro - self.m_dx
            self.m_v[i] = (ro * ro - ri * ri) / a2
            vwt += self.m_v[i]
        self.m_v[self.m_nodes - 1] = ri * ri / a2
        vwt += self.m_v[self.m_nodes - 1]

        # Temporary arrays for time stepping (numpy arrays for JIT)
        self.m_Twold = np.zeros(self.m_nodes, dtype=np.float64)
        self.m_Ttold = np.zeros(self.m_nodes, dtype=np.float64)
        self.m_Tsold = np.zeros(self.m_nodes, dtype=np.float64)
        self.m_Tv = np.zeros(self.m_nodes, dtype=np.float64)
        self.m_To = np.zeros(self.m_nodes, dtype=np.float64)
        self.m_Tg = np.zeros(self.m_nodes, dtype=np.float64)

        # Initialize the environment
        self.initializeEnvironment(
            20.0,  # Ambient air temperature (oC)
            0.20,  # Ambient air relative humidity (g/g)
            0.0,   # Solar radiation (W/m2)
            0.0,   # Cumulative rainfall (cm)
            20.0,  # Initial stick temperature (oC)
            0.20,  # Initial stick surface humidity (g/g)
            0.5 * self.m_wmx,  # Initial stick moisture content
            0.0218)  # Initial stick barometric pressure (cal/cm3)
        self.m_init = False

        # Computation optimization parameters
        self.m_hwf = 0.622 * self.m_hc * (Pr / Sc) ** 0.667
        self.m_amlf = self.m_hwf / (0.24 * self.m_density * self.m_radius)
        rcav = 0.5 * Aw * Wl
        self.m_capf = 3600.0 * Pi * St * rcav * rcav / (16.0 * self.m_radius * self.m_radius * self.m_length * self.m_density)
        self.m_vf = St / (self.m_density * Wl * Scr)

    def diffusivity(self, bp):
        """Compute diffusivity using JIT kernel if available."""
        _compute_diffusivity(
            self.m_nodes, self.m_t, self.m_w, self.m_wsa,
            self.m_hf, self.m_density, bp, self.m_d
        )

    # Class-level templates for fast cloning (populated on first use)
    _template_1hr = None
    _template_10hr = None
    _template_100hr = None
    _template_1000hr = None

    # Mutable array attribute names that must be deep-copied when cloning
    _MUTABLE_ARRAYS = (
        'm_t', 'm_s', 'm_d', 'm_w', 'm_x', 'm_v',
        'm_Twold', 'm_Ttold', 'm_Tsold', 'm_Tv', 'm_To', 'm_Tg',
    )

    @classmethod
    def _clone_from_template(cls, template):
        """Create a new instance by cloning a template's state.

        Copies all scalar attributes via dict update (fast) and deep-copies
        only the mutable numpy arrays. Much faster than full __init__ because
        it skips parameter derivation and array initialization math.
        """
        new = object.__new__(cls)
        new.__dict__.update(template.__dict__)
        for attr in cls._MUTABLE_ARRAYS:
            setattr(new, attr, getattr(template, attr).copy())
        return new

    @staticmethod
    def createDeadFuelMoisture1():
        """Create 1-hour fuel moisture model (fine fuels)."""
        if DeadFuelMoisture._template_1hr is None:
            DeadFuelMoisture._template_1hr = DeadFuelMoisture(0.20, 0.006, 0.85, 0.10)
        return DeadFuelMoisture._clone_from_template(DeadFuelMoisture._template_1hr)

    @staticmethod
    def createDeadFuelMoisture10():
        """Create 10-hour fuel moisture model."""
        if DeadFuelMoisture._template_10hr is None:
            DeadFuelMoisture._template_10hr = DeadFuelMoisture(0.64, 0.05, 0.60, 0.05)
        return DeadFuelMoisture._clone_from_template(DeadFuelMoisture._template_10hr)

    @staticmethod
    def createDeadFuelMoisture100():
        """Create 100-hour fuel moisture model."""
        if DeadFuelMoisture._template_100hr is None:
            DeadFuelMoisture._template_100hr = DeadFuelMoisture(2.00, 5.0, 0.40, 0.005)
        return DeadFuelMoisture._clone_from_template(DeadFuelMoisture._template_100hr)

    @staticmethod
    def createDeadFuelMoisture1000():
        """Create 1000-hour fuel moisture model (large logs)."""
        if DeadFuelMoisture._template_1000hr is None:
            DeadFuelMoisture._template_1000hr = DeadFuelMoisture(6.40, 7.5, 0.32, 0.003)
        return DeadFuelMoisture._clone_from_template(DeadFuelMoisture._template_1000hr)

    def initialized(self):
        """Check if environment has been initialized."""
        return self.m_init

    def initializeEnvironment(self, ta: float, ha: float, sr: float,
                              rc: float, ti: float, hi: float,
                              wi: float, bp: float):
        """Initialize environmental conditions.

        Args:
            ta (float): Ambient air temperature (°C).
            ha (float): Ambient air relative humidity (g/g).
            sr (float): Solar radiation (W/m²).
            rc (float): Cumulative rainfall (cm).
            ti (float): Initial stick temperature (°C).
            hi (float): Initial stick surface humidity (g/g).
            wi (float): Initial stick moisture content (g/g).
            bp (float): Barometric pressure (cal/cm³).
        """
        self.m_ta0 = self.m_ta1 = ta
        self.m_ha0 = self.m_ha1 = ha
        self.m_sv0 = self.m_sv1 = sr / Smv
        self.m_rc0 = self.m_rc1 = rc
        self.m_ra0 = self.m_ra1 = 0.0
        self.m_bp0 = self.m_bp1 = bp

        self.m_hf = hi
        self.m_wfilm = 0.0
        self.m_wsa = wi + 0.1
        self.m_t[:] = ti
        self.m_w[:] = wi
        self.m_s[:] = 0.0

        self.diffusivity(self.m_bp0)
        self.m_init = True

    def meanMoisture(self):
        """Calculate mean moisture using Simpson's rule integration."""
        wea, web = 0.0, 0.0
        wec = self.m_w[0]
        wei = self.m_dx / (3.0 * self.m_radius)
        for i in range(1, self.m_nodes - 1, 2):
            wea = 4.0 * self.m_w[i]
            web = 2.0 * self.m_w[i + 1]
            if (i + 1) == (self.m_nodes - 1):
                web = self.m_w[self.m_nodes - 1]
            wec += web + wea
        wbr = wei * wec
        wbr = min(wbr, self.m_wmx)
        wbr += self.m_wfilm
        return wbr

    def meanWtdMoisture(self):
        """Calculate volume-weighted mean moisture."""
        wbr = np.dot(self.m_w, self.m_v)
        wbr = min(wbr, self.m_wmx)
        wbr += self.m_wfilm
        return wbr

    def meanWtdTemperature(self):
        """Calculate volume-weighted mean temperature."""
        return np.dot(self.m_t, self.m_v)

    def pptRate(self):
        """Get precipitation rate."""
        return (self.m_ra1 / self.m_et) if self.m_et > 0.00 else 0.00

    def setAdsorptionRate(self, adsorptionRate):
        self.m_stca = adsorptionRate

    def setAllowRainfall2(self, allow=True):
        self.m_allowRainfall2 = allow

    def setAllowRainstorm(self, allow=True):
        self.m_allowRainstorm = allow

    def setDesorptionRate(self, desorptionRate):
        self.m_stcd = desorptionRate

    def setDiffusivitySteps(self, diffusivitySteps):
        self.m_dSteps = diffusivitySteps

    def setMaximumLocalMoisture(self, localMaxMc):
        self.m_wmx = localMaxMc

    def setMoistureSteps(self, moistureSteps):
        self.m_mSteps = moistureSteps

    def setPlanarHeatTransferRate(self, planarHeatTransferRate):
        self.m_hc = planarHeatTransferRate

    def setPertubateColumn(self, pertubate=True):
        self.m_pertubateColumn = pertubate

    def setRainfallRunoffFactor(self, rainfallRunoffFactor):
        self.m_rai0 = rainfallRunoffFactor

    def setRampRai0(self, ramp=True):
        self.m_rampRai0 = ramp

    def setStickDensity(self, stickDensity=0.40):
        self.m_density = stickDensity

    def setStickLength(self, stickLength=41.0):
        self.m_length = stickLength

    def setStickNodes(self, stickNodes=11):
        self.m_nodes = stickNodes

    def setWaterFilmContribution(self, waterFilm):
        self.m_wfilm = waterFilm

    def stateName(self):
        """Get the name of the current moisture state."""
        states = [
            "None",             # 0
            "Adsorption",       # 1
            "Desorption",       # 2
            "Condensation1",    # 3
            "Condensation2",    # 4
            "Evaporation",      # 5
            "Rainfall1",        # 6
            "Rainfall2",        # 7
            "Rainstorm",        # 8
            "Stagnation",       # 9
            "Error"             # 10
        ]
        return states[self.m_state]

    def state_dict(self) -> Dict[str, Any]:
        """Return the evolving state of this DFM instance for hashing.

        Used by determinism regression tests (see ``hash_fire_grid`` in
        ``applications/firefighting/tests/_determinism_helpers.py``).

        **Dtype contract**: every numpy array in the returned dict is
        ``float64``. The kernel initializes its node arrays
        (``m_t``, ``m_s``, ``m_d``, ``m_w``) with explicit
        ``dtype=np.float64`` (see :py:meth:`initializeStick`). If a future
        change introduces a different dtype, this method coerces to
        ``float64`` defensively so the hash output stays stable.

        **What is in scope** (Group A — kernel-mutated and time-evolving):
            - Stick arrays: ``m_t``, ``m_s``, ``m_d``, ``m_w``.
            - Surface state: ``m_hf``, ``m_wfilm``, ``m_wsa``.
            - Environmental window (previous/current pairs): ``m_ta0/1``,
              ``m_ha0/1``, ``m_sv0/1``, ``m_rc0/1``, ``m_ra0/1``,
              ``m_bp0/1``.
            - Time accounting: ``m_et``, ``m_semTime``, ``m_rdur``.
            - State machine: ``m_state``, ``m_sem``, ``m_pptrate``.
            - Time-step caches written mid-loop: ``m_mdt``, ``m_mdt_2``,
              ``m_ddt``, ``m_sf``.
            - Init flag: ``m_init``.

        **What is NOT in scope** (handled by other tests; see
        ``INTENTIONALLY_NOT_HASHED`` in the test helper module):
            - Mutable configuration (``m_radius``, ``m_wmx``, ``m_density``,
              setter-targets, behavioral flags). Rationale: defensive hashing
              of config creates false confidence — a future setter wouldn't
              automatically be covered. Config equality is asserted
              separately by the test harness before the grid hash compare.
            - Init-derived caches (``m_dx``, ``m_x``, ``m_v``, ``m_hwf``,
              etc.) — deterministic functions of mutable config.
            - Scratch buffers (``m_Twold``, ``m_Ttold``, ``m_Tsold``,
              ``m_Tv``, ``m_To``, ``m_Tg``) — overwritten before being
              read across step boundaries (see
              ``_compute_environment_change_coeffs`` lines 69–75).

        Returns:
            OrderedDict mapping field name to value. Scalar floats are
            kept native (caller rounds per ``FLOAT_HASH_DECIMALS``). Arrays
            are coerced to ``np.ndarray`` of dtype ``float64`` (caller
            calls ``.tobytes()`` after rounding).
        """
        def _f64(arr):
            a = np.asarray(arr)
            return a if a.dtype == np.float64 else a.astype(np.float64)

        return OrderedDict([
            # Stick arrays (load-bearing per-node state)
            ("m_t", _f64(self.m_t)),
            ("m_s", _f64(self.m_s)),
            ("m_d", _f64(self.m_d)),
            ("m_w", _f64(self.m_w)),
            # Surface / film state
            ("m_hf", float(self.m_hf)),
            ("m_wfilm", float(self.m_wfilm)),
            ("m_wsa", float(self.m_wsa)),
            # Environmental window
            ("m_ta0", float(self.m_ta0)),
            ("m_ta1", float(self.m_ta1)),
            ("m_ha0", float(self.m_ha0)),
            ("m_ha1", float(self.m_ha1)),
            ("m_sv0", float(self.m_sv0)),
            ("m_sv1", float(self.m_sv1)),
            ("m_rc0", float(self.m_rc0)),
            ("m_rc1", float(self.m_rc1)),
            ("m_ra0", float(self.m_ra0)),
            ("m_ra1", float(self.m_ra1)),
            ("m_bp0", float(self.m_bp0)),
            ("m_bp1", float(self.m_bp1)),
            # Time accounting
            ("m_et", float(getattr(self, "m_et", 0.0))),
            ("m_semTime", self.m_semTime),  # may be None or a datetime
            ("m_rdur", float(self.m_rdur)),
            # State machine + dependent caches
            ("m_state", int(getattr(self, "m_state", 0))),
            ("m_sem", float(getattr(self, "m_sem", 0.0))),
            ("m_pptrate", float(getattr(self, "m_pptrate", 0.0))),
            # Time-step caches written mid-loop
            ("m_mdt", float(getattr(self, "m_mdt", 0.0))),
            ("m_mdt_2", float(getattr(self, "m_mdt_2", 0.0))),
            ("m_ddt", float(getattr(self, "m_ddt", 0.0))),
            ("m_sf", float(getattr(self, "m_sf", 0.0))),
            # Init flag
            ("m_init", bool(self.m_init)),
        ])

    def update(self, year: int, month: int, day: int, hour: int,
               minute: int, second: int, at: float, rh: float,
               sW: float, rcum: float, bpr: float) -> bool:
        """Update moisture based on datetime and weather.

        Args:
            year (int): Year.
            month (int): Month (1-12).
            day (int): Day of month.
            hour (int): Hour (0-23).
            minute (int): Minute (0-59).
            second (int): Second (0-59).
            at (float): Air temperature (°C).
            rh (float): Relative humidity (fraction).
            sW (float): Solar radiation (W/m²).
            rcum (float): Cumulative rainfall (cm).
            bpr (float): Barometric pressure (cal/cm³).

        Returns:
            bool: True if update succeeded, False otherwise.
        """
        # Determine Julian date for this new observation
        jd0 = self.m_semTime.toordinal() + 1721424.5
        self.m_semTime = datetime.datetime(year, month, day, hour, minute, second)
        jd1 = self.m_semTime.toordinal() + 1721424.5

        # Determine elapsed time (h) between the current and previous dates
        et = 24.0 * (jd1 - jd0)

        # If the Julian date wasn't initialized, or if the new time is less than the old time, assume a 1-h elapsed time.
        if jd1 < jd0:
            et = 1.0

        # Update!
        return self.update_internal(et, at, rh, sW, rcum, bpr)

    def update_internal(self, et: float, at: float, rh: float,
                        sW: float, rcum: float, bpr: float) -> bool:
        """Update moisture state for elapsed time.

        This is the main computational method. Inner loops are JIT-compiled
        when Numba is available. The outer loop uses local variable caching
        and math.exp/math.log for scalar operations to minimize overhead.

        Args:
            et (float): Elapsed time (hours).
            at (float): Air temperature (°C).
            rh (float): Relative humidity (fraction).
            sW (float): Solar radiation (W/m²).
            rcum (float): Cumulative rainfall (cm).
            bpr (float): Barometric pressure (cal/cm³).

        Returns:
            bool: True if update succeeded, False otherwise.
        """
        # Validate inputs
        if et < 0.0000027:
            print(f"DeadFuelMoisture::update() has a regressive elapsed time of {et} hours.")
            return False

        if rcum < self.m_rc1:
            print(f"DeadFuelMoisture::update() has a regressive cumulative rainfall amount of {rcum} cm.")
            self.m_rc1 = rcum
            self.m_ra0 = 0.0
            return False

        if rh < 0.001 or rh > 1.0:
            print(f"DeadFuelMoisture::update() has an out-of-range relative humidity of {rh} g/g.")
            return False

        if at < -60.0 or at > 60.0:
            print(f"DeadFuelMoisture::update() has an out-of-range air temperature of {at} oC.")
            return False

        if sW < 0.0:
            sW = 0.0
        if sW > 2000.0:
            print(f"DeadFuelMoisture::update() has an out-of-range solar insolation of {sW} W/m2.")
            return False

        # Save previous weather values
        ta0 = self.m_ta1
        ha0 = self.m_ha1
        sv0 = self.m_sv1
        rc0 = self.m_rc1
        ra0 = self.m_ra1
        bp0_old = self.m_bp1
        self.m_ta0 = ta0
        self.m_ha0 = ha0
        self.m_sv0 = sv0
        self.m_rc0 = rc0
        self.m_ra0 = ra0
        self.m_bp0 = bp0_old

        # Save current weather values
        sv1 = sW / Smv
        self.m_ta1 = at
        self.m_ha1 = rh
        self.m_sv1 = sv1
        self.m_rc1 = rcum
        self.m_bp1 = bpr
        self.m_et = et

        # Precipitation calculations
        ra1 = rcum - rc0
        self.m_ra1 = ra1
        m_rdur = 0.0 if ra1 < 0.0001 else self.m_rdur
        self.m_rdur = m_rdur
        pptrate = ra1 / et / Pi
        self.m_pptrate = pptrate
        m_mSteps = self.m_mSteps
        mdt = et / m_mSteps
        self.m_mdt = mdt
        mdt_2 = mdt * 2.0
        self.m_mdt_2 = mdt_2
        m_dx = self.m_dx
        self.m_sf = 3600.0 * mdt / (self.m_dx_2 * self.m_density)
        m_dSteps = self.m_dSteps
        ddt = et / m_dSteps
        self.m_ddt = ddt

        rai0 = mdt * self.m_rai0 * (1.0 - math.exp(-100.0 * pptrate))
        if rh < ha0:
            if self.m_rampRai0:
                rai0 *= (1.0 - ((ha0 - rh) / ha0))
            else:
                rai0 *= 0.15
        rai1 = mdt * self.m_rai1 * pptrate

        # Time-stepping control
        ddtNext = ddt
        tt = mdt

        # Perturbation flag
        perturbate = self.m_pertubateColumn

        # Cache instance attributes used in JIT call
        m_nodes = self.m_nodes
        m_w = self.m_w
        m_s = self.m_s
        m_t = self.m_t
        m_d = self.m_d
        m_x = self.m_x
        m_Twold = self.m_Twold
        m_Tsold = self.m_Tsold
        m_Ttold = self.m_Ttold
        m_Tv = self.m_Tv
        m_To = self.m_To
        m_Tg = self.m_Tg
        m_wmax = self.m_wmax
        m_wmx = self.m_wmx
        m_wfilmk = self.m_wfilmk
        m_dx = self.m_dx

        # Main time-stepping loop (JIT-compiled)
        num_steps = int(et / mdt) + 1
        tstate_arr = np.zeros(11, dtype=np.int64)

        result = _update_internal_loop(
            num_steps, et, mdt, mdt_2,
            ta0, at, ha0, rh, sv0, sv1, bp0_old, bpr,
            m_w, m_s, m_t, m_d, m_x,
            m_Twold, m_Tsold, m_Ttold, m_Tv, m_To, m_Tg,
            m_nodes, m_dx, self.m_density,
            m_wmax, m_wmx, m_wfilmk, self.m_vf, self.m_hc, self.m_hwf,
            self.m_stca, self.m_stcd, self.m_stv,
            self.m_allowRainstorm, self.m_allowRainfall2, self.m_amlf, self.m_capf,
            ra1, self.m_rdur, pptrate,
            rai0, rai1,
            ddt, self.m_wsa, self.m_hf,
            perturbate,
            tstate_arr,
            tt, ddtNext
        )

        # Write back scalar results
        self.m_rdur = result[0]
        self.m_wsa = result[1]
        self.m_hf = result[2]
        self.m_wfilm = result[3]
        self.m_state = int(result[4])
        self.m_sem = result[5]

        # Set final state to most common state
        most_common = 0
        max_count = tstate_arr[0]
        for i in range(1, 11):
            if tstate_arr[i] > max_count:
                max_count = tstate_arr[i]
                most_common = i
        self.m_state = most_common
        return True

    def zero(self):
        """Reset all state to zero."""
        self.m_semTime = None
        self.m_density = 0.0
        self.m_dSteps = 0
        self.m_hc = 0.0
        self.m_length = 0.0
        self.m_nodes = 0
        self.m_radius = 0.0
        self.m_rai0 = 0.0
        self.m_rai1 = 0.0
        self.m_stca = 0.0
        self.m_stcd = 0.0
        self.m_mSteps = 0
        self.m_stv = 0.0
        self.m_wfilmk = 0.0
        self.m_wmx = 0.0
        self.m_dx = 0.0
        self.m_wmax = 0.0
        self.m_x = np.array([])
        self.m_v = np.array([])
        self.m_amlf = 0.0
        self.m_capf = 0.0
        self.m_hwf = 0.0
        self.m_dx_2 = 0.0
        self.m_vf = 0.0
        self.m_bp0 = 0.0
        self.m_ha0 = 0.0
        self.m_rc0 = 0.0
        self.m_sv0 = 0.0
        self.m_ta0 = 0.0
        self.m_init = False
        self.m_bp1 = 0.0
        self.m_et = 0.0
        self.m_ha1 = 0.0
        self.m_rc1 = 0.0
        self.m_sv1 = 0.0
        self.m_ta1 = 0.0
        self.m_ddt = 0.0
        self.m_mdt = 0.0
        self.m_mdt_2 = 0.0
        self.m_pptrate = 0.0
        self.m_ra0 = 0.0
        self.m_ra1 = 0.0
        self.m_rdur = 0.0
        self.m_sf = 0.0
        self.m_hf = 0.0
        self.m_wsa = 0.0
        self.m_sem = 0.0
        self.m_wfilm = 0.0
        self.m_t = np.array([])
        self.m_s = np.array([])
        self.m_d = np.array([])
        self.m_w = np.array([])
        self.m_state = 0

    def __str__(self):
        output = []
        output.append(f"m_semTime {self.m_semTime}")
        output.append(f"m_density {self.m_density}")
        output.append(f"m_dSteps {self.m_dSteps}")
        output.append(f"m_hc {self.m_hc}")
        output.append(f"m_length {self.m_length}")
        output.append(f"m_nodes {self.m_nodes}")
        output.append(f"m_radius {self.m_radius}")
        output.append(f"m_rai0 {self.m_rai0}")
        output.append(f"m_rai1 {self.m_rai1}")
        output.append(f"m_stca {self.m_stca}")
        output.append(f"m_stcd {self.m_stcd}")
        output.append(f"m_mSteps {self.m_mSteps}")
        output.append(f"m_stv {self.m_stv}")
        output.append(f"m_wfilmk {self.m_wfilmk}")
        output.append(f"m_wmx {self.m_wmx}")
        output.append(f"m_dx {self.m_dx}")
        output.append(f"m_wmax {self.m_wmax}")
        output.append(f"m_x ({len(self.m_x)}) {' '.join(map(str, self.m_x))}")
        output.append(f"m_v ({len(self.m_v)}) {' '.join(map(str, self.m_v))}")
        output.append(f"m_amlf {self.m_amlf}")
        output.append(f"m_capf {self.m_capf}")
        output.append(f"m_hwf {self.m_hwf}")
        output.append(f"m_dx_2 {self.m_dx_2}")
        output.append(f"m_vf {self.m_vf}")
        output.append(f"m_bp0 {self.m_bp0}")
        output.append(f"m_ha0 {self.m_ha0}")
        output.append(f"m_rc0 {self.m_rc0}")
        output.append(f"m_sv0 {self.m_sv0}")
        output.append(f"m_ta0 {self.m_ta0}")
        output.append(f"m_init {self.m_init}")
        output.append(f"m_bp1 {self.m_bp1}")
        output.append(f"m_et {self.m_et}")
        output.append(f"m_ha1 {self.m_ha1}")
        output.append(f"m_rc1 {self.m_rc1}")
        output.append(f"m_sv1 {self.m_sv1}")
        output.append(f"m_ta1 {self.m_ta1}")
        output.append(f"m_ddt {self.m_ddt}")
        output.append(f"m_mdt {self.m_mdt}")
        output.append(f"m_mdt_2 {self.m_mdt_2}")
        output.append(f"m_pptrate {self.m_pptrate}")
        output.append(f"m_ra0 {self.m_ra0}")
        output.append(f"m_ra1 {self.m_ra1}")
        output.append(f"m_rdur {self.m_rdur}")
        output.append(f"m_sf {self.m_sf}")
        output.append(f"m_hf {self.m_hf}")
        output.append(f"m_wsa {self.m_wsa}")
        output.append(f"m_sem {self.m_sem}")
        output.append(f"m_wfilm {self.m_wfilm}")
        output.append(f"m_t {len(self.m_t)} {' '.join(map(str, self.m_t))}")
        output.append(f"m_s {len(self.m_s)} {' '.join(map(str, self.m_s))}")
        output.append(f"m_d {len(self.m_d)} {' '.join(map(str, self.m_d))}")
        output.append(f"m_w {len(self.m_w)} {' '.join(map(str, self.m_w))}")
        output.append(f"m_state {self.m_state}")
        return '\n'.join(output)

    @staticmethod
    def from_string(input_str: str) -> 'DeadFuelMoisture':
        """Deserialize a DeadFuelMoisture instance from its string representation.

        Args:
            input_str (str): String produced by ``__str__``.

        Returns:
            DeadFuelMoisture: Reconstructed model instance.
        """
        lines = input_str.strip().split('\n')
        r = DeadFuelMoisture(0, "")
        r.m_semTime = datetime.datetime.strptime(lines[0].split()[1], "%Y/%m/%d %H:%M:%S.%f")
        r.m_density = float(lines[1].split()[1])
        r.m_dSteps = int(lines[2].split()[1])
        r.m_hc = float(lines[3].split()[1])
        r.m_length = float(lines[4].split()[1])
        r.m_name = lines[5].split()[1]
        r.m_nodes = int(lines[6].split()[1])
        r.m_radius = float(lines[7].split()[1])
        r.m_rai0 = float(lines[8].split()[1])
        r.m_rai1 = float(lines[9].split()[1])
        r.m_stca = float(lines[10].split()[1])
        r.m_stcd = float(lines[11].split()[1])
        r.m_mSteps = int(lines[12].split()[1])
        r.m_stv = float(lines[13].split()[1])
        r.m_wfilmk = float(lines[14].split()[1])
        r.m_wmx = float(lines[15].split()[1])
        r.m_dx = float(lines[16].split()[1])
        r.m_wmax = float(lines[17].split()[1])
        n = int(lines[18].split()[1])
        r.m_x = np.array([float(lines[19 + i]) for i in range(n)])
        n = int(lines[19 + n].split()[1])
        r.m_v = np.array([float(lines[20 + i]) for i in range(n)])
        r.m_amlf = float(lines[20 + n].split()[1])
        r.m_capf = float(lines[21 + n].split()[1])
        r.m_hwf = float(lines[22 + n].split()[1])
        r.m_dx_2 = float(lines[23 + n].split()[1])
        r.m_vf = float(lines[24 + n].split()[1])
        r.m_bp0 = float(lines[25 + n].split()[1])
        r.m_ha0 = float(lines[26 + n].split()[1])
        r.m_rc0 = float(lines[27 + n].split()[1])
        r.m_sv0 = float(lines[28 + n].split()[1])
        r.m_ta0 = float(lines[29 + n].split()[1])
        r.m_init = lines[30 + n].split()[1] == 'True'
        r.m_bp1 = float(lines[31 + n].split()[1])
        r.m_et = float(lines[32 + n].split()[1])
        r.m_ha1 = float(lines[33 + n].split()[1])
        r.m_rc1 = float(lines[34 + n].split()[1])
        r.m_sv1 = float(lines[35 + n].split()[1])
        r.m_ta1 = float(lines[36 + n].split()[1])
        r.m_ddt = float(lines[37 + n].split()[1])
        r.m_mdt = float(lines[38 + n].split()[1])
        r.m_mdt_2 = float(lines[39 + n].split()[1])
        r.m_pptrate = float(lines[40 + n].split()[1])
        r.m_ra0 = float(lines[41 + n].split()[1])
        r.m_ra1 = float(lines[42 + n].split()[1])
        r.m_rdur = float(lines[43 + n].split()[1])
        r.m_sf = float(lines[44 + n].split()[1])
        r.m_hf = float(lines[45 + n].split()[1])
        r.m_wsa = float(lines[46 + n].split()[1])
        r.m_sem = float(lines[47 + n].split()[1])
        r.m_wfilm = float(lines[48 + n].split()[1])
        n = int(lines[50 + n].split()[1])
        r.m_t = np.array([float(lines[51 + i + n]) for i in range(n)])
        n = int(lines[51 + n + n].split()[1])
        r.m_s = np.array([float(lines[52 + i + n]) for i in range(n)])
        n = int(lines[52 + n + n].split()[1])
        r.m_d = np.array([float(lines[53 + i + n]) for i in range(n)])
        n = int(lines[53 + n + n].split()[1])
        r.m_w = np.array([float(lines[54 + i + n]) for i in range(n)])
        r.m_state = int(lines[55 + n + n].split()[1])
        return r
