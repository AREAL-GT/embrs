"""Numba JIT compilation utilities for EMBRS.

This module provides configuration and helper utilities for Numba JIT
compilation used in performance-critical sections of the EMBRS codebase.

Environment Variables:
    EMBRS_DISABLE_JIT: Set to '1' to disable JIT compilation globally.
                       Useful for debugging or testing without JIT overhead.
    NUMBA_DISABLE_JIT: Numba's built-in flag, also respected.

Usage:
    # In modules that use JIT-compiled functions:
    from embrs.utilities.numba_utils import jit_if_enabled, NUMBA_AVAILABLE

    @jit_if_enabled(nopython=True, cache=True)
    def my_hot_function(x, y):
        # Numerical computation
        return x + y

    # Check if JIT is available:
    if NUMBA_AVAILABLE:
        # Use JIT-optimized path
        pass
    else:
        # Fallback to pure Python
        pass
"""

import os
import functools
from typing import Callable, Any

# Check if JIT should be disabled via environment variable
DISABLE_JIT = os.environ.get('EMBRS_DISABLE_JIT', '0') == '1'

# Try to import numba
try:
    import numba
    from numba import jit, njit, prange
    from numba import config as numba_config

    # Respect EMBRS_DISABLE_JIT
    if DISABLE_JIT:
        numba_config.DISABLE_JIT = True

    NUMBA_AVAILABLE = True
    NUMBA_VERSION = numba.__version__

except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_VERSION = None
    jit = None
    njit = None
    prange = range  # Fallback to regular range


def jit_if_enabled(**jit_kwargs: Any) -> Callable:
    """Decorator that applies Numba JIT if available and enabled.

    This decorator wraps functions with Numba's @jit decorator when Numba
    is available and JIT is not disabled. If Numba is unavailable or JIT
    is disabled, the function is returned unchanged.

    Args:
        **jit_kwargs: Keyword arguments to pass to numba.jit.
                      Common options:
                      - nopython=True: Compile in nopython mode (faster)
                      - cache=True: Cache compiled functions to disk
                      - parallel=True: Enable automatic parallelization
                      - fastmath=True: Use fast math optimizations

    Returns:
        Callable: Decorated function (JIT-compiled if enabled, else unchanged).

    Example:
        @jit_if_enabled(nopython=True, cache=True)
        def compute_moisture(w, t, s, params):
            # Numerical computation
            result = 0.0
            for i in range(len(w)):
                result += w[i] * t[i]
            return result
    """
    def decorator(func: Callable) -> Callable:
        if NUMBA_AVAILABLE and not DISABLE_JIT:
            return jit(**jit_kwargs)(func)
        else:
            return func
    return decorator


def njit_if_enabled(**jit_kwargs: Any) -> Callable:
    """Decorator that applies Numba njit if available and enabled.

    Equivalent to jit_if_enabled(nopython=True, **jit_kwargs).
    Use this for functions that must run in nopython mode.

    Args:
        **jit_kwargs: Keyword arguments to pass to numba.njit.

    Returns:
        Callable: Decorated function (JIT-compiled if enabled, else unchanged).
    """
    def decorator(func: Callable) -> Callable:
        if NUMBA_AVAILABLE and not DISABLE_JIT:
            return njit(**jit_kwargs)(func)
        else:
            return func
    return decorator


def get_numba_status() -> dict:
    """Get information about Numba availability and configuration.

    Returns:
        dict: Dictionary containing:
            - available: bool, whether Numba is installed
            - version: str or None, Numba version if available
            - jit_enabled: bool, whether JIT compilation is enabled
            - disable_jit_env: bool, whether EMBRS_DISABLE_JIT is set
    """
    return {
        'available': NUMBA_AVAILABLE,
        'version': NUMBA_VERSION,
        'jit_enabled': NUMBA_AVAILABLE and not DISABLE_JIT,
        'disable_jit_env': DISABLE_JIT,
    }


def warmup_jit_cache() -> None:
    """Warm up JIT-compiled functions to avoid first-call latency.

    This function can be called during application startup to pre-compile
    JIT-decorated functions. This moves the compilation overhead from the
    first simulation step to the initialization phase.

    Call this after all JIT-decorated functions are defined.
    """
    if not NUMBA_AVAILABLE or DISABLE_JIT:
        return

    # Import modules that contain JIT-compiled functions
    # This triggers compilation if cache=True is not set
    try:
        from embrs.models import dead_fuel_moisture
        # Add other modules with JIT functions here as they are implemented
    except ImportError:
        pass


# Parallel range helper
def get_prange() -> type:
    """Get the appropriate parallel range function.

    Returns:
        type: numba.prange if available and JIT enabled, else range.
    """
    if NUMBA_AVAILABLE and not DISABLE_JIT:
        return prange
    return range
