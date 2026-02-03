"""Tests for Numba utilities module.

These tests verify that the Numba JIT utilities work correctly both
when Numba is available and when it needs to fall back gracefully.
"""

import pytest
import numpy as np
import os


class TestNumbaUtilsImport:
    """Tests for basic numba_utils import and availability."""

    def test_module_imports(self):
        """Module should import without error."""
        from embrs.utilities import numba_utils
        assert numba_utils is not None

    def test_numba_available_flag(self):
        """NUMBA_AVAILABLE flag should be set correctly."""
        from embrs.utilities.numba_utils import NUMBA_AVAILABLE
        assert isinstance(NUMBA_AVAILABLE, bool)

    def test_numba_version_if_available(self):
        """NUMBA_VERSION should be set if Numba is available."""
        from embrs.utilities.numba_utils import NUMBA_AVAILABLE, NUMBA_VERSION
        if NUMBA_AVAILABLE:
            assert NUMBA_VERSION is not None
            assert isinstance(NUMBA_VERSION, str)
        else:
            assert NUMBA_VERSION is None


class TestGetNumbaStatus:
    """Tests for get_numba_status() function."""

    def test_returns_dict(self):
        """get_numba_status should return a dictionary."""
        from embrs.utilities.numba_utils import get_numba_status
        status = get_numba_status()
        assert isinstance(status, dict)

    def test_contains_required_keys(self):
        """Status dict should contain all required keys."""
        from embrs.utilities.numba_utils import get_numba_status
        status = get_numba_status()

        required_keys = ['available', 'version', 'jit_enabled', 'disable_jit_env']
        for key in required_keys:
            assert key in status, f"Missing key: {key}"

    def test_available_matches_flag(self):
        """Status 'available' should match NUMBA_AVAILABLE."""
        from embrs.utilities.numba_utils import get_numba_status, NUMBA_AVAILABLE
        status = get_numba_status()
        assert status['available'] == NUMBA_AVAILABLE


class TestJitIfEnabled:
    """Tests for jit_if_enabled decorator."""

    def test_decorator_returns_callable(self):
        """jit_if_enabled should return a callable decorator."""
        from embrs.utilities.numba_utils import jit_if_enabled

        @jit_if_enabled(nopython=True)
        def test_func(x):
            return x * 2

        assert callable(test_func)

    def test_decorated_function_works(self):
        """Decorated function should produce correct results."""
        from embrs.utilities.numba_utils import jit_if_enabled

        @jit_if_enabled(nopython=True, cache=True)
        def add_arrays(a, b):
            result = np.empty_like(a)
            for i in range(len(a)):
                result[i] = a[i] + b[i]
            return result

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = add_arrays(a, b)

        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_numerical_precision_preserved(self):
        """JIT compilation should preserve numerical precision."""
        from embrs.utilities.numba_utils import jit_if_enabled

        @jit_if_enabled(nopython=True)
        def compute_sum(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total

        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = compute_sum(arr)

        # Compare with numpy sum
        expected = np.sum(arr)
        assert result == pytest.approx(expected, rel=1e-14)


class TestNjitIfEnabled:
    """Tests for njit_if_enabled decorator."""

    def test_decorator_returns_callable(self):
        """njit_if_enabled should return a callable decorator."""
        from embrs.utilities.numba_utils import njit_if_enabled

        @njit_if_enabled()
        def test_func(x):
            return x * 2

        assert callable(test_func)

    def test_decorated_function_works(self):
        """Decorated function should produce correct results."""
        from embrs.utilities.numba_utils import njit_if_enabled

        @njit_if_enabled(cache=True)
        def multiply_sum(a, b):
            result = 0.0
            for i in range(len(a)):
                result += a[i] * b[i]
            return result

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = multiply_sum(a, b)

        expected = np.dot(a, b)
        assert result == pytest.approx(expected, rel=1e-14)


class TestGetPrange:
    """Tests for get_prange() helper."""

    def test_returns_callable(self):
        """get_prange should return a range-like callable."""
        from embrs.utilities.numba_utils import get_prange
        prange_func = get_prange()
        assert callable(prange_func)

    def test_prange_works_in_loop(self):
        """Returned prange should work in a for loop."""
        from embrs.utilities.numba_utils import get_prange
        prange_func = get_prange()

        result = []
        for i in prange_func(5):
            result.append(i)

        assert result == [0, 1, 2, 3, 4]


class TestNumbaIntegration:
    """Integration tests for Numba with typical EMBRS patterns."""

    def test_finite_difference_pattern(self):
        """Test typical finite difference computation pattern."""
        from embrs.utilities.numba_utils import jit_if_enabled

        @jit_if_enabled(nopython=True)
        def finite_diff_step(u, dx, dt):
            """Simplified finite difference step."""
            n = len(u)
            u_new = np.empty(n)

            # Boundary conditions
            u_new[0] = u[0]
            u_new[n-1] = u[n-1]

            # Interior points
            alpha = dt / (dx * dx)
            for i in range(1, n-1):
                u_new[i] = u[i] + alpha * (u[i+1] - 2*u[i] + u[i-1])

            return u_new

        u = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        result = finite_diff_step(u, 0.1, 0.001)

        assert len(result) == len(u)
        assert result[0] == u[0]  # Boundary preserved
        assert result[-1] == u[-1]  # Boundary preserved
        assert not np.any(np.isnan(result))

    def test_exponential_computation(self):
        """Test exponential computations (common in moisture model)."""
        from embrs.utilities.numba_utils import jit_if_enabled

        @jit_if_enabled(nopython=True)
        def compute_with_exp(t):
            """Computation with exponentials like in moisture model."""
            result = np.empty(len(t))
            for i in range(len(t)):
                tk = t[i] + 273.2
                qv = 13550.0 - 10.22 * tk
                ps = 0.0000239 * np.exp(20.58 - (5205.0 / tk))
                result[i] = qv * ps
            return result

        temps = np.array([10.0, 20.0, 30.0, 40.0])
        result = compute_with_exp(temps)

        assert len(result) == 4
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all(result > 0)  # Physical constraint

    def test_conditional_logic(self):
        """Test conditional logic patterns."""
        from embrs.utilities.numba_utils import jit_if_enabled

        @jit_if_enabled(nopython=True)
        def classify_values(arr, threshold):
            """Classify values above/below threshold."""
            result = np.empty(len(arr), dtype=np.int32)
            for i in range(len(arr)):
                if arr[i] > threshold:
                    result[i] = 1
                elif arr[i] < -threshold:
                    result[i] = -1
                else:
                    result[i] = 0
            return result

        arr = np.array([-2.0, -0.5, 0.1, 0.8, 1.5])
        result = classify_values(arr, 1.0)

        expected = np.array([-1, 0, 0, 0, 1], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)


class TestWarmupJitCache:
    """Tests for warmup_jit_cache function."""

    def test_warmup_does_not_error(self):
        """warmup_jit_cache should not raise exceptions."""
        from embrs.utilities.numba_utils import warmup_jit_cache

        # Should not raise any exception
        warmup_jit_cache()

    def test_warmup_idempotent(self):
        """Calling warmup multiple times should be safe."""
        from embrs.utilities.numba_utils import warmup_jit_cache

        # Should be safe to call multiple times
        warmup_jit_cache()
        warmup_jit_cache()
        warmup_jit_cache()


class TestDisableJitFlag:
    """Tests for JIT disable functionality."""

    def test_disable_jit_env_recognized(self):
        """EMBRS_DISABLE_JIT environment variable should be recognized."""
        from embrs.utilities.numba_utils import get_numba_status, DISABLE_JIT

        status = get_numba_status()
        assert 'disable_jit_env' in status
        assert status['disable_jit_env'] == DISABLE_JIT

    def test_functions_still_work_without_jit(self):
        """Functions decorated with jit_if_enabled should work without JIT."""
        from embrs.utilities.numba_utils import jit_if_enabled

        # This should work regardless of JIT status
        @jit_if_enabled(nopython=True)
        def simple_add(a, b):
            return a + b

        result = simple_add(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        expected = np.array([4.0, 6.0])
        np.testing.assert_array_equal(result, expected)
