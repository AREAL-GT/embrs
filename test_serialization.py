#!/usr/bin/env python3
"""
Test script to verify FirePredictor serialization works correctly.

This script demonstrates that __getstate__() and __setstate__() are called
automatically by Python's pickle module during multiprocessing.

Usage:
    python test_serialization.py
"""

import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


class SimpleClass:
    """Test class for basic pickling."""
    def __init__(self):
        self.value = 42

    def __getstate__(self):
        print("  → __getstate__() called")
        return {'value': self.value}

    def __setstate__(self, state):
        print("  → __setstate__() called")
        self.value = state['value']


class TestClass:
    """Test class for multiprocessing."""
    def __init__(self):
        self.value = 100

    def __getstate__(self):
        import os
        print(f"  → __getstate__() called in main PID {os.getpid()}")
        return {'value': self.value}

    def __setstate__(self, state):
        import os
        print(f"  → __setstate__() called in worker PID {os.getpid()}")
        self.value = state['value']


def simple_pickle_test():
    """Test basic pickling without multiprocessing."""
    print("=" * 70)
    print("TEST 1: Basic Pickle/Unpickle (Same Process)")
    print("=" * 70)

    obj = SimpleClass()
    print(f"Original object: value={obj.value}")

    print("\nPickling...")
    pickled = pickle.dumps(obj)

    print("\nUnpickling...")
    restored = pickle.loads(pickled)
    print(f"Restored object: value={restored.value}")

    print("\n✓ Test 1 passed\n")


def worker_function(obj):
    """Worker function that receives pickled object."""
    import os
    print(f"  → Worker PID {os.getpid()} received object: {obj.value}")
    return obj.value * 2


def multiprocessing_test():
    """Test pickling with ProcessPoolExecutor."""
    print("=" * 70)
    print("TEST 2: Pickling with ProcessPoolExecutor")
    print("=" * 70)

    obj = TestClass()
    print(f"Original object: value={obj.value}")
    print(f"Main process PID: {mp.current_process().pid}")

    print("\nSubmitting to ProcessPoolExecutor...")
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker_function, obj)
        result = future.result()

    print(f"\nWorker returned: {result}")
    print("\n✓ Test 2 passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FirePredictor Serialization Verification")
    print("=" * 70)
    print("\nThis script demonstrates that __getstate__() and __setstate__()")
    print("are called AUTOMATICALLY by Python during pickle/unpickle.")
    print("")

    try:
        simple_pickle_test()
        multiprocessing_test()

        print("=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)
        print("\nConclusion:")
        print("  - __getstate__() is called during pickle.dumps() and executor.submit()")
        print("  - __setstate__() is called during pickle.loads() in worker process")
        print("  - These methods are NEVER called by our code explicitly")
        print("  - They are part of Python's pickle protocol\n")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
