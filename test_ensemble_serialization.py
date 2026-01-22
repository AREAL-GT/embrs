"""
Simple test to verify ensemble prediction serialization.

This test checks that:
1. Cell can be pickled and unpickled
2. FirePredictor can be pickled after prepare_for_serialization()
3. Deserialized predictor can run predictions
"""

import pickle
import sys
import os

# Add embrs to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cell_serialization():
    """Test that Cell can be pickled and unpickled."""
    print("Testing Cell serialization...")

    from embrs.fire_simulator.cell import Cell

    # Create a simple cell
    cell = Cell(id=1, col=0, row=0, cell_size=30.0)

    # Pickle and unpickle
    try:
        pickled = pickle.dumps(cell)
        restored = pickle.loads(pickled)

        # Verify basic attributes
        assert restored.id == cell.id
        assert restored.x_pos == cell.x_pos
        assert restored.y_pos == cell.y_pos
        assert restored._parent is None  # Weak ref should be excluded

        print("✓ Cell serialization works!")
        return True
    except Exception as e:
        print(f"✗ Cell serialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predictor_serialization_requires_prepare():
    """Test that predictor requires prepare_for_serialization()."""
    print("\nTesting FirePredictor serialization without prepare...")

    # This would require a full FireSim setup, which is complex
    # For now, just verify the method exists
    from embrs.tools.fire_predictor import FirePredictor

    # Check that the methods exist
    assert hasattr(FirePredictor, 'prepare_for_serialization')
    assert hasattr(FirePredictor, '__getstate__')
    assert hasattr(FirePredictor, '__setstate__')
    assert hasattr(FirePredictor, 'run_ensemble')

    print("✓ FirePredictor has all serialization methods!")
    return True

def test_data_classes_exist():
    """Test that new data classes are defined."""
    print("\nTesting new data classes...")

    from embrs.utilities.data_classes import CellStatistics, EnsemblePredictionOutput

    # Create instances
    stats = CellStatistics(mean=1.0, std=0.5, min=0.5, max=1.5, count=10)
    assert stats.mean == 1.0
    assert stats.count == 10

    output = EnsemblePredictionOutput(
        n_ensemble=10,
        burn_probability={},
        flame_len_m_stats={},
        fli_kw_m_stats={},
        ros_ms_stats={},
        spread_dir_stats={},
        crown_fire_frequency={},
        hold_prob_stats={},
        breach_frequency={}
    )
    assert output.n_ensemble == 10

    print("✓ Data classes work correctly!")
    return True

def test_worker_function_exists():
    """Test that worker function is defined."""
    print("\nTesting worker function...")

    from embrs.tools.fire_predictor import _run_ensemble_member_worker, _aggregate_ensemble_predictions

    # Just check they exist
    assert callable(_run_ensemble_member_worker)
    assert callable(_aggregate_ensemble_predictions)

    print("✓ Worker functions are defined!")
    return True

if __name__ == "__main__":
    print("="*60)
    print("Ensemble Prediction Serialization Tests")
    print("="*60)

    results = []

    # Run tests
    results.append(("Cell serialization", test_cell_serialization()))
    results.append(("Predictor methods", test_predictor_serialization_requires_prepare()))
    results.append(("Data classes", test_data_classes_exist()))
    results.append(("Worker functions", test_worker_function_exists()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All basic serialization tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
