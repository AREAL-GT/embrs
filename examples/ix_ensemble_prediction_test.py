"""
Test control class for ensemble fire prediction.

This example demonstrates how to use the ensemble prediction feature to run
multiple predictions with different initial state estimates and aggregate
them into probabilistic fire spread predictions.

To use this example:
1. Start a fire simulation
2. Select this file as the "User Module"
3. The ensemble will run after 30 minutes of simulation time
"""

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import multiprocessing as mp


from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim
from embrs.tools.fire_predictor import FirePredictor
from embrs.utilities.ensemble_video import create_ensemble_video_from_predictor
from embrs.utilities.data_classes import PredictorParams, StateEstimate
from embrs.utilities.fire_util import UtilFuncs


class EnsemblePredictionTest(ControlClass):
    """
    Test control class for ensemble fire prediction.

    This class demonstrates:
    - Creating a FirePredictor instance
    - Generating multiple StateEstimate objects
    - Running ensemble predictions in parallel
    - Analyzing and displaying results
    """

    def __init__(self, fire: FireSim):
        """
        Initialize the ensemble prediction test.

        Args:
            fire: The FireSim instance
        """
        self.fire = fire
        self.predictor = None
        self.ensemble_run = False
        self.trigger_time_hr = 2

        # Configuration for ensemble
        self.n_ensemble = 20  # Number of ensemble members
        self.prediction_horizon_hr = 12.0  # How far ahead to predict
        self.use_seeds = True  # For reproducibility

        print("\n" + "="*70)
        print("Ensemble Prediction Test Initialized")
        print("="*70)
        print(f"  Trigger time: {self.trigger_time_hr} hours")
        print(f"  Ensemble size: {self.n_ensemble} members")
        print(f"  Prediction horizon: {self.prediction_horizon_hr} hours")
        print("="*70 + "\n")

    def process_state(self, fire: FireSim):
        """
        Process fire state and run ensemble prediction at trigger time.

        Args:
            fire: The FireSim instance
        """
        # Run ensemble prediction once after trigger time
        if not self.ensemble_run and fire.curr_time_h >= self.trigger_time_hr:
            self.ensemble_run = True

            print("\n" + "="*70)
            print(f"TRIGGER: Fire time = {fire.curr_time_h:.2f} hours")
            print("Starting ensemble prediction...")
            print("="*70 + "\n")

            # Run the ensemble prediction
            self._run_ensemble_prediction(fire)

    def _run_ensemble_prediction(self, fire: FireSim):
        """
        Execute ensemble prediction and analyze results.

        Args:
            fire: The FireSim instance
        """
        # Step 1: Create predictor if not already created
        if self.predictor is None:
            print("Creating FirePredictor...")
            predictor_params = self._create_predictor_params()
            self.predictor = FirePredictor(predictor_params, fire)
            print("✓ FirePredictor created\n")

        # Step 2: Generate ensemble of state estimates
        print(f"Generating {self.n_ensemble} state estimates...")
        state_estimates = self._generate_state_estimates(fire)
        print(f"✓ Generated {len(state_estimates)} state estimates\n")

        # Step 3: Generate random seeds if requested
        seeds = None
        if self.use_seeds:
            seeds = [42 + i for i in range(self.n_ensemble)]
            print(f"Using random seeds: {seeds}\n")

        # Step 4: Run ensemble prediction
        try:
            print("Running ensemble prediction in parallel...")
            result = self.predictor.run_ensemble(
                state_estimates=state_estimates,
                num_workers=min(mp.cpu_count(), self.n_ensemble),  # Use up to 4 workers
                random_seeds=seeds,
                visualize=True,
                return_individual=True  # Include individual predictions for analysis
            )
            print("✓ Ensemble prediction complete!\n")

            # Step 5: Analyze and display results
            self._analyze_results(result, fire)

            # Step 6: Create video visualization
            self._create_ensemble_video(result)

        except Exception as e:
            print(f"\n✗ Ensemble prediction failed: {e}")
            import traceback
            traceback.print_exc()

    def _create_predictor_params(self) -> PredictorParams:
        """
        Create predictor parameters for the ensemble.

        Returns:
            PredictorParams configured for this test
        """
        params = PredictorParams(
            time_horizon_hr=self.prediction_horizon_hr,
            cell_size_m=45.0,  # Same as fire sim
            time_step_s=20.0,   # Same as fire sim

            # Fuel moisture (can be constant or from fire state)
            dead_mf=0.10,
            live_mf=0.3,

            # Wind uncertainty (creates variation in ensemble members)
            wind_speed_bias=0.0,      # No systematic bias
            wind_dir_bias=0.0,        # No systematic bias
            max_wind_speed_bias=2.0,  # Max random bias in m/s
            max_wind_dir_bias=15.0,   # Max random bias in degrees
            wind_uncertainty_factor=0.0,  # Medium uncertainty

            # ROS uncertainty
            ros_bias=0.0,  # No systematic bias

            # Spotting (if enabled in main sim)
            model_spotting=False,
            spot_delay_s=300.0,  # 5 minute delay
        )

        return params

    def _generate_state_estimates(self, fire: FireSim) -> list:
        """
        Generate ensemble of state estimates by perturbing the current fire state.

        This method creates uncertainty in the initial fire state by:
        1. Using the exact current state (member 0)
        2. Expanding the burning boundary slightly (members 1-2)
        3. Contracting the burning boundary slightly (members 3-4)

        Args:
            fire: The FireSim instance

        Returns:
            List of StateEstimate objects
        """
        state_estimates = []

        # Get current fire polygons
        base_burnt = UtilFuncs.get_cell_polygons(fire._burnt_cells) if fire._burnt_cells else []
        base_burning = UtilFuncs.get_cell_polygons(fire.burning_cells)

        # Member 0: Exact current state (baseline)
        state_estimates.append(StateEstimate(
            burnt_polys=base_burnt,
            burning_polys=base_burning
        ))

        # Create perturbed versions
        cell_size = fire.cell_size

        for i in range(1, self.n_ensemble):
            # Alternate between expanding and contracting
            if i % 2 == 1:
                # Expand burning region (optimistic - more fire)
                buffer_dist = cell_size * 0.5 * (i // 2 + 1)
                perturbed_burning = self._buffer_polygons(base_burning, buffer_dist)
            else:
                # Contract burning region (pessimistic - less fire)
                buffer_dist = -cell_size * 0.3 * (i // 2)
                perturbed_burning = self._buffer_polygons(base_burning, buffer_dist)

            state_estimates.append(StateEstimate(
                burnt_polys=base_burnt,  # Keep burnt region same
                burning_polys=perturbed_burning
            ))

        return state_estimates

    def _buffer_polygons(self, polygons: list, buffer_dist: float) -> list:
        """
        Buffer a list of polygons.

        Args:
            polygons: List of Polygon objects
            buffer_dist: Buffer distance (positive=expand, negative=contract)

        Returns:
            List of buffered polygons
        """
        if not polygons:
            return []

        try:
            # Merge polygons, buffer, then split back into list
            merged = unary_union(polygons)
            buffered = merged.buffer(buffer_dist)

            # Handle both single polygon and multipolygon results
            if buffered.is_empty:
                return []
            elif hasattr(buffered, 'geoms'):
                return list(buffered.geoms)
            else:
                return [buffered]
        except Exception as e:
            print(f"Warning: Buffer operation failed: {e}")
            return polygons  # Return original on error


    # TODO: Update this function with more useful statistics
    def _analyze_results(self, result, fire: FireSim):
        """
        Analyze and display ensemble prediction results.

        Args:
            result: EnsemblePredictionOutput
            fire: The FireSim instance
        """
        print("\n" + "="*70)
        print("ENSEMBLE PREDICTION RESULTS")
        print("="*70)

        # Overall statistics
        print(f"\nEnsemble size: {result.n_ensemble} members")
        print(f"Successful members: {result.n_ensemble}")

        # Time steps in prediction
        time_steps = sorted(result.burn_probability.keys())
        print(f"Prediction time steps: {len(time_steps)}")
        if time_steps:
            print(f"  First: {time_steps[0]:.0f}s ({time_steps[0]/3600:.2f}h)")
            print(f"  Last:  {time_steps[-1]:.0f}s ({time_steps[-1]/3600:.2f}h)")

        # Burn probability statistics
        print(f"\n--- Burn Probability Analysis ---")

        # Find cells with high burn probability at final time
        if time_steps:
            final_time = time_steps[-1]
            probs = result.burn_probability[final_time]

            print(f"\nFinal time ({final_time/3600:.2f}h): {len(probs)} cells with fire")

            # Categorize by probability
            high_prob = {loc: p for loc, p in probs.items() if p >= 0.8}
            med_prob = {loc: p for loc, p in probs.items() if 0.5 <= p < 0.8}
            low_prob = {loc: p for loc, p in probs.items() if p < 0.5}

            print(f"  High probability (≥80%): {len(high_prob)} cells")
            print(f"  Medium probability (50-80%): {len(med_prob)} cells")
            print(f"  Low probability (<50%): {len(low_prob)} cells")

            if probs:
                prob_values = list(probs.values())
                print(f"\n  Probability statistics:")
                print(f"    Mean: {np.mean(prob_values):.2%}")
                print(f"    Median: {np.median(prob_values):.2%}")
                print(f"    Std: {np.std(prob_values):.2%}")

        # Flame length statistics
        if result.flame_len_m_stats:
            print(f"\n--- Flame Length Statistics ---")
            print(f"Cells with flame data: {len(result.flame_len_m_stats)}")

            mean_flames = [stats.mean for stats in result.flame_len_m_stats.values()]
            std_flames = [stats.std for stats in result.flame_len_m_stats.values()]

            print(f"  Mean flame length across cells:")
            print(f"    Average: {np.mean(mean_flames):.2f} m")
            print(f"    Range: {np.min(mean_flames):.2f} - {np.max(mean_flames):.2f} m")
            print(f"  Uncertainty (std) across cells:")
            print(f"    Average: {np.mean(std_flames):.2f} m")
            print(f"    Max: {np.max(std_flames):.2f} m")

        # ROS statistics
        if result.ros_ms_stats:
            print(f"\n--- Rate of Spread Statistics ---")
            mean_ros = [stats.mean for stats in result.ros_ms_stats.values()]
            print(f"  Mean ROS across cells:")
            print(f"    Average: {np.mean(mean_ros):.4f} m/s")
            print(f"    Range: {np.min(mean_ros):.4f} - {np.max(mean_ros):.4f} m/s")

        # Crown fire frequency
        if result.crown_fire_frequency:
            print(f"\n--- Crown Fire Analysis ---")
            crown_cells = [loc for loc, freq in result.crown_fire_frequency.items() if freq > 0]
            print(f"Cells with crown fire in any member: {len(crown_cells)}")

            if crown_cells:
                crown_freqs = [result.crown_fire_frequency[loc] for loc in crown_cells]
                print(f"  Crown fire frequency:")
                print(f"    Mean: {np.mean(crown_freqs):.2%}")
                print(f"    Max: {np.max(crown_freqs):.2%}")

        # Individual member comparison
        if result.individual_predictions:
            print(f"\n--- Individual Member Comparison ---")
            for i, pred in enumerate(result.individual_predictions):
                final_spread = len(pred.spread.get(max(pred.spread.keys()), []))
                print(f"  Member {i}: {final_spread} cells burning at final time")

        # High-risk areas (example query)
        print(f"\n--- High-Risk Areas (≥90% probability) ---")
        high_risk_count = 0
        if time_steps:
            for time_s in time_steps[-3:]:  # Check last 3 time steps
                if time_s in result.burn_probability:
                    high_risk = [(loc, p) for loc, p in result.burn_probability[time_s].items()
                                if p >= 0.9]
                    if high_risk:
                        high_risk_count += len(high_risk)
                        print(f"\n  Time {time_s/3600:.2f}h: {len(high_risk)} high-risk cells")
                        # Show a few examples
                        for loc, prob in sorted(high_risk, key=lambda x: x[1], reverse=True)[:5]:
                            print(f"    ({loc[0]:.0f}, {loc[1]:.0f}): {prob:.1%}")

        if high_risk_count == 0:
            print("  No cells with ≥90% probability found")

        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70 + "\n")

        # Save summary to file
        self._save_results_summary(result, fire)

    def _create_ensemble_video(self, result):
        """
        Create a video visualization of the ensemble prediction.

        Args:
            result: EnsemblePredictionOutput
        """
        try:
            from datetime import datetime

            print("\n" + "="*70)
            print("CREATING ENSEMBLE PREDICTION VIDEO")
            print("="*70 + "\n")

            # Create output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"ensemble_prediction_{timestamp}.mp4"

            # Get cell size from predictor params
            cell_size = self._create_predictor_params().cell_size_m

            # Create the video
            output_path = create_ensemble_video_from_predictor(
                predictor=self.predictor,
                ensemble_output=result,
                output_path=video_path,
                title=f"Ensemble Fire Spread Prediction ({result.n_ensemble} members)",
                fps=8,  # Slightly slower for better viewing
                dpi=150,
                colormap="YlOrRd",  # Yellow-Orange-Red gradient
                show_progress=True
            )

            print("\n" + "="*70)
            print(f"Video saved to: {output_path}")
            print("="*70 + "\n")

        except Exception as e:
            print(f"\nWarning: Could not create video: {e}")
            import traceback
            traceback.print_exc()

    def _save_results_summary(self, result, fire: FireSim):
        """
        Save ensemble results summary to a text file.

        Args:
            result: EnsemblePredictionOutput
            fire: The FireSim instance
        """
        try:
            import os
            from datetime import datetime

            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_results_{timestamp}.txt"

            with open(filename, 'w') as f:
                f.write("="*70 + "\n")
                f.write("ENSEMBLE PREDICTION RESULTS SUMMARY\n")
                f.write("="*70 + "\n\n")

                f.write(f"Simulation time: {fire.curr_time_h:.2f} hours\n")
                f.write(f"Ensemble size: {result.n_ensemble}\n")
                f.write(f"Prediction horizon: {self.prediction_horizon_hr} hours\n\n")

                # Write burn probability summary
                time_steps = sorted(result.burn_probability.keys())
                if time_steps:
                    f.write("Burn Probability Summary:\n")
                    f.write("-" * 40 + "\n")
                    for time_s in time_steps[::max(1, len(time_steps)//10)]:  # Sample times
                        probs = result.burn_probability[time_s]
                        high = sum(1 for p in probs.values() if p >= 0.8)
                        f.write(f"  {time_s/3600:.2f}h: {len(probs)} cells, {high} high-prob\n")

                f.write("\n" + "="*70 + "\n")

            print(f"Results summary saved to: {filename}")

        except Exception as e:
            print(f"Warning: Could not save results summary: {e}")


# For testing without a full simulation
if __name__ == "__main__":
    print("This module should be loaded as a control class in an EMBRS simulation.")
    print("To use:")
    print("  1. Start a fire simulation")
    print("  2. Select this file as the 'User Module'")
    print("  3. The ensemble will run after 30 minutes of simulation time")
