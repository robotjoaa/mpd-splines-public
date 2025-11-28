"""
Example script showing how to plot COBL trajectory results in original scale.

This demonstrates how to:
1. Load inference results
2. Load evaluation dataset with reference trajectories
3. Create comparison plots in original scale (meters)
"""

import os
import sys
import pickle
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from plot_trajectory_original_scale import (
    load_eval_dataset,
    plot_trajectory_comparison,
    animate_trajectory_comparison,
    convert_to_original_scale,
)


def load_inference_results(results_path):
    """
    Load inference results from pickle file.

    Args:
        results_path: Path to results pickle file

    Returns:
        Results dictionary
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def main():
    """
    Main example showing how to create comparison plots.
    """
    # ==================================================================================
    # CONFIGURATION - Update these paths to match your setup
    # ==================================================================================

    # Path to evaluation dataset
    eval_dataset_path = '/home/sisrel/pjw/mpd-splines-public/data_trajectories/EnvCoblEmpty2D-RobotPointMass2DBig/eval_dataset.pkl'

    # Path to inference results (you need to run inference first)
    # Example: results from launch_inference-experiments-cobl.py
    results_dir = '/home/sisrel/pjw/mpd-splines-public/outputs/launch_inference-experiments-cobl_YYYY-MM-DD_HH-MM-SS'
    results_file = 'results.pkl'  # Adjust based on your actual results file name

    # Output directory for plots
    output_dir = '/tmp/cobl_trajectory_comparison'
    os.makedirs(output_dir, exist_ok=True)

    # Which scenarios to plot (set to None to plot all)
    scenarios_to_plot = [0, 1, 2, 3, 4]  # Plot first 5 scenarios

    # ==================================================================================
    # LOAD DATA
    # ==================================================================================

    print("Loading evaluation dataset...")
    eval_data = load_eval_dataset(eval_dataset_path)

    scale_factor = eval_data['scale_factor']
    n_scenarios = eval_data['n_scenarios']
    horizon = eval_data['horizon']
    dt = eval_data['dt']

    print(f"\nEval Dataset Info:")
    print(f"  Scale factor: {scale_factor}")
    print(f"  Scenarios: {n_scenarios}")
    print(f"  Horizon: {horizon}")
    print(f"  Time step: {dt}s")
    print(f"  Total duration: {(horizon-1)*dt:.1f}s")
    print(f"  Obstacle radius: {eval_data['obstacle_radius'] * scale_factor:.3f}m")

    # Load inference results
    results_path = os.path.join(results_dir, results_file)

    if not os.path.exists(results_path):
        print(f"\n⚠ WARNING: Results file not found: {results_path}")
        print("\nTo use this script:")
        print("1. Run inference using launch_inference-experiments-cobl.py")
        print("2. Update the results_dir and results_file variables above")
        print("\nFor now, creating example plots with synthetic data...")

        # Use synthetic data for demonstration
        use_synthetic = True
    else:
        print(f"\nLoading inference results from: {results_path}")
        inference_results = load_inference_results(results_path)
        use_synthetic = False

    # ==================================================================================
    # CREATE PLOTS
    # ==================================================================================

    if scenarios_to_plot is None:
        scenarios_to_plot = range(n_scenarios)

    for idx in scenarios_to_plot:
        if idx >= n_scenarios:
            break

        print(f"\n{'='*80}")
        print(f"Processing scenario {idx}/{n_scenarios-1}")
        print(f"{'='*80}")

        # Get reference trajectory (already in original scale - meters)
        reference_traj = eval_data['ego_trajectories'][idx]
        obstacle_trajs = eval_data['obstacle_trajectories'][idx]

        # Get predicted trajectory
        if use_synthetic:
            # Create synthetic prediction for demonstration
            # Add noise to reference trajectory and convert to normalized scale
            noise = np.random.randn(*reference_traj.shape) * 0.3  # meters
            predicted_traj_orig = reference_traj + noise
            predicted_traj_norm = predicted_traj_orig / scale_factor
        else:
            # Extract from inference results
            # Adjust this based on your actual results structure
            predicted_traj_norm = inference_results['trajectories'][idx]

            if isinstance(predicted_traj_norm, torch.Tensor):
                predicted_traj_norm = predicted_traj_norm.cpu().numpy()

        # Get start and goal (in normalized scale)
        start_pos = predicted_traj_norm[0]
        goal_pos = predicted_traj_norm[-1]

        # ==================================================================================
        # STATIC PLOT
        # ==================================================================================

        print(f"  Creating static comparison plot...")
        plot_path = os.path.join(output_dir, f'comparison_scenario_{idx:03d}_static.png')

        plot_trajectory_comparison(
            predicted_traj=predicted_traj_norm,
            reference_traj=reference_traj,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacle_trajs,
            time=horizon // 2,  # Mid-point
            scale_factor=scale_factor,
            title=f"Trajectory Comparison - Scenario {idx}",
            save_path=plot_path,
            show_plot=False,
            obstacle_radius=eval_data['obstacle_radius'] * scale_factor,
        )

        # ==================================================================================
        # ANIMATED PLOT
        # ==================================================================================

        print(f"  Creating animated comparison...")
        video_path = os.path.join(output_dir, f'comparison_scenario_{idx:03d}_animation.mp4')

        animate_trajectory_comparison(
            predicted_traj=predicted_traj_norm,
            reference_traj=reference_traj,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacle_trajs,
            scale_factor=scale_factor,
            dt=dt,
            save_path=video_path,
            obstacle_radius=eval_data['obstacle_radius'] * scale_factor,
            fps=10,
        )

        # ==================================================================================
        # COMPUTE METRICS
        # ==================================================================================

        # Convert predicted to original scale for comparison
        predicted_traj_orig = convert_to_original_scale(predicted_traj_norm, scale_factor)

        # Path length
        ref_path_length = np.sum(np.linalg.norm(np.diff(reference_traj, axis=0), axis=1))
        pred_path_length = np.sum(np.linalg.norm(np.diff(predicted_traj_orig, axis=0), axis=1))

        # Endpoint error
        endpoint_error = np.linalg.norm(predicted_traj_orig[-1] - reference_traj[-1])

        # Average trajectory error
        traj_error = np.mean(np.linalg.norm(predicted_traj_orig - reference_traj, axis=1))

        print(f"\n  Metrics (Original Scale - Meters):")
        print(f"    Reference path length: {ref_path_length:.3f}m")
        print(f"    Predicted path length: {pred_path_length:.3f}m")
        print(f"    Endpoint error: {endpoint_error:.3f}m")
        print(f"    Average trajectory error: {traj_error:.3f}m")

    print(f"\n{'='*80}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*80}")

    # ==================================================================================
    # PRINT USAGE INSTRUCTIONS
    # ==================================================================================

    if use_synthetic:
        print("\n" + "="*80)
        print("📝 INSTRUCTIONS FOR USING WITH REAL INFERENCE RESULTS")
        print("="*80)
        print("""
1. Run inference on COBL dataset:
   cd scripts/inference
   python launch_inference-experiments-cobl.py

2. Update the paths in this script:
   - results_dir: Path to your inference output directory
   - results_file: Name of your results pickle file

3. Adjust data extraction based on your results structure:
   - Check the keys in your results dictionary
   - Update line: predicted_traj_norm = inference_results['trajectories'][idx]

4. Run this script again to create comparison plots with real predictions

Note: The comparison plots show:
  - Green line: Reference trajectory from COBL dataset (ground truth)
  - Blue line: Predicted trajectory from your planner
  - All measurements are in METERS (original scale)
  - Obstacles are shown at the mid-point of the trajectory
        """)


if __name__ == "__main__":
    main()
