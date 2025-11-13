import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from torch_robotics.torch_utils.torch_utils import (
    to_numpy,
)
from dotmap import DotMap

# load results dict from experiment
LOG_DIR = "/home/sisrel/pjw/mpd-splines-public/scripts/inference/logs"

def load_results(exp_name, option_l):
    all_results = {}

    # find
    # pattern = os.path.join(LOG_DIR, exp_name,"**/results_single_plan-*")
    pattern = os.path.join(LOG_DIR, exp_name,"**/args.yaml")
    exp_base = glob.glob(pattern, recursive = True)
    exp_base = [os.path.dirname(exp) for exp in exp_base]
    print("num settings : ", len(exp_base))
    for exp in exp_base :
        n_idx = []
        for option in option_l :
            match_idx = [i for i, p in enumerate(option) if p in exp]
            assert len(match_idx) == 1
            n_idx.append(match_idx[0])

        selected_name = tuple(n_idx)
        pattern = os.path.join(exp,"results_single_plan-*")
        file_names = glob.glob(pattern)
        n_exps = len(file_names)
        result = []
        # for i in range(n_exps) :
        #     tmp_result = torch.load(
        #         os.path.join(exp, f"results_single_plan-{i:03d}.pt"),
        #         weights_only=False
        #     )

        for filename in file_names : 
            tmp_result = torch.load(
                os.path.join(exp,os.path.basename(filename)),
                weights_only=False
            )
            result.append(tmp_result)
        print(selected_name)
        all_results[selected_name] = result

    return all_results, option_l


def get_option_name_str(option_tuple, option_l):
    """Convert option index tuple to readable string using option_l."""
    parts = []
    for idx, option_idx in enumerate(option_tuple):
        if idx < len(option_l):
            parts.append(option_l[idx][option_idx])
    return "-".join(parts)


def plot_start_goal_problems(all_results, option_l=None, save_dir=None, show_trajectories=True):
    """
    Plot all start/goal/obstacle samples for each experimental option.
    Creates one plot per option showing all 100 (or however many) problem instances overlaid.

    Args:
        all_results: Dict mapping experiment options to list of results
        option_l: List of option lists for converting indices to names
        save_dir: Directory to save plots (if None, displays plots)
        show_trajectories: Whether to show moving obstacle trajectories
    """
    print(f"Plotting problems for {len(all_results)} experimental options")

    # Create one plot per experimental option
    for option_tuple, results_list in all_results.items():
        n_samples = len(results_list)

        # Get readable option name
        if option_l is not None:
            option_name = get_option_name_str(option_tuple, option_l)
        else:
            option_name = str(option_tuple)

        print(f"\nPlotting option {option_name}: {n_samples} samples")

        fig, ax = plt.subplots(figsize=(12, 12))

        # Collect all start/goal positions for this option
        all_starts = []
        all_goals = []

        # Plot all samples for this option
        for sample_idx, result in enumerate(results_list):
            q_start = to_numpy(result.q_pos_start)
            q_goal = to_numpy(result.q_pos_goal)

            all_starts.append(q_start)
            all_goals.append(q_goal)

            # Use different alpha/size for better visualization with many samples
            alpha_val = 0.6 if n_samples > 50 else 0.8
            start_size = 80 if n_samples > 50 else 120
            goal_size = 100 if n_samples > 50 else 150

            # Plot start (green circle)
            ax.scatter(q_start[0], q_start[1], c='green', s=start_size, marker='o',
                      edgecolors='black', linewidths=1, alpha=alpha_val, zorder=10,
                      label='Start' if sample_idx == 0 else '')

            # Plot goal (purple star)
            ax.scatter(q_goal[0], q_goal[1], c='purple', s=goal_size, marker='*',
                      edgecolors='black', linewidths=1, alpha=alpha_val, zorder=10,
                      label='Goal' if sample_idx == 0 else '')

            # Plot arrow from start to goal
            arrow_alpha = 0.15 if n_samples > 50 else 0.25
            arrow = FancyArrowPatch(
                (q_start[0], q_start[1]),
                (q_goal[0], q_goal[1]),
                arrowstyle='->',
                mutation_scale=15,
                linewidth=1.5,
                color='blue',
                alpha=arrow_alpha,
                zorder=5
            )
            ax.add_patch(arrow)

            # Plot dynamic obstacle trajectories if available
            if hasattr(result, 'dyn_obj_config') and result.dyn_obj_config is not None:
                for obj_idx, (obj_name, obj_config) in enumerate(result.dyn_obj_config.items()):
                    if 'keyframe_positions' in obj_config:
                        positions = obj_config['keyframe_positions']

                        # Obstacle colors
                        obs_start_color = 'red' if obj_idx == 0 else 'darkred'
                        obs_end_color = 'orange' if obj_idx == 0 else 'darkorange'

                        obs_alpha = 0.4 if n_samples > 50 else 0.6
                        obs_size = 40 if n_samples > 50 else 60

                        # Plot obstacle start position
                        label = f'Obstacle {obj_idx+1} start' if sample_idx == 0 and obj_idx < 2 else ''
                        ax.scatter(positions[0, 0], positions[0, 1],
                                  c=obs_start_color, s=obs_size, marker='s', alpha=obs_alpha, zorder=8,
                                  edgecolors='black', linewidths=0.5, label=label)

                        # Plot obstacle end position
                        label = f'Obstacle {obj_idx+1} end' if sample_idx == 0 and obj_idx < 2 else ''
                        ax.scatter(positions[1, 0], positions[1, 1],
                                  c=obs_end_color, s=obs_size, marker='s', alpha=obs_alpha*0.7, zorder=8,
                                  edgecolors='black', linewidths=0.5, label=label)

                        if show_trajectories:
                            # Plot trajectory line
                            traj_alpha = 0.2 if n_samples > 50 else 0.3
                            ax.plot([positions[0, 0], positions[1, 0]],
                                   [positions[0, 1], positions[1, 1]],
                                   '--', linewidth=1, color='red', alpha=traj_alpha, zorder=7)

        # Set limits and labels
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_title(f'Option: {option_name}\n({n_samples} problem instances)',
                    fontsize=14, fontweight='bold')

        # Create legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # Add info text
        info_text = f"Experimental Option: {option_name}\n"
        info_text += f"Number of samples: {n_samples}\n"
        if len(all_starts) > 0:
            all_starts_np = np.array(all_starts)
            all_goals_np = np.array(all_goals)
            info_text += f"Start range X: [{all_starts_np[:, 0].min():.2f}, {all_starts_np[:, 0].max():.2f}]\n"
            info_text += f"Start range Y: [{all_starts_np[:, 1].min():.2f}, {all_starts_np[:, 1].max():.2f}]\n"
            info_text += f"Goal range X: [{all_goals_np[:, 0].min():.2f}, {all_goals_np[:, 0].max():.2f}]\n"
            info_text += f"Goal range Y: [{all_goals_np[:, 1].min():.2f}, {all_goals_np[:, 1].max():.2f}]"

        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # Use option name in filename
            option_str = str(option_name).replace('(', '').replace(')', '').replace(', ', '_')
            save_path = os.path.join(save_dir, f'problems_option_{option_str}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    print(f"\nCompleted plotting for all {len(all_results)} options")


def extract_nested_metric(result, metric_path):
    """
    Extract a metric from a nested DotMap/dict structure.

    Args:
        result: DotMap result object
        metric_path: String path like 'trajs_all.success' or 't_inference_total'

    Returns:
        Metric value or None if not found
    """
    parts = metric_path.split('.')
    current = result

    for part in parts:
        if isinstance(current, (DotMap, dict)):
            if part in current:
                current = current[part]
            else:
                return None
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None

    return current


def aggregate_metrics(all_results, metric_paths):
    """
    Aggregate metrics from multiple results and calculate mean, std.
    Handles nested metric structures like trajs_all.success, trajs_valid.path_length_mean, etc.

    Args:
        all_results: Dict mapping experiment options to list of results
        metric_paths: List of metric paths (e.g., ['trajs_all.success', 't_inference_total'])

    Returns:
        aggregated_metrics: Dict mapping option names to metric statistics
    """
    aggregated_metrics = {}

    for option_name, results_list in all_results.items():
        n_samples = len(results_list)

        # Initialize metric storage
        metrics = {name: [] for name in metric_paths}

        # Collect metrics from each result
        for result in results_list:
            for metric_path in metric_paths:
                value = extract_nested_metric(result, metric_path)

                if value is not None:
                    # Convert to numpy if tensor
                    if hasattr(value, 'cpu'):
                        value = value.cpu().numpy()
                    # Convert to float if numpy array
                    if isinstance(value, np.ndarray):
                        value = float(value.item()) if value.size == 1 else float(value)
                    # Convert boolean to int
                    if isinstance(value, bool):
                        value = int(value)

                    metrics[metric_path].append(value)

        # Calculate statistics
        stats = {}
        for metric_path, values in metrics.items():
            if len(values) > 0:
                values_array = np.array(values)
                # Filter out None/NaN values
                values_array = values_array[~np.isnan(values_array)]

                if len(values_array) > 0:
                    stats[metric_path] = {
                        'mean': np.mean(values_array),
                        'std': np.std(values_array),
                        'min': np.min(values_array),
                        'max': np.max(values_array),
                        'n_samples': len(values_array),
                        'values': values_array
                    }

        aggregated_metrics[option_name] = stats

    return aggregated_metrics


def print_metric_summary(aggregated_metrics):
    """
    Print formatted summary of aggregated metrics.

    Args:
        aggregated_metrics: Dict from aggregate_metrics()
    """
    print("\n" + "="*80)
    print("METRIC SUMMARY")
    print("="*80)

    for option_name, metrics in aggregated_metrics.items():
        print(f"\nOption: {option_name}")
        print("-" * 80)

        for metric_name, stats in metrics.items():
            print(f"  {metric_name}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Std:  {stats['std']:.4f}")
            print(f"    Min:  {stats['min']:.4f}")
            print(f"    Max:  {stats['max']:.4f}")
            print(f"    N:    {stats['n_samples']}")


def plot_metric_comparison(aggregated_metrics, save_dir=None, option_l=None):
    """
    Create bar plots comparing metrics across different options.

    Args:
        aggregated_metrics: Dict from aggregate_metrics()
        save_dir: Directory to save plots (if None, displays plots)
    """
    # Get all metric names
    all_metric_names = set()
    for metrics in aggregated_metrics.values():
        all_metric_names.update(metrics.keys())

    # Sort for consistent ordering
    metric_names = sorted(all_metric_names)

    # Get option names
    option_names = list(aggregated_metrics.keys())
    n_options = len(option_names)

    # Metrics that should be in [0, 1] range
    ratio_metrics = {
        'success', 'fraction_valid', 'n_trajectory_fraction',
        'success_no_joint_limits_vel_acc', 'fraction_valid_no_joint_limits_vel_acc',
        'n_trajectories_free_fraction'
    }

    # Create subplots for each metric
    n_metrics = len(metric_names)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)
    fig.suptitle('Metric Comparison Across Options', fontsize=16)

    for idx, metric_name in enumerate(metric_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Collect data for this metric
        means = []
        stds = []
        labels = []

        for option_name in option_names:
            if metric_name in aggregated_metrics[option_name]:
                stats = aggregated_metrics[option_name][metric_name]
                means.append(stats['mean'])
                stds.append(stats['std'])
                if option_l is not None :
                    labels.append(get_option_name_str(option_name, option_l))
                else :
                    labels.append(str(option_name))

        # Create bar plot
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                      color='steelblue', edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Option', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Check if this metric should be in [0, 1] range
        metric_base = metric_name.split('.')[-1]  # Get the last part of the metric path
        if any(ratio_name in metric_base for ratio_name in ratio_metrics):
            ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}\n±{std:.3f}',
                   ha='center', va='bottom', fontsize=7)

    # Remove empty subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'metric_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metric comparison to {save_path}")
        plt.close()
    else:
        plt.show()


def find_failed_problems(all_results, option_l=None, save_dir=None):
    """
    Find problem configurations where NO valid trajectory was found.

    Args:
        all_results: Dict mapping experiment options to list of results
        option_l: List of option lists for converting indices to names
        save_dir: Directory to save failed problems report

    Returns:
        Dict mapping option names to list of failed problem indices and details
    """
    failed_problems = {}

    for option_tuple, results_list in all_results.items():
        # Get readable option name
        if option_l is not None:
            option_name = get_option_name_str(option_tuple, option_l)
        else:
            option_name = str(option_tuple)

        failed_list = []

        for idx, result in enumerate(results_list):
            # Check if there are NO valid trajectories
            has_valid = False

            # Check if q_trajs_pos_valid exists and is not None/empty
            if hasattr(result, 'q_trajs_pos_valid') and result.q_trajs_pos_valid is not None:
                if hasattr(result.q_trajs_pos_valid, 'shape'):
                    has_valid = result.q_trajs_pos_valid.shape[0] > 0
                else:
                    has_valid = len(result.q_trajs_pos_valid) > 0

            # Also check metrics.trajs_all.success if available
            if hasattr(result, 'metrics') and hasattr(result.metrics, 'trajs_all'):
                if hasattr(result.metrics.trajs_all, 'success'):
                    has_valid = result.metrics.trajs_all.success > 0

            if not has_valid:
                # Extract problem configuration
                problem_info = {
                    'idx': idx,
                    'q_pos_start': to_numpy(result.q_pos_start) if hasattr(result, 'q_pos_start') else None,
                    'q_pos_goal': to_numpy(result.q_pos_goal) if hasattr(result, 'q_pos_goal') else None,
                }

                # Add dynamic obstacle info if available
                if hasattr(result, 'dyn_obj_config') and result.dyn_obj_config is not None:
                    problem_info['dyn_obj_config'] = {}
                    for obj_name, obj_config in result.dyn_obj_config.items():
                        if 'keyframe_positions' in obj_config:
                            problem_info['dyn_obj_config'][obj_name] = {
                                'keyframe_positions': obj_config['keyframe_positions']
                            }

                # Add metrics if available
                if hasattr(result, 'metrics'):
                    problem_info['collision_intensity'] = extract_nested_metric(result, 'metrics.trajs_all.collision_intensity')
                    problem_info['fraction_valid'] = extract_nested_metric(result, 'metrics.trajs_all.fraction_valid')

                failed_list.append(problem_info)

        failed_problems[option_name] = failed_list

    # Print summary
    print("\n" + "="*80)
    print("FAILED PROBLEMS SUMMARY")
    print("="*80)
    for option_name, failed_list in failed_problems.items():
        print(f"\n{option_name}: {len(failed_list)} failed problems")
        if len(failed_list) > 0:
            print(f"  Failed indices: {[p['idx'] for p in failed_list[:10]]}" +
                  ("..." if len(failed_list) > 10 else ""))

    # Save to file if save_dir provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import json

        # Convert numpy arrays to lists for JSON
        failed_problems_json = {}
        for option_name, failed_list in failed_problems.items():
            failed_problems_json[option_name] = []
            for problem in failed_list:
                problem_json = {'idx': problem['idx']}
                if problem['q_pos_start'] is not None:
                    problem_json['q_pos_start'] = problem['q_pos_start'].tolist()
                if problem['q_pos_goal'] is not None:
                    problem_json['q_pos_goal'] = problem['q_pos_goal'].tolist()
                if 'dyn_obj_config' in problem:
                    problem_json['dyn_obj_config'] = {}
                    for obj_name, obj_data in problem['dyn_obj_config'].items():
                        problem_json['dyn_obj_config'][obj_name] = {
                            'keyframe_positions': obj_data['keyframe_positions'].tolist()
                        }
                if 'collision_intensity' in problem:
                    problem_json['collision_intensity'] = float(problem['collision_intensity']) if problem['collision_intensity'] is not None else None
                if 'fraction_valid' in problem:
                    problem_json['fraction_valid'] = float(problem['fraction_valid']) if problem['fraction_valid'] is not None else None

                failed_problems_json[option_name].append(problem_json)

        failed_file = os.path.join(save_dir, 'failed_problems.json')
        with open(failed_file, 'w') as f:
            json.dump(failed_problems_json, f, indent=2)
        print(f"\nSaved failed problems to: {failed_file}")

    return failed_problems


def find_hard_problems(all_results, option_l=None, k=10, save_dir=None):
    """
    Find hard problems based on top-k least fraction_valid and highest collision_intensity
    from trajs_all metrics. Returns the intersection of problem indices across all options.

    Args:
        all_results: Dict mapping experiment options to list of results
        option_l: List of option lists for converting indices to names
        k: Number of top hard problems to identify per metric
        save_dir: Directory to save hard problems report

    Returns:
        Dict mapping option names to list of hard problem indices and details
    """
    hard_problems = {}

    for option_tuple, results_list in all_results.items():
        # Get readable option name
        if option_l is not None:
            option_name = get_option_name_str(option_tuple, option_l)
        else:
            option_name = str(option_tuple)

        # Collect metrics for all problems
        problem_metrics = []
        for idx, result in enumerate(results_list):
            fraction_valid = extract_nested_metric(result, 'metrics.trajs_all.fraction_valid')
            collision_intensity = extract_nested_metric(result, 'metrics.trajs_all.collision_intensity')

            # Convert to float
            if fraction_valid is not None:
                if hasattr(fraction_valid, 'cpu'):
                    fraction_valid = float(fraction_valid.cpu().numpy())
                else:
                    fraction_valid = float(fraction_valid)
            else:
                fraction_valid = 1.0  # Default to max if not available

            if collision_intensity is not None:
                if hasattr(collision_intensity, 'cpu'):
                    collision_intensity = float(collision_intensity.cpu().numpy())
                else:
                    collision_intensity = float(collision_intensity)
            else:
                collision_intensity = 0.0  # Default to min if not available

            problem_metrics.append({
                'idx': idx,
                'fraction_valid': fraction_valid,
                'collision_intensity': collision_intensity,
                'result': result
            })

        # Sort by fraction_valid (ascending - least valid first)
        sorted_by_fraction_valid = sorted(problem_metrics, key=lambda x: x['fraction_valid'])
        top_k_least_valid = sorted_by_fraction_valid[:k]

        # Sort by collision_intensity (descending - highest collision first)
        sorted_by_collision = sorted(problem_metrics, key=lambda x: x['collision_intensity'], reverse=True)
        top_k_highest_collision = sorted_by_collision[:k]

        # Intersection of indices (problems that are hard by BOTH criteria)
        least_valid_indices = set([p['idx'] for p in top_k_least_valid])
        highest_collision_indices = set([p['idx'] for p in top_k_highest_collision])
        hard_indices = least_valid_indices.intersection(highest_collision_indices)

        # Build detailed info for hard problems
        hard_list = []
        for idx in sorted(hard_indices):
            result = results_list[idx]
            problem_info = {
                'idx': idx,
                'q_pos_start': to_numpy(result.q_pos_start) if hasattr(result, 'q_pos_start') else None,
                'q_pos_goal': to_numpy(result.q_pos_goal) if hasattr(result, 'q_pos_goal') else None,
                'fraction_valid': extract_nested_metric(result, 'metrics.trajs_all.fraction_valid'),
                'collision_intensity': extract_nested_metric(result, 'metrics.trajs_all.collision_intensity'),
                'success': extract_nested_metric(result, 'metrics.trajs_all.success'),
            }

            # Convert tensors to floats
            for key in ['fraction_valid', 'collision_intensity', 'success']:
                if problem_info[key] is not None:
                    if hasattr(problem_info[key], 'cpu'):
                        problem_info[key] = float(problem_info[key].cpu().numpy())
                    else:
                        problem_info[key] = float(problem_info[key])

            # Add dynamic obstacle info if available
            if hasattr(result, 'dyn_obj_config') and result.dyn_obj_config is not None:
                problem_info['dyn_obj_config'] = {}
                for obj_name, obj_config in result.dyn_obj_config.items():
                    if 'keyframe_positions' in obj_config:
                        problem_info['dyn_obj_config'][obj_name] = {
                            'keyframe_positions': obj_config['keyframe_positions']
                        }

            hard_list.append(problem_info)

        hard_problems[option_name] = hard_list

    # Print summary
    print("\n" + "="*80)
    print(f"HARD PROBLEMS SUMMARY (top-{k} per metric, intersection)")
    print("="*80)
    for option_name, hard_list in hard_problems.items():
        print(f"\n{option_name}: {len(hard_list)} hard problems (intersection of top-{k})")
        if len(hard_list) > 0:
            print(f"  Hard problem indices: {sorted([p['idx'] for p in hard_list])}")
            # Print some statistics
            fv_values = [p['fraction_valid'] for p in hard_list if p['fraction_valid'] is not None]
            ci_values = [p['collision_intensity'] for p in hard_list if p['collision_intensity'] is not None]
            if fv_values:
                print(f"  Fraction valid range: [{min(fv_values):.3f}, {max(fv_values):.3f}]")
            if ci_values:
                print(f"  Collision intensity range: [{min(ci_values):.3f}, {max(ci_values):.3f}]")
        else:
            print(f"  No problems meet both criteria (top-{k} least valid AND top-{k} highest collision)")

    # Save to file if save_dir provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import json

        # Convert numpy arrays to lists for JSON
        hard_problems_json = {}
        for option_name, hard_list in hard_problems.items():
            hard_problems_json[option_name] = []
            for problem in hard_list:
                problem_json = {
                    'idx': problem['idx'],
                    'fraction_valid': problem['fraction_valid'],
                    'collision_intensity': problem['collision_intensity'],
                    'success': problem['success'],
                }
                if problem['q_pos_start'] is not None:
                    problem_json['q_pos_start'] = problem['q_pos_start'].tolist()
                if problem['q_pos_goal'] is not None:
                    problem_json['q_pos_goal'] = problem['q_pos_goal'].tolist()
                if 'dyn_obj_config' in problem:
                    problem_json['dyn_obj_config'] = {}
                    for obj_name, obj_data in problem['dyn_obj_config'].items():
                        problem_json['dyn_obj_config'][obj_name] = {
                            'keyframe_positions': obj_data['keyframe_positions'].tolist()
                        }

                hard_problems_json[option_name].append(problem_json)

        hard_file = os.path.join(save_dir, 'hard_problems.json')
        with open(hard_file, 'w') as f:
            json.dump(hard_problems_json, f, indent=2)
        print(f"\nSaved hard problems to: {hard_file}")

    return hard_problems


def plot_metric_kde_comparison(aggregated_metrics, metric_paths_to_plot, save_dir=None, option_l=None):
    """
    Create KDE (Kernel Density Estimation) plots comparing distributions of metrics across options.
    Each metric gets its own subplot with overlaid KDE curves for each experimental option.

    Args:
        aggregated_metrics: Dict from aggregate_metrics() containing 'values' arrays
        metric_paths_to_plot: List of metric paths to plot (e.g., ['metrics.trajs_all.fraction_valid'])
        save_dir: Directory to save plots (if None, displays plots)
    """
    # Get option names and colors
    option_names = list(aggregated_metrics.keys())
    n_options = len(option_names)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_options, 10)))

    # Metrics that should be in [0, 1] range
    ratio_metrics = {
        'success', 'fraction_valid', 'n_trajectory_fraction',
        'success_no_joint_limits_vel_acc', 'fraction_valid_no_joint_limits_vel_acc',
        'n_trajectories_free_fraction'
    }

    # Filter metrics that exist in the aggregated data
    available_metrics = []
    for metric_path in metric_paths_to_plot:
        # Check if at least one option has this metric
        has_metric = any(
            metric_path in aggregated_metrics[opt]
            for opt in option_names
        )
        if has_metric:
            available_metrics.append(metric_path)
        else:
            print(f"Warning: Metric '{metric_path}' not found in any option, skipping")

    if len(available_metrics) == 0:
        print("No valid metrics to plot KDE for")
        return

    # Create subplots
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)
    fig.suptitle('Metric Distribution Comparison (KDE)', fontsize=16, fontweight='bold')

    for idx, metric_path in enumerate(available_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Plot KDE for each option
        for opt_idx, option_name in enumerate(option_names):
            if metric_path in aggregated_metrics[option_name]:
                stats = aggregated_metrics[option_name][metric_path]
                values = stats['values']
                if option_l is not None :
                    label_str = get_option_name_str(option_name, option_l)
                else :
                    label_str = f"{option_name}"
                if len(values) > 1:  # Need at least 2 points for KDE
                    # Plot KDE
                    sns.kdeplot(
                        data=values,
                        ax=ax,
                        color=colors[opt_idx],
                        linewidth=2.5,
                        label=label_str,
                        fill=True,
                        alpha=0.3
                    )

                    # Add vertical line for mean
                    ax.axvline(
                        stats['mean'],
                        color=colors[opt_idx],
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8
                    )

        # Formatting
        ax.set_xlabel('Value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(metric_path, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

        # Check if this metric should be in [0, 1] range
        metric_base = metric_path.split('.')[-1]  # Get the last part of the metric path
        if any(ratio_name in metric_base for ratio_name in ratio_metrics):
            ax.set_xlim(0, 1)

    # Remove empty subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'metric_kde_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved KDE comparison to {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    #exp_name = "launch_inference-experiments-test_2025-11-07_02-28-40"

    # exp_name = "launch_inference-experiments-test_2025-11-07_14-38-42" #failed idxs
    
    # exp_name = "launch_inference-experiments-test_2025-11-07_16-48-33" # margin scale 2 
    # exp_name = "launch_inference-experiments-test_2025-11-07_17-18-05" # margin scale 2 failed
    # exp_name = "launch_inference-experiments-test_2025-11-07_17-32-29" # margin scale 1.5
    exp_name = "launch_inference-experiments-test_2025-11-07_17-35-52" # margin scale 4 
    # exp_name = "launch_inference-experiments-test_2025-11-07_17-40-57" # margin scale 4 failed
    option_l = [["waypoints", "bspline"]] #bspline plot front

    # Load results
    print("Loading results...")
    all_results, option_l = load_results(exp_name, option_l)

    # Print sample result structure
    for k, v in all_results.items():
        option_name_str = get_option_name_str(k, option_l)
        print(f"\nOption {option_name_str}: {len(v)} samples")
        if len(v) > 0:
            print(f"  Sample result keys: {list(v[0].keys())}")
            if hasattr(v[0], 'dyn_obj_config') and v[0].dyn_obj_config is not None:
                print(f"  dyn_obj_config keys: {list(v[0].dyn_obj_config.keys())}")
            if hasattr(v[0], 'metrics') and v[0].metrics is not None:
                print(f"  metrics keys: {list(v[0].metrics.keys())}")
            if hasattr(v[0], 'isaacgym_statistics') and v[0].isaacgym_statistics is not None:
                print(f"  isaacgym keys: {list(v[0].isaacgym_statistics.keys())}")
            # if hasattr(v[0], 'trajs_all'):
            #     print(f"  trajs_all metrics: {list(v[0].trajs_all.keys())}")
            # if hasattr(v[0], 'trajs_valid') and v[0].trajs_valid:
            #     print(f"  trajs_valid metrics: {list(v[0].trajs_valid.keys())}")
            # if hasattr(v[0], 'trajs_best') and v[0].trajs_best:
            #     print(f"  trajs_best metrics: {list(v[0].trajs_best.keys())}")

    # Create output directory
    output_dir = os.path.join(LOG_DIR, exp_name, "analysis")

    # 1. Plot start/goal/problem distributions
    print("\nPlotting start/goal/problem distributions...")
    plot_start_goal_problems(all_results, option_l=option_l, save_dir=output_dir, show_trajectories=True)

    # 2. Aggregate metrics
    print("\nAggregating metrics...")

    # Define comprehensive metric paths based on compute_metrics structure
    metric_paths = [
        # Timing metrics
        't_inference_total',
        't_generator',
        't_guide',

        # trajs_all metrics
        'metrics.trajs_all.success',
        'metrics.trajs_all.success_no_joint_limits_vel_acc',
        'metrics.trajs_all.fraction_valid',
        'metrics.trajs_all.fraction_valid_no_joint_limits_vel_acc',
        'metrics.trajs_all.collision_intensity',
        'metrics.trajs_all.ee_pose_goal_error_position_norm_mean',
        'metrics.trajs_all.ee_pose_goal_error_position_norm_std',
        'metrics.trajs_all.ee_pose_goal_error_orientation_norm_mean',
        'metrics.trajs_all.ee_pose_goal_error_orientation_norm_std',

        # trajs_valid metrics
        'metrics.trajs_valid.ee_pose_goal_error_position_norm_mean',
        'metrics.trajs_valid.ee_pose_goal_error_position_norm_std',
        'metrics.trajs_valid.ee_pose_goal_error_orientation_norm_mean',
        'metrics.trajs_valid.ee_pose_goal_error_orientation_norm_std',
        'metrics.trajs_valid.path_length_mean',
        'metrics.trajs_valid.path_length_std',
        'metrics.trajs_valid.smoothness_mean',
        'metrics.trajs_valid.smoothness_std',
        'metrics.trajs_valid.diversity',

        # trajs_best metrics
        'metrics.trajs_best.ee_pose_goal_error_position_norm',
        'metrics.trajs_best.ee_pose_goal_error_orientation_norm',
        'metrics.trajs_best.path_length',
        'metrics.trajs_best.smoothness',

        # IsaacGym statistics (if available)
        'isaacgym_statistics.n_trajectories_collision',
        'isaacgym_statistics.n_trajectories_free',
        'isaacgym_statistics.n_trajectories_free_fraction',
    ]

    aggregated_metrics = aggregate_metrics(all_results, metric_paths)

    # 3. Print summary
    print_metric_summary(aggregated_metrics)

    # 4. Plot metric comparison
    print("\nPlotting metric comparison...")
    plot_metric_comparison(aggregated_metrics, save_dir=output_dir, option_l=option_l)

    # 5. Plot KDE comparison for selected metrics
    print("\nPlotting KDE comparison...")
    kde_metric_paths = [
        'metrics.trajs_all.fraction_valid',
        'metrics.trajs_all.collision_intensity',
        'metrics.trajs_valid.diversity',
        'metrics.trajs_valid.path_length_mean',
        'metrics.trajs_valid.smoothness_mean',
        'metrics.trajs_best.path_length',
        'metrics.trajs_best.smoothness',
        't_inference_total',
    ]
    plot_metric_kde_comparison(aggregated_metrics, kde_metric_paths, save_dir=output_dir, option_l=option_l)

    # 6. Find and save failed problems
    print("\nFinding failed problems...")
    failed_problems = find_failed_problems(all_results, option_l=option_l, save_dir=output_dir)

    # 7. Find and save hard problems
    print("\nFinding hard problems...")
    hard_problems = find_hard_problems(all_results, option_l=option_l, k=10, save_dir=output_dir)

    # 8. Save aggregated metrics to file
    import json
    metrics_file = os.path.join(output_dir, 'aggregated_metrics.json')
    # Convert numpy arrays to lists for JSON serialization
    metrics_for_json = {}
    for option_name, stats in aggregated_metrics.items():
        option_name_str = get_option_name_str(option_name, option_l) if isinstance(option_name, tuple) else str(option_name)
        metrics_for_json[option_name_str] = {}
        for metric_name, metric_stats in stats.items():
            metrics_for_json[option_name_str][metric_name] = {
                'mean': float(metric_stats['mean']),
                'std': float(metric_stats['std']),
                'min': float(metric_stats['min']),
                'max': float(metric_stats['max']),
                'n_samples': int(metric_stats['n_samples']),
            }

    with open(metrics_file, 'w') as f:
        json.dump(metrics_for_json, f, indent=2)
    print(f"\nSaved aggregated metrics to: {metrics_file}")

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
