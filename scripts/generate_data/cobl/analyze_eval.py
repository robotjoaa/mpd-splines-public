import pickle
import numpy as np

# Load evaluation dataset
eval_path = 'data_trajectories/EnvCoblEmpty2D-RobotPointMass2DBig/eval_dataset.pkl'

print("=" * 80)
print("ANALYZING OBSTACLE TRAJECTORY RANGES IN EVAL DATASET")
print("=" * 80)

with open(eval_path, 'rb') as f:
    eval_data = pickle.load(f)

print(f"\nDataset info:")
print(f"  Number of scenarios: {eval_data['n_scenarios']}")
print(f"  Scale factor: {eval_data['scale_factor']}")
print(f"  Obstacle radius: {eval_data['obstacle_radius']}")

obstacle_trajectories = eval_data['obstacle_trajectories']

# Collect all obstacle positions
all_x = []
all_y = []

for scenario_idx, obs_list in enumerate(obstacle_trajectories):
    for obs_traj in obs_list:
        # obs_traj: [T, 6] where [:, 0:2] are [x, y] positions
        x_positions = obs_traj[:, 0]
        y_positions = obs_traj[:, 1]
        
        all_x.extend(x_positions)
        all_y.extend(y_positions)

all_x = np.array(all_x)
all_y = np.array(all_y)

# Calculate ranges
x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()
x_range = x_max - x_min
y_range = y_max - y_min

print(f"\n{'='*80}")
print("OBSTACLE TRAJECTORY POSITION RANGES (scaled)")
print(f"{'='*80}")
print(f"  Total obstacle waypoints: {len(all_x):,}")
print(f"\n  X range:")
print(f"    Min: {x_min:.4f}")
print(f"    Max: {x_max:.4f}")
print(f"    Range: {x_range:.4f}")
print(f"\n  Y range:")
print(f"    Min: {y_min:.4f}")
print(f"    Max: {y_max:.4f}")
print(f"    Range: {y_range:.4f}")

# Original scale (multiply by scale_factor)
scale_factor = eval_data['scale_factor']
print(f"\n{'='*80}")
print(f"OBSTACLE TRAJECTORY POSITION RANGES (original, meters)")
print(f"{'='*80}")
print(f"\n  X range:")
print(f"    Min: {x_min * scale_factor:.2f}m")
print(f"    Max: {x_max * scale_factor:.2f}m")
print(f"    Range: {x_range * scale_factor:.2f}m")
print(f"\n  Y range:")
print(f"    Min: {y_min * scale_factor:.2f}m")
print(f"    Max: {y_max * scale_factor:.2f}m")
print(f"    Range: {y_range * scale_factor:.2f}m")

# Compare with ego trajectory ranges
ego_trajectories = eval_data['ego_trajectories']  # [300, 80, 2]
ego_x = ego_trajectories[:, :, 0].flatten()
ego_y = ego_trajectories[:, :, 1].flatten()

ego_x_min, ego_x_max = ego_x.min(), ego_x.max()
ego_y_min, ego_y_max = ego_y.min(), ego_y.max()

print(f"\n{'='*80}")
print("EGO TRAJECTORY POSITION RANGES (for comparison, scaled)")
print(f"{'='*80}")
print(f"\n  X range: [{ego_x_min:.4f}, {ego_x_max:.4f}] (width: {ego_x_max - ego_x_min:.4f})")
print(f"  Y range: [{ego_y_min:.4f}, {ego_y_max:.4f}] (height: {ego_y_max - ego_y_min:.4f})")

# Combined ranges (obstacles + ego)
combined_x_min = min(x_min, ego_x_min)
combined_x_max = max(x_max, ego_x_max)
combined_y_min = min(y_min, ego_y_min)
combined_y_max = max(y_max, ego_y_max)

print(f"\n{'='*80}")
print("COMBINED RANGES (obstacles + ego, scaled)")
print(f"{'='*80}")
print(f"\n  X range: [{combined_x_min:.4f}, {combined_x_max:.4f}] (width: {combined_x_max - combined_x_min:.4f})")
print(f"  Y range: [{combined_y_min:.4f}, {combined_y_max:.4f}] (height: {combined_y_max - combined_y_min:.4f})")

# Suggested workspace limits with margin
margin = 0.1  # 10% margin on each side
x_margin = (combined_x_max - combined_x_min) * margin
y_margin = (combined_y_max - combined_y_min) * margin

suggested_x_min = combined_x_min - x_margin
suggested_x_max = combined_x_max + x_margin
suggested_y_min = combined_y_min - y_margin
suggested_y_max = combined_y_max + y_margin

print(f"\n{'='*80}")
print(f"SUGGESTED WORKSPACE LIMITS (with {margin*100:.0f}% margin)")
print(f"{'='*80}")
print(f"  X: [{suggested_x_min:.3f}, {suggested_x_max:.3f}]")
print(f"  Y: [{suggested_y_min:.3f}, {suggested_y_max:.3f}]")
print(f"\n  Rounded (convenient):")
print(f"  X: [{np.floor(suggested_x_min*10)/10:.1f}, {np.ceil(suggested_x_max*10)/10:.1f}]")
print(f"  Y: [{np.floor(suggested_y_min*10)/10:.1f}, {np.ceil(suggested_y_max*10)/10:.1f}]")

print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}")
