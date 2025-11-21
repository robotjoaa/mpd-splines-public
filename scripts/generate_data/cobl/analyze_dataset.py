import torch
import numpy as np

# Load training data
train_path = '../../../data_trajectories/EnvCoblEmpty2D-RobotPointMass2DBig/train80.pt'
print("=" * 80)
print("ANALYZING THRESHOLD: 14m (1.4 scaled)")
print("=" * 80)

train_data = torch.load(train_path, map_location='cpu')
positions = train_data[:, :2, :].numpy()

# Scale by 1/10
SCALE_FACTOR = 10.0
positions = positions / SCALE_FACTOR

# Get start and goal positions
start_positions = positions[:, :, 0]  # [N, 2]
goal_positions = positions[:, :, -1]  # [N, 2]

# Calculate distances
distances = np.linalg.norm(goal_positions - start_positions, axis=1)

# Check threshold
threshold_scaled = 0.4
threshold_original = 14.0

#a = np.where(np.logical_and(distances >= 0.4, distances <= 1.4))
a = np.where(distances > 0.4)
print(len(a[0]))
print(a[0][:10])


# print(f"\nThreshold: {threshold_scaled} (scaled) = {threshold_original}m (original)")
# print(f"Total trajectories: {len(distances):,}")

# # Count trajectories above threshold
# above_threshold = distances >= threshold_scaled
# n_above = above_threshold.sum()
# n_below = len(distances) - n_above

# print(f"\nTrajectories >= {threshold_scaled} (>= {threshold_original}m): {n_above:,} ({100*n_above/len(distances):.2f}%)")
# print(f"Trajectories < {threshold_scaled} (< {threshold_original}m): {n_below:,} ({100*n_below/len(distances):.2f}%)")

# # Statistics for trajectories above threshold
# if n_above > 0:
#     distances_above = distances[above_threshold]
#     print(f"\nStatistics for trajectories >= {threshold_scaled}:")
#     print(f"  Count: {len(distances_above):,}")
#     print(f"  Min: {distances_above.min():.4f} ({distances_above.min()*SCALE_FACTOR:.2f}m)")
#     print(f"  Max: {distances_above.max():.4f} ({distances_above.max()*SCALE_FACTOR:.2f}m)")
#     print(f"  Mean: {distances_above.mean():.4f} ({distances_above.mean()*SCALE_FACTOR:.2f}m)")
#     print(f"  Median: {np.median(distances_above):.4f} ({np.median(distances_above)*SCALE_FACTOR:.2f}m)")

# # Statistics for trajectories below threshold (that would be filtered)
# if n_below > 0:
#     distances_below = distances[~above_threshold]
#     print(f"\nStatistics for trajectories < {threshold_scaled} (WOULD BE FILTERED):")
#     print(f"  Count: {len(distances_below):,}")
#     print(f"  Min: {distances_below.min():.4f} ({distances_below.min()*SCALE_FACTOR:.2f}m)")
#     print(f"  Max: {distances_below.max():.4f} ({distances_below.max()*SCALE_FACTOR:.2f}m)")
#     print(f"  Mean: {distances_below.mean():.4f} ({distances_below.mean()*SCALE_FACTOR:.2f}m)")
#     print(f"  Median: {np.median(distances_below):.4f} ({np.median(distances_below)*SCALE_FACTOR:.2f}m)")

# # Compare with other common thresholds
# print(f"\n{'='*80}")
# print("COMPARISON WITH OTHER THRESHOLDS")
# print(f"{'='*80}")

# thresholds = [
#     (0.05, 0.5),
#     (0.10, 1.0),
#     (0.20, 2.0),
#     (0.50, 5.0),
#     (1.00, 10.0),
#     (1.40, 14.0),
# ]

# for thresh_scaled, thresh_orig in thresholds:
#     n_keep = (distances >= thresh_scaled).sum()
#     n_remove = len(distances) - n_keep
#     print(f"Threshold >= {thresh_scaled:4.2f} ({thresh_orig:5.1f}m): Keep {n_keep:,} ({100*n_keep/len(distances):5.2f}%), Remove {n_remove:,} ({100*n_remove/len(distances):5.2f}%)")

# print(f"\n{'='*80}")
# print("COMPLETE")
# print(f"{'='*80}")

