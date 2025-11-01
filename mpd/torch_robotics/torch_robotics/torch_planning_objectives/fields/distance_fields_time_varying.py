"""
Time-varying distance fields for moving obstacles.

This module extends the distance field classes to handle time-varying obstacles
by incorporating temporal queries into the SDF computation.
"""

import torch
import einops
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionObjectBase,
    EmbodimentDistanceFieldBase
)
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class CollisionObjectDistanceFieldTimeVarying(CollisionObjectBase):
    """
    Collision distance field for time-varying objects.

    This class handles collision checking with moving obstacles by querying
    a time-varying SDF grid (GridMapSDFTimeVarying).
    """

    def __init__(
        self,
        *args,
        df_time_varying_obj_fn=None,
        parametric_trajectory=None,
        **kwargs
    ):
        """
        Args:
            df_time_varying_obj_fn: Function that returns GridMapSDFTimeVarying instance
            parametric_trajectory: ParametricTrajectory instance to get time values
            *args, **kwargs: Passed to parent CollisionObjectBase
        """
        super().__init__(*args, **kwargs)
        self.df_time_varying_obj_fn = df_time_varying_obj_fn
        self.parametric_trajectory = parametric_trajectory

    def object_signed_distances(self, link_pos, get_gradient=False, timesteps=None, **kwargs):
        """
        Compute signed distances to time-varying objects.

        Args:
            link_pos: Link positions, shape (batch, horizon, num_links, dim) or (batch*horizon, num_links, dim)
            get_gradient: Whether to return gradients
            timesteps: Time values corresponding to trajectory points, shape (horizon,) or (batch, horizon)
                      If None, will get from parametric_trajectory
            **kwargs: Additional arguments

        Returns:
            dfs_th: Signed distances, shape (batch, horizon, num_sdfs, num_links)
            dfs_gradient: (if get_gradient) Gradients, shape (batch, horizon, num_sdfs, num_links, dim)
        """
        if self.df_time_varying_obj_fn is None:
            return torch.inf

        df_time_varying = self.df_time_varying_obj_fn()

        # Get timesteps if not provided
        if timesteps is None:
            if self.parametric_trajectory is None:
                raise ValueError("Either timesteps or parametric_trajectory must be provided")
            timesteps = self.parametric_trajectory.get_timesteps()  # Shape: (horizon,)

        # Determine input shape
        original_shape = link_pos.shape
        if len(original_shape) == 4:
            # Shape: (batch, horizon, num_links, dim)
            batch_size, horizon, num_links, dim = original_shape
            link_pos_flat = einops.rearrange(link_pos, "b h l d -> (b h) l d")

            # Expand timesteps to match batch dimension
            if timesteps.ndim == 1:
                # timesteps shape: (horizon,) -> (batch, horizon) -> (batch*horizon,)
                timesteps_expanded = timesteps.unsqueeze(0).expand(batch_size, horizon)
                timesteps_flat = einops.rearrange(timesteps_expanded, "b h -> (b h)")
            else:
                # timesteps shape: (batch, horizon) -> (batch*horizon,)
                timesteps_flat = einops.rearrange(timesteps, "b h -> (b h)")
        elif len(original_shape) == 3:
            # Shape: (batch*horizon, num_links, dim)
            batch_horizon, num_links, dim = original_shape
            link_pos_flat = link_pos
            timesteps_flat = timesteps  # Assume already flat

            # Try to infer batch and horizon
            if self.parametric_trajectory is not None:
                horizon = len(self.parametric_trajectory.get_timesteps())
                batch_size = batch_horizon // horizon
            else:
                # Cannot reshape without knowing structure
                batch_size = 1
                horizon = batch_horizon
        else:
            raise ValueError(f"Unexpected link_pos shape: {original_shape}")

        # Flatten for vectorized query
        # link_pos_flat: (batch*horizon, num_links, dim)
        # timesteps_flat: (batch*horizon,)
        bh = link_pos_flat.shape[0]

        # Expand timesteps to match link dimension
        # timesteps_for_links: (batch*horizon, num_links)
        timesteps_for_links = timesteps_flat.unsqueeze(-1).expand(bh, num_links)

        # Reshape for query
        # (batch*horizon, num_links, dim) -> (batch*horizon*num_links, dim)
        link_pos_query = link_pos_flat.reshape(-1, dim)
        timesteps_query = timesteps_for_links.reshape(-1)

        # Query time-varying SDF
        if get_gradient:
            sdf_vals, sdf_gradient = df_time_varying.compute_signed_distance(
                link_pos_query, timesteps_query, get_gradient=True
            )
            # sdf_vals: (batch*horizon*num_links,)
            # sdf_gradient: (batch*horizon*num_links, dim)

            # Reshape back
            sdf_vals = sdf_vals.reshape(bh, num_links)
            sdf_gradient = sdf_gradient.reshape(bh, num_links, dim)

            # Add object dimension (assuming single time-varying object field)
            sdf_vals = sdf_vals.unsqueeze(-2)  # (bh, 1, num_links)
            sdf_gradient = sdf_gradient.unsqueeze(-3)  # (bh, 1, num_links, dim)

            # Reshape to (batch, horizon, num_sdfs, num_links) format
            if len(original_shape) == 4:
                sdf_vals = einops.rearrange(sdf_vals, "(b h) s l -> b h s l", b=batch_size, h=horizon)
                sdf_gradient = einops.rearrange(sdf_gradient, "(b h) s l d -> b h s l d", b=batch_size, h=horizon)

            return sdf_vals, sdf_gradient
        else:
            sdf_vals = df_time_varying.compute_signed_distance(
                link_pos_query, timesteps_query, get_gradient=False
            )
            # sdf_vals: (batch*horizon*num_links,)

            # Reshape back
            sdf_vals = sdf_vals.reshape(bh, num_links)

            # Add object dimension
            sdf_vals = sdf_vals.unsqueeze(-2)  # (bh, 1, num_links)

            # Reshape to (batch, horizon, num_sdfs, num_links) format
            if len(original_shape) == 4:
                sdf_vals = einops.rearrange(sdf_vals, "(b h) s l -> b h s l", b=batch_size, h=horizon)

            return sdf_vals

    def compute_distance_field_cost_and_gradient(self, link_pos, timesteps=None, **kwargs):
        """
        Compute collision cost and gradient for time-varying obstacles.

        Args:
            link_pos: Link positions, shape (batch, horizon, num_links, dim) or (batch*horizon, num_links, dim)
            timesteps: Time values, shape (horizon,) or (batch, horizon)
            **kwargs: Additional arguments

        Returns:
            cost: Collision cost, shape depends on input
            gradient: Gradient wrt link positions, shape matches link_pos
        """
        margin = self.collision_margins + self.cutoff_margin

        # Get SDF values and gradients
        sdf_vals, sdf_gradient = self.object_signed_distances(
            link_pos, get_gradient=True, timesteps=timesteps, **kwargs
        )

        # Apply margin and clamp
        margin_minus_sdf = -(sdf_vals - margin)
        if self.clamp_sdf:
            margin_minus_sdf_clamped = torch.relu(margin_minus_sdf)
        else:
            margin_minus_sdf_clamped = margin_minus_sdf

        # Handle multiple objects case
        if margin_minus_sdf_clamped.ndim >= 3:
            if margin_minus_sdf_clamped.shape[-2] == 1:
                # Single object
                margin_minus_sdf_clamped = margin_minus_sdf_clamped.squeeze(-2)
                sdf_gradient = sdf_gradient.squeeze(-3)
            else:
                # Multiple objects - take max
                margin_minus_sdf_clamped, idxs_max = margin_minus_sdf_clamped.max(-2)
                # Gather corresponding gradients
                sdf_gradient = sdf_gradient.gather(
                    -3, idxs_max.unsqueeze(-3).unsqueeze(-1).expand(*sdf_gradient.shape[:-3], 1, *sdf_gradient.shape[-2:])
                ).squeeze(-3)

        # Set gradient to zero where not in collision
        idxs = torch.argwhere(margin_minus_sdf_clamped <= 0)
        if idxs.numel() > 0:
            if sdf_gradient.ndim == 3:  # (bh, num_links, dim)
                sdf_gradient[idxs[:, 0], idxs[:, 1], :] = 0.0
            elif sdf_gradient.ndim == 4:  # (b, h, num_links, dim)
                sdf_gradient[idxs[:, 0], idxs[:, 1], idxs[:, 2], :] = 0.0

        # Gradient sign (cost increases as we move into obstacle)
        sdf_gradient = -1.0 * sdf_gradient

        return margin_minus_sdf_clamped, sdf_gradient


class CombinedCollisionDistanceField(EmbodimentDistanceFieldBase):
    """
    Combined collision distance field handling both static and time-varying obstacles.

    This class combines:
    - Static obstacles (via CollisionObjectDistanceField)
    - Time-varying obstacles (via CollisionObjectDistanceFieldTimeVarying)
    """

    def __init__(
        self,
        robot,
        df_static_obj_list_fn=None,
        df_time_varying_obj_fn=None,
        parametric_trajectory=None,
        **kwargs
    ):
        """
        Args:
            robot: Robot instance
            df_static_obj_list_fn: Function returning list of static SDF objects
            df_time_varying_obj_fn: Function returning GridMapSDFTimeVarying
            parametric_trajectory: ParametricTrajectory instance
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(robot=robot, **kwargs)

        from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionObjectDistanceField

        self.df_static = None
        if df_static_obj_list_fn is not None:
            self.df_static = CollisionObjectDistanceField(
                robot,
                df_obj_list_fn=df_static_obj_list_fn,
                link_margins_for_object_collision_checking_tensor=robot.link_collision_spheres_radii,
                cutoff_margin=self.cutoff_margin,
                clamp_sdf=self.clamp_sdf,
                tensor_args=self.tensor_args
            )

        self.df_time_varying = None
        if df_time_varying_obj_fn is not None:
            self.df_time_varying = CollisionObjectDistanceFieldTimeVarying(
                robot,
                df_time_varying_obj_fn=df_time_varying_obj_fn,
                parametric_trajectory=parametric_trajectory,
                link_margins_for_object_collision_checking_tensor=robot.link_collision_spheres_radii,
                cutoff_margin=self.cutoff_margin,
                clamp_sdf=self.clamp_sdf,
                tensor_args=self.tensor_args
            )

    def compute_embodiment_signed_distances(self, q_pos, link_pos, timesteps=None, **kwargs):
        """
        Compute signed distances to both static and time-varying obstacles.

        Args:
            q_pos: Joint positions
            link_pos: Link positions
            timesteps: Time values for time-varying obstacles

        Returns:
            Combined signed distances
        """
        dfs = []

        # Static obstacles
        if self.df_static is not None:
            sdf_static = self.df_static.object_signed_distances(link_pos, get_gradient=False, **kwargs)
            dfs.append(sdf_static)

        # Time-varying obstacles
        if self.df_time_varying is not None:
            sdf_time_varying = self.df_time_varying.object_signed_distances(
                link_pos, get_gradient=False, timesteps=timesteps, **kwargs
            )
            dfs.append(sdf_time_varying)

        if len(dfs) == 0:
            return torch.inf

        # Combine (take minimum across all objects)
        return torch.cat(dfs, dim=-2)  # Concatenate along object dimension

    def compute_embodiment_collision(self, q_pos, link_pos, timesteps=None, **kwargs):
        """
        Check collision with both static and time-varying obstacles.

        Args:
            q_pos: Joint positions
            link_pos: Link positions
            timesteps: Time values

        Returns:
            Boolean collision indicator
        """
        cutoff_margin = kwargs.get("margin", self.cutoff_margin)
        margin = self.collision_margins + cutoff_margin

        signed_distances = self.compute_embodiment_signed_distances(
            q_pos, link_pos, timesteps=timesteps, **kwargs
        )

        collisions = signed_distances <= margin
        any_collision = torch.any(torch.any(collisions, dim=-1), dim=-1)

        return any_collision

    def compute_embodiment_rbf_distances(self, *args, **kwargs):
        """RBF distances not implemented for combined field."""
        raise NotImplementedError("RBF distances not implemented for combined collision field")

    def compute_distance_field_cost_and_gradient(self, link_pos, timesteps=None, **kwargs):
        """
        Compute combined cost and gradient from static and time-varying obstacles.

        Args:
            link_pos: Link positions
            timesteps: Time values

        Returns:
            Combined cost and gradient
        """
        cost_total = 0.0
        gradient_total = 0.0

        # Static obstacles
        if self.df_static is not None:
            cost_static, grad_static = self.df_static.compute_distance_field_cost_and_gradient(
                link_pos, **kwargs
            )
            cost_total = cost_total + cost_static
            gradient_total = gradient_total + grad_static

        # Time-varying obstacles
        if self.df_time_varying is not None:
            cost_tv, grad_tv = self.df_time_varying.compute_distance_field_cost_and_gradient(
                link_pos, timesteps=timesteps, **kwargs
            )
            cost_total = cost_total + cost_tv
            gradient_total = gradient_total + grad_tv

        return cost_total, gradient_total


if __name__ == "__main__":
    print("Time-varying distance fields module")
    print("This module provides distance field classes for moving obstacles")
    print("\nMain classes:")
    print("  - CollisionObjectDistanceFieldTimeVarying: Handles time-varying obstacles")
    print("  - CombinedCollisionDistanceField: Combines static and time-varying obstacles")
    print("\nUsage: See integration example in the test file")
