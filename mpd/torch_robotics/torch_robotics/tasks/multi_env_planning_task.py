"""
Multi-Environment Planning Task

Extension of PlanningTask that supports multiple environment configurations.
Compatible with pb_diff_envs WallGroupList pattern.

Author: Claude Code
Date: 2025-10-28
"""

from functools import partial

import torch
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionObjectDistanceField,
)
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class MultiEnvPlanningTask(PlanningTask):
    """
    Planning task that can handle multiple environment configurations.

    Compatible with pb_diff_envs WallGroupList pattern where multiple
    environment configurations are generated (e.g., 5000 random mazes).

    Features:
    - Holds multiple ObjectField configurations
    - Dynamically switches active environment via set_env_idx()
    - Maps task_id to env_idx for dataset compatibility
    - Maintains single robot instance (shared across environments)

    Example:
        >>> wall_group_list = RandRectangleGroup(num_groups=100, ...)
        >>> planning_task = MultiEnvPlanningTask(
        ...     env_base_class=EnvMaze2DBase,
        ...     wall_group_list=wall_group_list,
        ...     robot=robot,
        ...     parametric_trajectory=parametric_trajectory,
        ... )
        >>> planning_task.set_env_idx(0)  # Use first maze
        >>> planning_task.set_env_idx(50)  # Use 51st maze
    """

    def __init__(
        self,
        env_base_class,
        wall_group_list,
        robot,
        parametric_trajectory,
        map_task_id_to_env_idx=None,
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs
    ):
        """
        Initialize multi-environment planning task.

        Args:
            env_base_class: Environment class to instantiate (e.g., EnvMaze2DBase)
            wall_group_list: RandRectangleGroup or similar with multiple configs
                Must have:
                - num_groups: Number of environment configurations
                - object_fields: List of ObjectField instances
            robot: Robot instance (shared across all environments)
            parametric_trajectory: Trajectory representation (B-spline or waypoints)
            map_task_id_to_env_idx: Dict mapping task_id → env_idx (optional)
            tensor_args: Tensor device and dtype configuration
            **kwargs: Additional arguments passed to PlanningTask
        """
        # Store multiple environment configurations
        self.wall_group_list = wall_group_list
        self.num_envs = wall_group_list.num_groups
        self.object_fields_list = wall_group_list.object_fields  # List of ObjectFields

        # Create base environment instance (will be updated dynamically)
        self.env_base_class = env_base_class
        env = env_base_class(tensor_args=tensor_args, **kwargs)

        # Initialize parent with first environment
        super().__init__(
            env=env,
            robot=robot,
            parametric_trajectory=parametric_trajectory,
            tensor_args=tensor_args,
            **kwargs
        )

        # Task-to-environment mapping
        self.map_task_id_to_env_idx = map_task_id_to_env_idx or {}
        self.current_env_idx = None

        # Store obstacle cutoff margin for rebuilding collision fields
        self.obstacle_cutoff_margin = kwargs.get(
            "obstacle_cutoff_margin",
            kwargs.get("min_distance_robot_env", 0.01)
        )

        # Set first environment as active
        self.set_env_idx(0)

    def set_env_idx(self, env_idx):
        """
        Switch to a different environment configuration.

        Updates the environment's obstacle list to the specified configuration.
        This changes which obstacles are used for collision checking.

        Args:
            env_idx: Index of environment configuration (0 to num_envs-1)

        Raises:
            AssertionError: If env_idx is out of range
        """
        assert 0 <= env_idx < self.num_envs, \
            f"env_idx {env_idx} out of range [0, {self.num_envs})"

        # Skip if already set to this environment
        if self.current_env_idx == env_idx:
            return

        self.current_env_idx = env_idx

        # Update environment's object field to the selected configuration
        self.env.obj_fixed_list = [self.object_fields_list[env_idx]]

        # Update combined object list
        self.env.update_obj_all_list()

        # Rebuild collision field with new obstacles
        self._rebuild_collision_field()

    def set_env_for_task(self, task_id):
        """
        Switch to the environment corresponding to a task_id.

        Uses the task_id → env_idx mapping from the dataset.

        Args:
            task_id: Task identifier from dataset

        Raises:
            ValueError: If task_id not found in mapping
        """
        if task_id in self.map_task_id_to_env_idx:
            env_idx = self.map_task_id_to_env_idx[task_id]
            self.set_env_idx(env_idx)
        else:
            raise ValueError(
                f"task_id {task_id} not found in map_task_id_to_env_idx. "
                f"Available task_ids: {list(self.map_task_id_to_env_idx.keys())[:10]}..."
            )

    def _rebuild_collision_field(self):
        """
        Rebuild the collision distance field after changing environments.

        This is necessary because CollisionObjectDistanceField caches
        references to the environment's object list.
        """
        # Recreate collision field for fixed objects
        self.df_collision_objects = CollisionObjectDistanceField(
            self.robot,
            df_obj_list_fn=self.env.get_df_obj_list,
            link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
            cutoff_margin=self.obstacle_cutoff_margin,
            tensor_args=self.tensor_args,
        )

        # Recreate collision field for extra objects if they exist
        if self.env.obj_extra_list is not None:
            self.df_collision_extra_objects = CollisionObjectDistanceField(
                self.robot,
                df_obj_list_fn=partial(self.env.get_df_obj_list, return_extra_objects_only=True),
                link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
                cutoff_margin=self.obstacle_cutoff_margin,
                tensor_args=self.tensor_args,
            )
            self._collision_fields_extra_objects = [self.df_collision_extra_objects]
        else:
            self._collision_fields_extra_objects = []

        # Update collision fields list
        self._collision_fields = [
            self.df_collision_self,
            self.df_collision_objects,
            self.df_collision_ws_boundaries,
        ]

    def get_env_idx_for_task(self, task_id):
        """
        Get environment index for a task_id.

        Args:
            task_id: Task identifier

        Returns:
            env_idx (int) or None if task_id not in mapping
        """
        return self.map_task_id_to_env_idx.get(task_id, None)

    def get_all_env_indices(self):
        """
        Get list of all environment indices.

        Returns:
            List of environment indices [0, 1, ..., num_envs-1]
        """
        return list(range(self.num_envs))

    def get_current_env_idx(self):
        """
        Get currently active environment index.

        Returns:
            Current env_idx
        """
        return self.current_env_idx

    def update_task_id_mapping(self, map_task_id_to_env_idx):
        """
        Update the task_id → env_idx mapping.

        Useful when loading dataset that provides this mapping.

        Args:
            map_task_id_to_env_idx: Dict mapping task_id → env_idx
        """
        self.map_task_id_to_env_idx = map_task_id_to_env_idx

    def __repr__(self):
        return (
            f"MultiEnvPlanningTask(\n"
            f"  num_envs={self.num_envs},\n"
            f"  current_env_idx={self.current_env_idx},\n"
            f"  robot={self.robot.__class__.__name__},\n"
            f"  env_base_class={self.env_base_class.__name__},\n"
            f"  parametric_trajectory={self.parametric_trajectory.__class__.__name__},\n"
            f"  num_task_mappings={len(self.map_task_id_to_env_idx)}\n"
            f")"
        )


class DynamicEnvPlanningTask(PlanningTask):
    """
    Alternative approach: Planning task with dynamically configurable environment.

    Instead of holding multiple environments, this allows the environment
    to be reconfigured on-the-fly by changing obstacle lists.

    Use this if you want more fine-grained control over environment updates
    or if environments differ in more than just obstacle configurations.

    Example:
        >>> planning_task = DynamicEnvPlanningTask(env, robot, ...)
        >>> # Update obstacles dynamically
        >>> new_obstacles = [ObjectField(...), ObjectField(...)]
        >>> planning_task.set_obstacles(new_obstacles)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacle_cutoff_margin = kwargs.get(
            "obstacle_cutoff_margin",
            kwargs.get("min_distance_robot_env", 0.01)
        )

    def set_obstacles(self, obj_fixed_list=None, obj_extra_list=None):
        """
        Update environment obstacles dynamically.

        Args:
            obj_fixed_list: List of ObjectField instances for fixed obstacles
            obj_extra_list: List of ObjectField instances for extra obstacles
        """
        if obj_fixed_list is not None:
            self.env.obj_fixed_list = obj_fixed_list

        if obj_extra_list is not None:
            self.env.obj_extra_list = obj_extra_list

        # Update combined list
        self.env.update_obj_all_list()

        # Rebuild collision fields
        self._rebuild_collision_field()

    def _rebuild_collision_field(self):
        """Rebuild collision distance fields after obstacle update."""
        # Same as MultiEnvPlanningTask
        self.df_collision_objects = CollisionObjectDistanceField(
            self.robot,
            df_obj_list_fn=self.env.get_df_obj_list,
            link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
            cutoff_margin=self.obstacle_cutoff_margin,
            tensor_args=self.tensor_args,
        )

        if self.env.obj_extra_list is not None:
            self.df_collision_extra_objects = CollisionObjectDistanceField(
                self.robot,
                df_obj_list_fn=partial(self.env.get_df_obj_list, return_extra_objects_only=True),
                link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
                cutoff_margin=self.obstacle_cutoff_margin,
                tensor_args=self.tensor_args,
            )
            self._collision_fields_extra_objects = [self.df_collision_extra_objects]
        else:
            self._collision_fields_extra_objects = []

        self._collision_fields = [
            self.df_collision_self,
            self.df_collision_objects,
            self.df_collision_ws_boundaries,
        ]
