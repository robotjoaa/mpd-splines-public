import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Optional
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.dynamic_extension.moving_primitives import MovingObjectField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch


'''
    Migrated from pb_diff
'''

class AbstractTrajectory(ABC):
    @abstractmethod
    def get_spec(self, t):
        raise NotImplementedError  
        
    @abstractmethod
    def set_spec(self, obstacle, t):
        raise NotImplementedError

# class WaypointDiscreteTrajectory(AbstractTrajectory):
#     '''
#     following the waypoints moving in discrete time
#     '''
    
#     def __init__(self, waypoints):
#         self.waypoints = waypoints

#     def get_spec(self, t):
#         assert isinstance(t, int)
#         if t != -1:
#             assert 0<=t<=(len(self.waypoints)-1)
#         return self.waypoints[t]
        
#     def set_spec(self, obstacle, spec):
#         obstacle.set_config(spec)
        
        
class WaypointLinearTrajectory(AbstractTrajectory):
    '''
    following the waypoints moving in continuous time
    the motion is linear between adjacent timesteps
    '''
    
    def __init__(self, waypoints, noise_config={}):
        assert type(waypoints) == np.ndarray or type(waypoints) == list
        self.waypoints = waypoints
        self.noisy = len(noise_config) > 0
        self.noise_config = noise_config

    def get_spec(self, t):
        '''impl abstract method
        t (float or int): should be [0, len(waypoints)]
        do linear interp to get traj
        return: 
        np
        '''
        if t == -1 or t >= len(self.waypoints)-1:
            return self.waypoints[-1]
        # if t != -1:
        #     assert 0<=t<=(len(self.waypoints)-1)
        t_prev, t_next = int(np.floor(t)), int(np.ceil(t))
        # print('t', t, t_prev, t_next)
        spec_prev, spec_next = self.waypoints[t_prev], self.waypoints[t_next]
        spec_interp = spec_prev + (spec_next-spec_prev)*(t-t_prev)
        if self.noisy and t > 0:
            spec_interp = spec_interp + np.random.randn(*spec_interp.shape) * self.noise_config['std']

        return spec_interp

    def set_spec(self, obj_field, spec):
        '''impl abstract method
        place the obstace in new/next place one by one
        obstacle ()
        '''

        assert type(obj_field) in [MovingObjectField]

        # obstacle.set_config(spec)       
        obj_field.

        
# class WaypointProportionTrajectory(AbstractTrajectory):
#     '''
#     following the waypoints moving in continuous time
#     the motion is linear between adjacent timesteps
#     '''
    
#     def __init__(self, waypoints, noise_config={}):
#         self.waypoints = waypoints
#         self.noisy = len(noise_config) > 0
#         self.noise_config = noise_config

#     def get_spec(self, t):
#         '''impl abstract method
#         t (float or int): should be [0, len(waypoints)]
#         do linear interp to get traj
#         return: 
#         np
#         '''
#         if t == -1 or t >= len(self.waypoints)-1:
#             return self.waypoints[-1]
#         # if t != -1:
#         #     assert 0<=t<=(len(self.waypoints)-1)
#         t_prev, t_next = int(np.floor(t)), int(np.ceil(t))
#         spec_prev, spec_next = self.waypoints[t_prev], self.waypoints[t_next]

#         norm_diff = np.linalg.norm(spec_next - spec_prev)
#         # print('norm_diff', norm_diff)
#         spec_interp = spec_prev + (spec_next-spec_prev)*(t-t_prev)
#         if self.noisy and t > 0:
#             spec_interp = spec_interp + np.random.randn(*spec_interp.shape) * self.noise_config['std']

#         return spec_interp

#     def set_spec(self, obstacle, spec):
#         '''impl abstract method
#         place the obstace in new/next place one by one
#         obstacle ()
#         '''
#         obstacle.set_config(spec)        
        
class TrajectoryInterpolator:
    """
    Base class for trajectory interpolation methods.
    """

    def __call__(self, t):
        """
        Get position and orientation at time t.

        Args:
            t: Time value (scalar or tensor)

        Returns:
            pos: Position, shape (3,) or (batch, 3)
            ori: Orientation matrix, shape (3, 3) or (batch, 3, 3)
        """
        raise NotImplementedError


class LinearTrajectory(TrajectoryInterpolator):
    """
    Linear interpolation between keyframes.
    """

    def __init__(self, keyframe_times, keyframe_positions, keyframe_orientations=None, tensor_args=DEFAULT_TENSOR_ARGS):
        """
        Args:
            keyframe_times: List or tensor of time values, shape (num_keyframes,)
            keyframe_positions: Positions at keyframes, shape (num_keyframes, 3)
            keyframe_orientations: Orientations at keyframes, shape (num_keyframes, 3, 3) or None
            tensor_args: Tensor device and dtype
        """
        self.tensor_args = tensor_args
        self.keyframe_times = to_torch(keyframe_times, **tensor_args)
        self.keyframe_positions = to_torch(keyframe_positions, **tensor_args)

        if keyframe_orientations is not None:
            self.ignore_ori = False
            self.keyframe_orientations = to_torch(keyframe_orientations, **tensor_args)
        else:
            # Default to identity orientation
            self.ignore_ori = True
            num_keyframes = self.keyframe_positions.shape[0]
            self.keyframe_orientations = torch.eye(3, **tensor_args).unsqueeze(0).repeat(num_keyframes, 1, 1)

        assert self.keyframe_times.shape[0] == self.keyframe_positions.shape[0]
        assert self.keyframe_times.shape[0] == self.keyframe_orientations.shape[0]

    def __call__(self, t):
        """
        Linear interpolation of position and orientation.

        Args:
            t: Time value (scalar or tensor)

        Returns:
            pos: Position, shape (3,) or (batch, 3)
            ori: Orientation matrix, shape (3, 3) or (batch, 3, 3)
        """
        # Convert t to tensor if needed
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, **self.tensor_args)

        is_scalar = t.ndim == 0
        if is_scalar:
            t = t.unsqueeze(0)

        # Find bounding keyframes for each time value
        # keyframe_times: (num_keyframes,), t: (batch,)
        # Expand dimensions for broadcasting
        t_expanded = t.unsqueeze(-1)  # (batch, 1)
        times_expanded = self.keyframe_times.unsqueeze(0)  # (1, num_keyframes)

        # Find the index of the first keyframe after each t
        idx_high = torch.searchsorted(self.keyframe_times, t, right=False)
        idx_high = idx_high.clamp(1, len(self.keyframe_times) - 1)
        idx_low = idx_high - 1

        # Get bounding keyframes
        t_low = self.keyframe_times[idx_low]
        t_high = self.keyframe_times[idx_high]
        pos_low = self.keyframe_positions[idx_low]
        pos_high = self.keyframe_positions[idx_high]
        
        ori_low = self.keyframe_orientations[idx_low]
        ori_high = self.keyframe_orientations[idx_high]

        # Compute interpolation weight
        alpha = ((t - t_low) / (t_high - t_low + 1e-8)).clamp(0, 1)

        # Linear interpolation of position
        pos = (1 - alpha.unsqueeze(-1)) * pos_low + alpha.unsqueeze(-1) * pos_high

        # Linear interpolation of orientation (SLERP would be better but more complex)
        # For small rotations, linear interpolation is reasonable
        ori = (1 - alpha.unsqueeze(-1).unsqueeze(-1)) * ori_low + alpha.unsqueeze(-1).unsqueeze(-1) * ori_high

        # Renormalize rotation matrix (ensure orthonormality)
        # This is a simple approach; for better results use SLERP
        ori = self._orthonormalize_rotation(ori)

        if is_scalar:
            pos = pos.squeeze(0)
            ori = ori.squeeze(0)

        return pos, ori

    @staticmethod
    def _orthonormalize_rotation(R):
        """
        Orthonormalize rotation matrices using Gram-Schmidt.

        Args:
            R: Rotation matrices, shape (..., 3, 3)

        Returns:
            Orthonormalized rotation matrices
        """
        # Extract columns
        c1 = R[..., :, 0]
        c2 = R[..., :, 1]
        c3 = R[..., :, 2]

        # Gram-Schmidt
        u1 = c1 / (torch.linalg.norm(c1, dim=-1, keepdim=True) + 1e-8)
        u2 = c2 - (u1 * c2).sum(dim=-1, keepdim=True) * u1
        u2 = u2 / (torch.linalg.norm(u2, dim=-1, keepdim=True) + 1e-8)
        u3 = torch.cross(u1, u2, dim=-1)

        # Stack back
        R_ortho = torch.stack([u1, u2, u3], dim=-1)
        return R_ortho


class CircularTrajectory(TrajectoryInterpolator):
    """
    Circular motion around a center point.
    """

    def __init__(self, center, radius, angular_velocity, initial_phase=0.0,
                 axis='z', tensor_args=DEFAULT_TENSOR_ARGS):
        """
        Args:
            center: Center of circular motion, shape (3,)
            radius: Radius of circular motion
            angular_velocity: Angular velocity (rad/s)
            initial_phase: Initial phase angle (rad)
            axis: Axis of rotation ('x', 'y', or 'z')
            tensor_args: Tensor device and dtype
        """
        self.tensor_args = tensor_args
        self.center = to_torch(center, **tensor_args)
        self.radius = float(radius)
        self.angular_velocity = float(angular_velocity)
        self.initial_phase = float(initial_phase)
        self.axis = axis

    def __call__(self, t):
        """
        Compute position on circular trajectory.

        Args:
            t: Time value (scalar or tensor)

        Returns:
            pos: Position, shape (3,) or (batch, 3)
            ori: Orientation matrix (identity), shape (3, 3) or (batch, 3, 3)
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, **self.tensor_args)

        is_scalar = t.ndim == 0
        if is_scalar:
            t = t.unsqueeze(0)

        # Compute angle
        angle = self.angular_velocity * t + self.initial_phase

        # Compute position based on rotation axis
        if self.axis == 'z':
            # Rotate in xy-plane
            x = self.center[0] + self.radius * torch.cos(angle)
            y = self.center[1] + self.radius * torch.sin(angle)
            z = self.center[2].expand_as(t)
        elif self.axis == 'y':
            # Rotate in xz-plane
            x = self.center[0] + self.radius * torch.cos(angle)
            y = self.center[1].expand_as(t)
            z = self.center[2] + self.radius * torch.sin(angle)
        elif self.axis == 'x':
            # Rotate in yz-plane
            x = self.center[0].expand_as(t)
            y = self.center[1] + self.radius * torch.cos(angle)
            z = self.center[2] + self.radius * torch.sin(angle)
        else:
            raise ValueError(f"Unknown axis: {self.axis}")

        pos = torch.stack([x, y, z], dim=-1)

        # Identity orientation (could be extended to make object face tangent direction)
        batch_size = t.shape[0]
        ori = torch.eye(3, **self.tensor_args).unsqueeze(0).repeat(batch_size, 1, 1)

        if is_scalar:
            pos = pos.squeeze(0)
            ori = ori.squeeze(0)

        return pos, ori
