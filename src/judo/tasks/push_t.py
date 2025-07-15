# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
from dataclasses import dataclass, field
from typing import Any
import mujoco
import numpy as np

from judo.tasks.cost_functions import quadratic_norm
from judo.tasks.mujoco_task import MujocoTask
from judo.tasks.task import TaskConfig
from judo.viser_app.path_utils import MODEL_PATH
MODEL_PATH = str(MODEL_PATH / "xml/push_t.xml")

# works perfectly with
# cross_entropy_method
# horizon = 4
# numnodes = 6

@dataclass
class PushTConfig(TaskConfig):
    """Reward configuration for the push-T task."""
    w_pusher_proximity: float = 0.05
    w_pusher_velocity: float = 0.05
    w_T_position: float = 0.5
    w_T_angle: float = 0.25
    pusher_goal_offset: float = 0.05
    # We make the position 3 dimensional so that it triggers goal visualization in Viser.
    # T_goal_position: np.ndarray = field(default_factory=lambda: np.array([1.,1., 0]))
    T_goal_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    T_goal_angle: float = 0.0
    cutoff_time: float = 0.15
class PushT(MujocoTask[PushTConfig]):
    """Defines the Push-T task.
    The task is to push the T-shaped object to a goal position, defined by the xy-coordinates
    and the angle of the T (T_goal_position and T_goal_angle).
    qpos[0:2] are the pusher's xy-coordinates, qpos[2:4] are the T's xy-coordinates, and
    qpos[4] is the T's angle.
    The controls are the pusher's xy-position.
    """
    def __init__(self) -> None:
        super().__init__(MODEL_PATH)
        self.reset()
        config = PushTConfig()
        self.data.mocap_pos[:, :2] = config.T_goal_position[:2]
        angle = config.T_goal_angle
        self.data.mocap_quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        self.ctrl_qpos_idx = np.array([0, 1])
        self.goal_qpos_idx = np.array([2, 3])  # 2d position of T which can be used as contact prior
    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: PushTConfig,
        additional_info: dict[str, Any],
    ) -> np.ndarray:
        """Implements the T-pushing reward from MJPC.
        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.
        The cylinder push reward has four terms:
            * `pusher_reward`, penalizing the distance between the pusher and the T.
            * `velocity_reward` penalizing squared linear velocity of the pusher.
            * `goal_reward`, penalizing the distance of the T to the goal.
        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        batch_size = states.shape[0]
        pusher_pos = states[..., 0:2]
        T_pos = states[..., 2:4]
        T_angle = states[..., 5]
        T_pos_goal = config.T_goal_position[0:2]
        T_to_goal_pos = T_pos_goal - T_pos  # (batch_size, K, 2)
        T_to_goal_pos_norm = np.linalg.norm(T_to_goal_pos, axis=-1)  # (batch_size, K)
        T_to_goal_direction = T_to_goal_pos / T_to_goal_pos_norm[:, :, None]  # (batch_size, K, 2)
        pusher_goal = T_pos - config.pusher_goal_offset * T_to_goal_direction  # (batch_size, K, 2)
        pusher_proximity = quadratic_norm(pusher_pos - pusher_goal)  # (batch_size, K)
        pusher_reward = -config.w_pusher_proximity * pusher_proximity.sum(-1)  # (batch_size,)
        # Compute distance to goal position (pusher goal)
        goal_proximity_pos = quadratic_norm(T_pos - T_pos_goal)
        goal_reward_pos = -config.w_T_position * goal_proximity_pos.sum(-1)
        goal_proximity_angle = T_angle - config.T_goal_angle
        goal_proximity_angle = (goal_proximity_angle + np.pi) % (
            2 * np.pi
        ) - np.pi  # Normalize angle difference to [-pi, pi]
        angular_penalty = 0.5 * goal_proximity_angle**2
        goal_reward_angle = -config.w_T_angle * angular_penalty.sum(-1)
        progress = (
            goal_proximity_pos[:, 0]
            - goal_proximity_pos[:, -1]
            + 0.1 * (goal_proximity_angle[:, 0] - goal_proximity_angle[:, -1])
        )
        progress_cost = np.exp(-progress)
        goal_reward = goal_reward_pos + goal_reward_angle - progress_cost
        velocity_reward = -config.w_pusher_velocity * quadratic_norm(states[..., 6:8]).sum(-1)
        assert pusher_reward.shape == (batch_size,)
        assert velocity_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        total_reward = pusher_reward + velocity_reward + goal_reward
        return total_reward
    def is_terminated(self, config: PushTConfig) -> bool:
        """Termination condition for cylinder push. End if position is close enough to goal and velocity is small enough."""
        T_pos = self.data.qpos[2:4]
        T_goal_pos = config.T_goal_position[0:2]
        pos_diff = np.linalg.norm(T_goal_pos - T_pos)
        T_angle = self.data.qpos[5]
        angle_diff = T_angle - config.T_goal_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        self.pos_diff = pos_diff
        self.angle_diff = angle_diff
        return np.logical_and(pos_diff <= 0.01, np.abs(angle_diff) <= 0.05).astype(bool)
    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        theta = 2 * np.pi * np.random.rand()
        self.data.time = 0.0
        z_block = 0.0095
        self.data.qpos = np.array([0.1, 0.1, np.sin(theta) / 5, np.cos(theta) / 5, z_block, theta])
        self.data.qvel = np.zeros(6)
        mujoco.mj_forward(self.model, self.data)