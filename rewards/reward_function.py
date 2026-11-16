"""Reward functions for reinforcement learning pick-and-place.

Calculates the robot's performance at every simulation step.
Provides both dense (shaped multi-stage) and sparse reward formulations,
along with a detailed component breakdown for TensorBoard logging.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from environment import scene_loader as sl


@dataclass
class RewardBreakdown:
    """Detailed breakdown of individual reward terms at the current step."""

    total_reward: float
    r_reach: float
    r_grasp: float
    r_lift: float
    r_place: float
    r_success: float
    r_penalty: float
    dist_reach: float
    dist_place: float
    lift_height: float
    is_grasped: bool
    is_lifted: bool
    is_success: bool

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary of floats for RL logger / TensorBoard."""
        d = asdict(self)
        # Ensure booleans are converted to floats (0.0 or 1.0) for loggers
        for k, v in d.items():
            if isinstance(v, bool):
                d[k] = 1.0 if v else 0.0
        return d


class PickPlaceReward:
    """Evaluates the robot's pick-and-place performance at each environment step.

    Stages tracked:
    1. Reaching: Moving the gripper site towards the cube center.
    2. Grasping: Making two-finger contact with the cube.
    3. Lifting: Raising the cube vertically off the table.
    4. Placing: Guiding the lifted cube directly towards the target site.
    5. Success: Bringing the cube within the target tolerance radius.
    6. Penalties: Action effort and excessive joint velocity penalties.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or sl.load_config()
        rew_cfg = self.config.get("reward", {})
        self.mode = rew_cfg.get("type", "dense")

        # Thresholds
        self.success_dist = rew_cfg.get("success_distance", 0.035)
        self.lift_thresh = rew_cfg.get("lift_threshold", 0.03)
        self.target_lift = rew_cfg.get("target_lift_height", 0.15)
        self.rest_z = sl.cube_rest_height(self.config)

        # Weights
        w = rew_cfg.get("weights", {})
        self.w_reach = w.get("reach", 1.0)
        self.w_grasp = w.get("grasp", 0.5)
        self.w_lift = w.get("lift", 1.5)
        self.w_place = w.get("place", 2.5)
        self.w_success = w.get("success", 10.0)
        self.w_action_penalty = w.get("action_penalty", 0.01)
        self.w_vel_penalty = w.get("velocity_penalty", 0.005)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: np.ndarray | None = None,
    ) -> tuple[float, RewardBreakdown]:
        """Compute the step reward and detailed metrics from current state.

        Args:
            model: Compiled MuJoCo MjModel.
            data: Dynamic simulation MjData.
            action: Optional action array applied at this step (for effort penalty).

        Returns:
            total_reward: Scalar reward value for the RL optimizer.
            breakdown: RewardBreakdown dataclass with each term and status booleans.
        """
        # 1. Kinematic positions
        cube_pos = sl.cube_position(model, data)
        target_pos = sl.target_position(model, data)
        grip_pos = sl.gripper_position(model, data)

        # 2. Distances
        dist_reach = float(np.linalg.norm(grip_pos - cube_pos))
        dist_place = float(np.linalg.norm(cube_pos[:2] - target_pos[:2]))
        lift_height = float(max(0.0, cube_pos[2] - self.rest_z))

        # 3. Contact & state conditions
        left_touch, right_touch = sl.finger_cube_contact(model, data)
        is_grasped = bool(left_touch and right_touch)
        is_lifted = bool(lift_height >= self.lift_thresh)
        is_success = bool(dist_place <= self.success_dist and abs(cube_pos[2] - self.rest_z) < 0.04)

        # 4. Dense shaped components (using smooth tanh bounds)
        # Stage 1: Reaching (smoothly approaches 1.0 as distance -> 0)
        r_reach = 1.0 - np.tanh(10.0 * dist_reach)

        # Stage 2: Grasping (bonus when both fingers securely contact the cube)
        r_grasp = 1.0 if is_grasped else (0.5 if (left_touch or right_touch) else 0.0)

        # Stage 3: Lifting (smooth progress up to target lift height, conditioned on grasp)
        if is_lifted:
            r_lift = float(np.clip(lift_height / self.target_lift, 0.0, 1.0))
        elif is_grasped:
            r_lift = 0.2
        else:
            r_lift = 0.0

        # Stage 4: Placing (bring cube close to target; active when cube is off table)
        if is_lifted:
            r_place = 1.0 - np.tanh(5.0 * dist_place)
        else:
            r_place = 0.0

        # Stage 5: Success bonus
        r_success = 1.0 if is_success else 0.0

        # Stage 6: Smoothness / Energy penalties
        r_penalty = 0.0
        if action is not None:
            # Normalize actions: arm joints are in radians, gripper is [0, 255]
            norm_action = np.array(action, dtype=np.float64).copy()
            if len(norm_action) >= 8:
                norm_action[7] = norm_action[7] / 255.0  # normalize gripper to [0, 1]
            r_penalty += self.w_action_penalty * float(np.sum(np.square(norm_action)))

        # Joint velocity penalty on arm DOFs (first 7 DOFs)
        qvel_arm = data.qvel[:7]
        r_penalty += self.w_vel_penalty * float(np.sum(np.square(qvel_arm)))

        # 5. Total Reward calculation
        if self.mode == "sparse":
            # Sparse: +1.0 upon achieving success, 0.0 otherwise
            total_reward = 1.0 if is_success else 0.0
        else:
            # Dense shaped sum
            total_reward = (
                self.w_reach * r_reach
                + self.w_grasp * r_grasp
                + self.w_lift * r_lift
                + self.w_place * r_place
                + self.w_success * r_success
                - r_penalty
            )

        breakdown = RewardBreakdown(
            total_reward=float(total_reward),
            r_reach=float(r_reach),
            r_grasp=float(r_grasp),
            r_lift=float(r_lift),
            r_place=float(r_place),
            r_success=float(r_success),
            r_penalty=float(r_penalty),
            dist_reach=dist_reach,
            dist_place=dist_place,
            lift_height=lift_height,
            is_grasped=is_grasped,
            is_lifted=is_lifted,
            is_success=is_success,
        )

        return float(total_reward), breakdown
