"""Gymnasium Environment for Franka Panda Pick-and-Place in MuJoCo.

Observation Space (24-dim Box):
    - End-effector position [x, y, z] (3)
    - End-effector linear velocity (3)
    - Gripper finger joint positions (2)
    - Cube position [x, y, z] (3)
    - Cube orientation quaternion [qw, qx, qy, qz] (4)
    - Target marker position [x, y, z] (3)
    - Relative vector: cube_pos - gripper_pos (3)
    - Relative vector: target_pos - cube_pos (3)

Action Space (4-dim Box, range [-1, 1]):
    - action[0:3]: End-effector Cartesian delta [dx, dy, dz] (scaled to max 3 cm/step)
    - action[3]: Gripper control (> 0: open [255], <= 0: close [0])
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from environment import scene_loader as sl
from environment.ik_controller import CartesianIK
from rewards.reward_function import PickPlaceReward, RewardBreakdown


class PickPlaceEnv(gym.Env):
    """Reinforcement learning environment for Franka Panda Pick and Place."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        max_episode_steps: int = 150,
        substeps: int = 10,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config or sl.load_config()
        self.max_episode_steps = max_episode_steps
        self.substeps = substeps  # 10 substeps * 0.002s = 0.02s (50 Hz control rate)
        self.render_mode = render_mode

        # Load MuJoCo model and data
        self.model, self.data = sl.load_model(self.config)
        self.ik = CartesianIK(self.config)
        self.reward_fn = PickPlaceReward(self.config)

        # Simulation indices
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sl.GRIP_SITE)
        self.finger1_adr = self.model.joint("finger_joint1").qposadr[0]
        self.finger2_adr = self.model.joint("finger_joint2").qposadr[0]

        # Action space: [dx, dy, dz, gripper]
        self.max_cartesian_step = 0.03  # 3 cm max displacement per control step
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: 24-dimensional continuous vector
        obs_dim = 24
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.np_random = np.random.default_rng(self.config.get("seed", 42))
        self.viewer = None
        self._last_grip_pos = np.zeros(3)

    def _get_obs(self) -> np.ndarray:
        """Extract continuous observation vector."""
        grip_pos = self.data.site_xpos[self.site_id].copy()
        dt = self.substeps * self.model.opt.timestep
        grip_vel = (grip_pos - self._last_grip_pos) / max(dt, 1e-6)
        self._last_grip_pos = grip_pos.copy()

        finger_pos = np.array([
            self.data.qpos[self.finger1_adr],
            self.data.qpos[self.finger2_adr],
        ], dtype=np.float32)

        cube_pos = sl.cube_position(self.model, self.data)
        cube_jnt = self.model.joint(sl.CUBE_JOINT)
        cube_quat = self.data.qpos[cube_jnt.qposadr[0] + 3 : cube_jnt.qposadr[0] + 7].copy()
        target_pos = sl.target_position(self.model, self.data)

        rel_grip_cube = cube_pos - grip_pos
        rel_cube_target = target_pos - cube_pos

        obs = np.concatenate([
            grip_pos,
            grip_vel,
            finger_pos,
            cube_pos,
            cube_quat,
            target_pos,
            rel_grip_cube,
            rel_cube_target,
        ]).astype(np.float32)
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        sl.reset_to_home(self.model, self.data, self.config)
        self.current_step = 0

        # Sample randomized cube and target positions on table
        cube_xy, tgt_xy = sl.sample_cube_and_target(self.config, self.np_random)
        rest_z = sl.cube_rest_height(self.config)
        sl.set_cube_pose(self.model, self.data, np.array([cube_xy[0], cube_xy[1], rest_z + 0.005]))
        sl.set_target_position(self.model, self.data, tgt_xy, self.config)

        # Settle simulation under gravity for 100 physics steps
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self._last_grip_pos = self.data.site_xpos[self.site_id].copy()
        obs = self._get_obs()
        info = {
            "is_success": False,
            "dist_place": float(np.linalg.norm(cube_xy - tgt_xy)),
            "cube_pos": cube_xy.tolist(),
            "target_pos": tgt_xy.tolist(),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1. Compute desired Cartesian position for gripper
        current_grip = self.data.site_xpos[self.site_id].copy()
        delta_xyz = action[:3] * self.max_cartesian_step
        target_grip = current_grip + delta_xyz

        # Bound target position inside workspace box
        target_grip[0] = np.clip(target_grip[0], 0.25, 0.75)
        target_grip[1] = np.clip(target_grip[1], -0.35, 0.35)
        target_grip[2] = np.clip(target_grip[2], self.config["scene"]["table_top_z"] + 0.02, 0.70)

        # 2. Solve inverse kinematics to obtain joint target
        q_target = self.ik.solve(target_grip, self.data.qpos[:7], max_iters=15)
        self.data.ctrl[:7] = q_target

        # 3. Gripper action: > 0 opens (255), <= 0 closes (0)
        self.data.ctrl[7] = 255.0 if action[3] > 0.0 else 0.0

        # 4. Step physics for substeps
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        # 5. Compute observations, reward and termination
        obs = self._get_obs()
        reward, bd = self.reward_fn.compute(self.model, self.data, action)

        cube_pos = sl.cube_position(self.model, self.data)
        # Terminate if successfully placed or cube falls off table (z < 0.2)
        terminated = bool(bd.is_success or cube_pos[2] < 0.20)
        truncated = bool(self.current_step >= self.max_episode_steps)

        info = {
            "is_success": bd.is_success,
            "reward_breakdown": bd.to_dict(),
            "dist_reach": bd.dist_reach,
            "dist_place": bd.dist_place,
            "lift_height": bd.lift_height,
        }

        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data, camera="overview")
            img = renderer.render()
            renderer.close()
            return img
        elif self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

