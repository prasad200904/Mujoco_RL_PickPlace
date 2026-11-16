"""Inverse Kinematics and Adaptive Pick-and-Place Controller.

Provides:
- CartesianIK: Damped Least Squares IK solver with nullspace posture regularization.
- AdaptivePickPlaceController: Dynamic closed-loop state machine that continuously
  tracks the cube's live position, picks it up, transports it to the target, and
  adapts immediately if the user moves the cube.
"""
from __future__ import annotations

import time
from typing import Any

import mujoco
import numpy as np

from environment import scene_loader as sl


class CartesianIK:
    """Damped Least-Squares IK solver targeting the robot end-effector (grip_site)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        xml_path = sl.PROJECT_ROOT / config["paths"]["scene_xml"]
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        sl.reset_to_home(self.model, self.data, config)

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sl.GRIP_SITE)
        # Default desired orientation: gripper pointed vertically downwards (-Z)
        self.target_mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.q_home = self.data.qpos[:7].copy()

    def solve(
        self,
        target_pos: np.ndarray,
        q_seed: np.ndarray,
        target_mat: np.ndarray | None = None,
        max_iters: int = 25,
        tol: float = 1e-3,
    ) -> np.ndarray:
        """Solve for joint angles [q0..q6] reaching target_pos with specified orientation."""
        des_mat = self.target_mat if target_mat is None else target_mat
        self.data.qpos[:7] = q_seed.copy()

        for _ in range(max_iters):
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site_xpos[self.site_id]
            mat = self.data.site_xmat[self.site_id].reshape(3, 3)

            pos_err = target_pos - pos
            rot_err = 0.5 * (
                np.cross(mat[:, 0], des_mat[:, 0])
                + np.cross(mat[:, 1], des_mat[:, 1])
                + np.cross(mat[:, 2], des_mat[:, 2])
            )
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(err) < tol:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])  # 6 x 7

            lambda_sq = 1e-3
            # Damped pseudo-inverse
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
            # Nullspace projection towards home pose for posture stability
            null_proj = np.eye(7) - J_pinv @ J
            dq = J_pinv @ err + null_proj @ (0.05 * (self.q_home - self.data.qpos[:7]))

            self.data.qpos[:7] += 0.5 * dq

        return self.data.qpos[:7].copy()


class AdaptivePickPlaceController:
    """Closed-loop adaptive state machine for continuous pick-and-place.

    Monitors the real-time world position of the cube and target. If the user
    perturbs or moves the cube, the controller dynamically re-plans and tracks it.
    """

    STATES = (
        "APPROACH",
        "DESCEND",
        "GRASP",
        "LIFT",
        "TRANSPORT",
        "PLACE",
        "RELEASE",
        "RETREAT",
        "IDLE",
    )

    def __init__(self, model: mujoco.MjModel, config: dict[str, Any]):
        self.model = model
        self.config = config
        self.ik = CartesianIK(config)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sl.GRIP_SITE)

        self.q_home = self.ik.q_home.copy()
        self.current_q_des = self.q_home.copy()
        self.gripper_ctrl = 255.0  # Open gripper (0 = closed, 255 = open)

        self.state = "APPROACH"
        self.state_time = 0.0
        self.ik_interval = 0.02  # Update IK solution at 50 Hz
        self.last_ik_time = -1.0
        self.target_q_cached = self.q_home.copy()
        self.status_msg = "Initializing..."

    def reset(self) -> None:
        """Reset state machine to start picking."""
        self.state = "APPROACH"
        self.state_time = 0.0
        self.current_q_des = self.q_home.copy()
        self.target_q_cached = self.q_home.copy()
        self.gripper_ctrl = 255.0

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, dt: float) -> str:
        """Execute one control step, updating data.ctrl. Returns current status."""
        self.state_time += dt
        cube_p = sl.cube_position(model, data)
        tgt_p = sl.target_position(model, data)
        grip_p = data.site_xpos[self.site_id]

        dist_cube_tgt = np.linalg.norm(cube_p[:2] - tgt_p[:2])
        horiz_dist_to_cube = np.linalg.norm(grip_p[:2] - cube_p[:2])

        # Height parameters (meters)
        table_z = self.config["scene"]["table_top_z"]
        hover_z = table_z + 0.16  # Hover height above table
        lift_z = table_z + 0.20   # Carry height during transport
        grasp_z = table_z + 0.032 # Gripper contact height for 5cm cube

        # State transitions
        prev_state = self.state

        if self.state == "IDLE":
            self.gripper_ctrl = 255.0
            target_ee = np.array([tgt_p[0], tgt_p[1], hover_z])
            self.status_msg = "Task Complete! Waiting. (Move the cube to restart)"
            # If user drags the cube away from target, wake up and pick it again!
            if dist_cube_tgt > 0.05:
                self.state = "APPROACH"
                self.state_time = 0.0

        elif self.state == "APPROACH":
            self.gripper_ctrl = 255.0
            target_ee = np.array([cube_p[0], cube_p[1], hover_z])
            self.status_msg = f"Tracking cube at [{cube_p[0]:.2f}, {cube_p[1]:.2f}]"
            # Once hovered directly above the cube for enough time:
            if horiz_dist_to_cube < 0.025 and self.state_time > 0.8:
                self.state = "DESCEND"
                self.state_time = 0.0

        elif self.state == "DESCEND":
            self.gripper_ctrl = 255.0
            target_ee = np.array([cube_p[0], cube_p[1], grasp_z])
            self.status_msg = "Descending over cube..."
            # If user pulls the cube away while descending, re-align
            if horiz_dist_to_cube > 0.04:
                self.state = "APPROACH"
                self.state_time = 0.0
            elif np.linalg.norm(grip_p - target_ee) < 0.015 and self.state_time > 0.7:
                self.state = "GRASP"
                self.state_time = 0.0

        elif self.state == "GRASP":
            self.gripper_ctrl = 0.0  # Close fingers firmly
            target_ee = np.array([cube_p[0], cube_p[1], grasp_z])
            self.status_msg = "Grasping cube..."
            if self.state_time > 0.6:
                self.state = "LIFT"
                self.state_time = 0.0

        elif self.state == "LIFT":
            self.gripper_ctrl = 0.0
            target_ee = np.array([cube_p[0], cube_p[1], lift_z])
            self.status_msg = "Lifting cube..."
            if abs(grip_p[2] - lift_z) < 0.03 and self.state_time > 0.8:
                self.state = "TRANSPORT"
                self.state_time = 0.0

        elif self.state == "TRANSPORT":
            self.gripper_ctrl = 0.0
            target_ee = np.array([tgt_p[0], tgt_p[1], lift_z])
            self.status_msg = f"Transporting to target [{tgt_p[0]:.2f}, {tgt_p[1]:.2f}]..."
            if np.linalg.norm(grip_p[:2] - tgt_p[:2]) < 0.025 and self.state_time > 1.2:
                self.state = "PLACE"
                self.state_time = 0.0

        elif self.state == "PLACE":
            self.gripper_ctrl = 0.0
            target_ee = np.array([tgt_p[0], tgt_p[1], grasp_z + 0.015])
            self.status_msg = "Placing on target..."
            if abs(grip_p[2] - target_ee[2]) < 0.02 and self.state_time > 0.8:
                self.state = "RELEASE"
                self.state_time = 0.0

        elif self.state == "RELEASE":
            self.gripper_ctrl = 255.0  # Open gripper
            target_ee = np.array([tgt_p[0], tgt_p[1], grasp_z + 0.02])
            self.status_msg = "Releasing..."
            if self.state_time > 0.5:
                self.state = "RETREAT"
                self.state_time = 0.0

        elif self.state == "RETREAT":
            self.gripper_ctrl = 255.0
            target_ee = np.array([tgt_p[0], tgt_p[1], hover_z])
            self.status_msg = "Retreating..."
            if abs(grip_p[2] - hover_z) < 0.03 and self.state_time > 0.8:
                self.state = "IDLE"
                self.state_time = 0.0
        else:
            target_ee = np.array([cube_p[0], cube_p[1], hover_z])

        # Solve IK periodically (50 Hz) or on state transition to stay fast and responsive
        sim_time = data.time
        if (sim_time - self.last_ik_time >= self.ik_interval) or (self.state != prev_state):
            self.target_q_cached = self.ik.solve(target_ee, self.current_q_des)
            self.last_ik_time = sim_time

        # Smooth low-pass joint interpolation for natural, jerk-free arm movement
        alpha = min(1.0, 6.0 * dt)
        self.current_q_des = (1.0 - alpha) * self.current_q_des + alpha * self.target_q_cached

        # Write to actuators
        data.ctrl[:7] = self.current_q_des
        data.ctrl[7] = self.gripper_ctrl

        return self.status_msg

