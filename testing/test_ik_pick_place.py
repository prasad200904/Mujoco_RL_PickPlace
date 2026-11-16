"""Unit tests for Cartesian IK and Adaptive Pick-and-Place Controller."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np

from environment import scene_loader as sl
from environment.ik_controller import AdaptivePickPlaceController, CartesianIK


def test_cartesian_ik():
    cfg = sl.load_config()
    ik = CartesianIK(cfg)

    # Test reaching a point above the table
    target_pos = np.array([0.50, -0.10, 0.50])
    q_sol = ik.solve(target_pos, ik.q_home)

    ik.data.qpos[:7] = q_sol
    mujoco.mj_forward(ik.model, ik.data)
    pos = ik.data.site_xpos[ik.site_id]

    err = np.linalg.norm(pos - target_pos)
    assert err < 0.005, f"IK target position error too large: {err:.4f} m"
    print(f"CartesianIK test passed! Pos error: {err * 1000:.2f} mm")


def test_pick_and_place_adaptive():
    cfg = sl.load_config()
    model, data = sl.load_model(cfg)
    ctrl = AdaptivePickPlaceController(model, cfg)

    # 1. Run until first cycle completes
    for _ in range(int(7.0 / model.opt.timestep)):
        ctrl.step(model, data, model.opt.timestep)
        mujoco.mj_step(model, data)

    cube_pos = sl.cube_position(model, data)
    tgt_pos = sl.target_position(model, data)
    dist1 = np.linalg.norm(cube_pos[:2] - tgt_pos[:2])
    assert dist1 < 0.04, f"Cube not placed at target in cycle 1: dist={dist1:.4f} m"
    print(f"Cycle 1 passed! Distance to target: {dist1 * 1000:.1f} mm")

    # 2. Perturb cube to new position (simulating user dragging the cube)
    new_cube_xy = np.array([0.46, -0.18])
    sl.set_cube_pose(model, data, np.array([new_cube_xy[0], new_cube_xy[1], sl.cube_rest_height(cfg)]))

    for _ in range(int(7.0 / model.opt.timestep)):
        ctrl.step(model, data, model.opt.timestep)
        mujoco.mj_step(model, data)

    cube_pos2 = sl.cube_position(model, data)
    dist2 = np.linalg.norm(cube_pos2[:2] - tgt_pos[:2])
    assert dist2 < 0.04, f"Cube not re-placed at target in cycle 2: dist={dist2:.4f} m"
    print(f"Cycle 2 (Adaptive re-pick) passed! Distance to target: {dist2 * 1000:.1f} mm")


if __name__ == "__main__":
    test_cartesian_ik()
    test_pick_and_place_adaptive()
    print("All IK & Adaptive Pick-and-Place tests passed!")

