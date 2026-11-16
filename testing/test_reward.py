"""Unit tests for PickPlaceReward function."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np

from environment import scene_loader as sl
from environment.ik_controller import AdaptivePickPlaceController
from rewards.reward_function import PickPlaceReward


def test_reward_progression():
    cfg = sl.load_config()
    model, data = sl.load_model(cfg)
    ctrl = AdaptivePickPlaceController(model, cfg)
    reward_fn = PickPlaceReward(cfg)

    # 1. Test at Initial Home Pose
    r_initial, breakdown_init = reward_fn.compute(model, data)
    print(f"Step 0 (Home pose): Reward={r_initial:.3f}")
    assert not breakdown_init.is_grasped
    assert not breakdown_init.is_lifted
    assert not breakdown_init.is_success
    assert breakdown_init.dist_reach > 0.1

    # 2. Step controller through full pick and place
    max_reward = -np.inf
    saw_grasp = False
    saw_lift = False
    saw_success = False

    for _ in range(int(7.5 / model.opt.timestep)):
        ctrl.step(model, data, model.opt.timestep)
        mujoco.mj_step(model, data)
        r, bd = reward_fn.compute(model, data, data.ctrl)
        max_reward = max(max_reward, r)

        if bd.is_grasped:
            saw_grasp = True
        if bd.is_lifted:
            saw_lift = True
        if bd.is_success:
            saw_success = True

    print(f"Post-pick & place: Max Reward observed={max_reward:.3f}")
    print(f"Saw grasp: {saw_grasp}, Saw lift: {saw_lift}, Saw success: {saw_success}")

    assert saw_grasp, "Grasp contact was not detected during picking!"
    assert saw_lift, "Lift condition was not detected during lifting!"
    assert saw_success, "Success condition was not achieved at target!"

    # 3. Test dictionary serialization for TensorBoard
    log_dict = bd.to_dict()
    assert "total_reward" in log_dict
    assert "r_reach" in log_dict
    assert "is_success" in log_dict
    print("Reward dict keys verified:", list(log_dict.keys()))
    print("All reward tests passed successfully!")


if __name__ == "__main__":
    test_reward_progression()

