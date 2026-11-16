"""Generates and saves a visual plot of reward progression across an episode.

Usage:
    python testing/plot_reward.py
Saves:
    results/graphs/reward_profile.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from environment import scene_loader as sl
from environment.ik_controller import AdaptivePickPlaceController
from rewards.reward_function import PickPlaceReward


def record_and_plot_episode():
    cfg = sl.load_config()
    model, data = sl.load_model(cfg)
    ctrl = AdaptivePickPlaceController(model, cfg)
    reward_fn = PickPlaceReward(cfg)

    times = []
    total_rewards = []
    r_reaches = []
    r_grasps = []
    r_lifts = []
    r_places = []
    r_successes = []
    dist_reaches = []
    dist_places = []
    cube_heights = []

    print("Simulating episode and recording step-by-step reward values...")
    # Simulate 8.0 seconds of pick and place
    total_steps = int(8.0 / model.opt.timestep)
    for step in range(total_steps):
        t = step * model.opt.timestep
        ctrl.step(model, data, model.opt.timestep)
        step_reward, bd = reward_fn.compute(model, data, data.ctrl)
        mujoco.mj_step(model, data)

        times.append(t)
        total_rewards.append(step_reward)
        r_reaches.append(bd.r_reach)
        r_grasps.append(bd.r_grasp)
        r_lifts.append(bd.r_lift)
        r_places.append(bd.r_place)
        r_successes.append(bd.r_success)
        dist_reaches.append(bd.dist_reach)
        dist_places.append(bd.dist_place)
        cube_heights.append(bd.lift_height)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1. Total Reward
    ax1.plot(times, total_rewards, color="#1f77b4", linewidth=2.0, label="Total Step Reward")
    ax1.set_ylabel("Step Reward", fontsize=11, fontweight="bold")
    ax1.set_title("Robot Performance & Reward Values across Pick-and-Place", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Individual Reward Components
    ax2.plot(times, r_reaches, label="Reaching (r_reach)", alpha=0.85)
    ax2.plot(times, r_grasps, label="Grasping (r_grasp)", alpha=0.85)
    ax2.plot(times, r_lifts, label="Lifting (r_lift)", alpha=0.85)
    ax2.plot(times, r_places, label="Placing (r_place)", alpha=0.85)
    ax2.plot(times, r_successes, label="Success (r_success)", color="gold", linewidth=2.0)
    ax2.set_ylabel("Component Values", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", ncol=3)

    # 3. Geometric Distances
    ax3.plot(times, np.array(dist_reaches) * 100, label="Gripper to Cube Distance (cm)", color="crimson")
    ax3.plot(times, np.array(dist_places) * 100, label="Cube to Target Distance (cm)", color="forestgreen")
    ax3.plot(times, np.array(cube_heights) * 100, label="Lift Height (cm)", color="purple", linestyle="--")
    ax3.set_ylabel("Distance / Height (cm)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    out_dir = sl.PROJECT_ROOT / cfg["paths"]["graphs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reward_profile.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Successfully saved reward plot to: {out_path}")
    return out_path


if __name__ == "__main__":
    record_and_plot_episode()

