"""Entry point for Mujoco_RL_PickPlace.

Phase 1 commands:
    python main.py view        # open the interactive MuJoCo 3D viewer
    python main.py validate    # run headless scene validation
    python main.py render      # save an overview PNG to results/graphs/

Later phases add: train, evaluate, test, dynamic.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def cmd_view(args: argparse.Namespace) -> int:
    """Interactive viewer. Physics runs in real time; can run passive or autonomous."""
    import mujoco
    import mujoco.viewer
    from environment import scene_loader as sl

    cfg = sl.load_config(args.config)
    model, data = sl.load_model(cfg)
    rng = np.random.default_rng(cfg["seed"])

    if args.randomize:
        cube_xy, tgt_xy = sl.sample_cube_and_target(cfg, rng)
        sl.set_cube_pose(model, data, [cube_xy[0], cube_xy[1], sl.cube_rest_height(cfg)])
        sl.set_target_position(model, data, tgt_xy, cfg)

    is_auto = getattr(args, "auto", False)
    ctrl = None
    reward_fn = None
    if is_auto:
        from environment.ik_controller import AdaptivePickPlaceController
        from rewards.reward_function import PickPlaceReward

        ctrl = AdaptivePickPlaceController(model, cfg)
        reward_fn = PickPlaceReward(cfg)
        print("\n" + "=" * 70)
        print("  [AUTONOMOUS PICK & PLACE MODE ACTIVE]")
        print("  - Franka Panda will autonomously pick the red cube and place it on target.")
        print("  - Real-time step reward calculated at every physics step.")
        print("  - ADAPTIVE TRACKING: Move the cube with your mouse anytime:")
        print("      Double-click the red cube -> hold Ctrl + Right-Mouse-Drag.")
        print("      The robot will instantly detect the new position and pick it up!")
        print("=" * 70 + "\n")
    else:
        print("Viewer controls:")
        print("  - Double-click the red cube, then Ctrl + right-mouse-drag to push it around")
        print("  - Left-drag: rotate camera | Right-drag: pan | Scroll: zoom")
        print("  - Backspace: reset simulation | Space: pause | Close the window to exit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg["viewer"]["camera"])
        if cam_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id

        prev_time = data.time
        last_print_time = 0.0
        last_hud_time = 0.0
        last_state = ""

        while viewer.is_running():
            step_start = time.time()

            if ctrl is not None:
                # Detect simulation reset (e.g. Backspace key)
                if data.time < prev_time:
                    ctrl.reset()
                prev_time = data.time

                status = ctrl.step(model, data, model.opt.timestep)
                step_reward, bd = reward_fn.compute(model, data, data.ctrl)

                now = time.time()
                # Onscreen 3D HUD overlay inside the viewer window
                if now - last_hud_time > 0.05:
                    viewer.set_texts([
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                            "Stage:\nTotal Step Reward:\nReach Distance:\nTarget Distance:\nGrasped:\nLifted:\nSuccess:",
                            f"{ctrl.state}\n{step_reward:+.2f}\n{bd.dist_reach*100:.1f} cm\n{bd.dist_place*100:.1f} cm\n{'YES' if bd.is_grasped else 'NO'}\n{'YES' if bd.is_lifted else 'NO'}\n{'YES' if bd.is_success else 'NO'}",
                        )
                    ])
                    last_hud_time = now

                # Terminal console output
                if ctrl.state != last_state or (now - last_print_time > 1.5):
                    print(
                        f"[{ctrl.state:<9}] {status:<42} | "
                        f"Reward: {step_reward:5.2f} | "
                        f"Reach: {bd.dist_reach*100:4.1f}cm | "
                        f"Target Dist: {bd.dist_place*100:4.1f}cm"
                    )
                    last_state = ctrl.state
                    last_print_time = now

            mujoco.mj_step(model, data)
            viewer.sync()
            if cfg["viewer"]["realtime"]:
                sleep = model.opt.timestep - (time.time() - step_start)
                if sleep > 0:
                    time.sleep(sleep)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from testing import validate_scene

    sys.argv = ["validate_scene.py"] + (["--no-render"] if args.no_render else [])
    if args.config:
        sys.argv += ["--config", args.config]
    return validate_scene.main()


def cmd_render(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt
    import mujoco
    from environment import scene_loader as sl

    cfg = sl.load_config(args.config)
    model, data = sl.load_model(cfg)
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, cfg["validation"]["render_height"], cfg["validation"]["render_width"])
    renderer.update_scene(data, camera=args.camera)
    out_dir = sl.PROJECT_ROOT / cfg["paths"]["graphs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"scene_{args.camera}.png"
    plt.imsave(out, renderer.render())
    renderer.close()
    print(f"Saved {out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:
        from training import train as train_module
    except ImportError as exc:
        if "stable_baselines3" in str(exc):
            print("\n[ERROR] 'stable_baselines3' is not installed in the current Python environment.")
            print("Your virtual environment (venv) already has all dependencies installed.")
            print("Please activate your virtual environment before running:")
            print("    Windows (PowerShell): .\\venv\\Scripts\\Activate.ps1")
            print("    macOS / Linux:        source .venv/bin/activate")
            print("Or run directly using:")
            print("    .\\venv\\Scripts\\python.exe main.py train\n")
            return 1
        raise exc

    train_module.train(timesteps=args.timesteps, seed=args.seed, config_path=args.config)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate and visualize the trained PPO policy in the interactive 3D viewer."""
    import mujoco
    import mujoco.viewer
    from stable_baselines3 import PPO
    from environment.pick_place_env import PickPlaceEnv
    from environment import scene_loader as sl

    model_path = sl.PROJECT_ROOT / "results" / "models" / "ppo_pick_place.zip"
    if not model_path.is_file():
        print(f"\n[ERROR] Trained model not found at {model_path}. Run 'python main.py train' first.\n")
        return 1

    print(f"\nLoading trained PPO policy from: {model_path}...")
    model = PPO.load(str(model_path))

    cfg = sl.load_config(args.config)
    env = PickPlaceEnv(config=cfg, render_mode="human")
    obs, info = env.reset()
    print("=" * 65)
    print("  [RUNNING TRAINED REINFORCEMENT LEARNING POLICY IN 3D VIEWER]")
    print("  - Neural network is predicting 4-dim Cartesian actions in real-time.")
    print("  - Close the viewer window to exit.")
    print("=" * 65 + "\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        while viewer.is_running():
            step_start = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            sleep = 0.02 - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)
            if terminated or truncated:
                obs, info = env.reset()
    env.close()
    return 0


def cmd_plot_training(args: argparse.Namespace) -> int:
    from visualization import plot_training

    plot_training.plot_training_metrics()
    return 0


def cmd_plot_reward(args: argparse.Namespace) -> int:
    from testing import plot_reward

    plot_reward.record_and_plot_episode()
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    args.auto = True
    return cmd_view(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL-based adaptive pick-and-place in MuJoCo")
    parser.add_argument("--config", default=None, help="path to a custom config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_auto = sub.add_parser("auto", help="run autonomous adaptive pick-and-place tracking the cube")
    p_auto.add_argument("--randomize", action="store_true", help="random cube and target placement")
    p_auto.set_defaults(func=cmd_auto)

    p_train = sub.add_parser("train", help="train PPO policy on pick-and-place")
    p_train.add_argument("--timesteps", type=int, default=30000, help="timesteps to train")
    p_train.add_argument("--seed", type=int, default=42, help="random seed")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="visualize trained PPO policy in the interactive 3D viewer")
    p_eval.set_defaults(func=cmd_eval)

    p_plot_train = sub.add_parser("plot-training", help="plot training performance and success rate curves")
    p_plot_train.set_defaults(func=cmd_plot_training)

    p_plot = sub.add_parser("plot-reward", help="plot reward curves and save to results/graphs/reward_profile.png")
    p_plot.set_defaults(func=cmd_plot_reward)

    p_view = sub.add_parser("view", help="open the interactive 3D viewer")
    p_view.add_argument("--auto", action="store_true", help="enable autonomous adaptive pick-and-place")
    p_view.add_argument("--randomize", action="store_true", help="random cube and target placement")
    p_view.set_defaults(func=cmd_view)

    p_val = sub.add_parser("validate", help="run headless scene validation")
    p_val.add_argument("--no-render", action="store_true")
    p_val.set_defaults(func=cmd_validate)

    p_ren = sub.add_parser("render", help="save an offscreen PNG")
    p_ren.add_argument("--camera", default="overview", choices=["overview", "topdown"])
    p_ren.set_defaults(func=cmd_render)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(args.func(args))
