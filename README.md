# Mujoco_RL_PickPlace

Reinforcement-learning-based adaptive pick-and-place with a Franka Panda in MuJoCo.

**Features Implemented:**
- **Scene & Physics:** Franka Panda 7-DOF robot mounted on pedestal, table, 5 cm cube, target marker.
- **Gymnasium Environment:** `PickPlaceEnv` with 24-dim continuous observations and 4-dim Cartesian actions.
- **Adaptive Controller:** Dynamic inverse kinematics state machine that tracks live cube relocation.
- **Reward Function:** Multi-stage shaped dense reward considering reach, grasp, lift, placement, success bonus, and effort penalties.
- **RL Training (PPO):** Full training pipeline via Stable-Baselines3 with live console feedback and TensorBoard logging.
- **Visualization:** Graph generators for episode reward profiles and training performance curves.

## Project Structure

| Item | Where |
|---|---|
| Franka Panda mounted on a pedestal | `assets/robot/panda.xml` |
| Table, cube, target marker | `assets/table.xml`, `assets/object.xml` |
| Scene: floor, lights, cameras, home keyframe | `assets/scene.xml` |
| Central configuration | `configs/config.yaml` |
| Gymnasium RL Environment | `environment/pick_place_env.py` |
| Inverse Kinematics & Adaptive Tracking | `environment/ik_controller.py` |
| Scene loader & geometric helpers | `environment/scene_loader.py` |
| Step-by-Step Reward Function | `rewards/reward_function.py` |
| PPO Training Pipeline | `training/train.py` |
| Training Performance Graphs | `visualization/plot_training.py` |
| Reward Profile Visualizer | `testing/plot_reward.py` |
| CLI Entrypoint | `main.py` |
| Scene loader / helpers (cube pose, target pose, random spawns) | `environment/scene_loader.py` |
| Headless validation (21 checks) | `testing/validate_scene.py` |
| CLI: `view`, `validate`, `render` | `main.py` |

Changes made to the stock Menagerie `panda.xml`: base raised to z = 0.40 (sits on the pedestal),
removed its stand-alone keyframe (the scene owns it), added a `grip_site` between the fingertip pads,
and removed `meshdir` (set in `scene.xml`).

## Setup

### macOS / Linux (Terminal)

```bash
# 1. Create and activate virtual environment (Python 3.10 - 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Important for macOS:** On macOS, the interactive viewer requires MuJoCo's Cocoa-compatible launcher **`mjpython`** (installed automatically into your virtual environment by `mujoco`). Use `mjpython` instead of `python` whenever opening the 3D viewer window.

### Windows (PowerShell)

```powershell
# 1. Create and activate virtual environment (Python 3.11 or newer)
py -3.11 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run

### On macOS

```bash
# Interactive viewer (requires mjpython on macOS)
mjpython main.py auto                # autonomous pick-and-place with adaptive tracking
mjpython main.py auto --randomize    # random initial cube + target placement
mjpython main.py view                # passive 3D viewer (arm holds home pose)

# RL Training & Performance Graphs
python main.py train --timesteps 30000 # train PPO policy on pick-and-place
python main.py plot-training          # plot cumulative reward & success rate graphs
python main.py plot-reward            # plot reward curves to results/graphs/reward_profile.png

# Validation & Unit tests
python main.py validate               # headless physics checks (21/21 checks)
python main.py render                 # offscreen PNG render to results/graphs/
python testing/test_reward.py         # unit tests for step reward function
python testing/test_ik_pick_place.py  # unit tests for IK & adaptive picking
```

### On Windows

```powershell
# Interactive 3D Viewer
python main.py auto                   # autonomous pick-and-place with adaptive tracking
python main.py auto --randomize       # random initial cube + target placement
python main.py view                   # passive 3D viewer (arm holds home pose)

# RL Training & Graphs
python main.py train --timesteps 30000# train PPO policy
python main.py plot-training          # plot cumulative reward & success rate graphs
python main.py plot-reward            # plot reward curves to results\graphs\reward_profile.png

# Validation & Tests
python main.py validate               # headless physics checks, expect "21/21 checks passed"
python main.py render                 # saves results\graphs\scene_overview.png
```

Viewer tips:
- In `python main.py auto`: The robot automatically identifies and picks up the red cube, and places it onto the green target.
- **Adaptive tracking**: Double-click the red cube and hold **Ctrl + right-mouse-drag** to relocate the cube anywhere on the table. The robot immediately identifies the new position and picks it up again!
- Press **Backspace** to reset the simulation, and **Space** to pause/unpause.

## Scene geometry (world frame, metres)

- Robot base: (0, 0, 0.40). Table top: x 0.15-0.95, y -0.5-0.5, z = 0.40.
- Cube rests at z = 0.425. Default cube (0.50, -0.15), default target (0.50, 0.20).
- Home pose: gripper pointing down at about (0.50, 0, 0.61).

## Roadmap

1. Scene (done) 2. Robot/table/cube/target (done with 1) 3. Cartesian/IK control
4. Perception + observations 5. `PickPlaceEnv` 6. Reward function 7. PPO training
8. Training visualisation 9. Dynamic adaptation 10. Evaluation + docs
