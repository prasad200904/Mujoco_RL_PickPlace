"""Phase 1 scene validation.

Runs headless checks that the MuJoCo scene is physically sane before any control
or RL code is built on top of it.  Exit code is 0 only if every check passes.

Usage (from the project root):
    python testing/validate_scene.py
    python testing/validate_scene.py --no-render      # skip the PNG snapshot
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python testing/validate_scene.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np

from environment import scene_loader as sl


class Checker:
    """Collects PASS/FAIL lines so the whole report prints even if one check fails."""

    def __init__(self) -> None:
        self.results: list[tuple[bool, str]] = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        self.results.append((bool(ok), label))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))

    @property
    def all_ok(self) -> bool:
        return all(ok for ok, _ in self.results)


def simulate(model, data, seconds: float) -> None:
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)


def check_model_structure(cfg, model, data, chk: Checker) -> None:
    print("1. Model structure")
    chk.check(model.nu == 8, "8 actuators (7 arm joints + gripper)", f"nu={model.nu}")
    chk.check(model.nmocap == 1, "target is a mocap body", f"nmocap={model.nmocap}")
    chk.check(np.allclose(model.opt.gravity, [0, 0, -9.81]), "gravity is -9.81 m/s^2")
    cube_mass = model.body(sl.CUBE_BODY).mass[0]
    chk.check(abs(cube_mass - 0.1) < 1e-6, "cube mass is 0.1 kg", f"{cube_mass:.3f} kg")
    friction = model.geom(sl.CUBE_GEOM).friction[0]
    chk.check(friction >= 1.0, "cube sliding friction >= 1.0", f"mu={friction:.2f}")
    for name in sl.ARM_JOINTS + sl.FINGER_JOINTS:
        model.joint(name)  # raises KeyError if missing
    chk.check(True, "all 7 arm joints + 2 finger joints present")
    for cam in ("overview", "topdown"):
        chk.check(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0, f"camera '{cam}' defined")


def check_settle(cfg, model, data, chk: Checker) -> None:
    print("2. Settling at home pose")
    sl.reset_to_home(model, data, cfg)
    q_home = data.qpos[:7].copy()
    simulate(model, data, cfg["validation"]["settle_seconds"])
    finite = np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    chk.check(finite, "no NaN/Inf in state after settling")

    rest_z = sl.cube_rest_height(cfg)
    cube = sl.cube_position(model, data)
    tol = cfg["validation"]["cube_rest_tolerance"]
    chk.check(abs(cube[2] - rest_z) < tol, "cube rests on table surface", f"z={cube[2]:.4f}, expected {rest_z:.4f}")
    chk.check(sl.cube_table_contact(model, data), "cube-table contact detected")
    chk.check(np.linalg.norm(data.qvel[model.joint(sl.CUBE_JOINT).dofadr[0]:][:6]) < 1e-2, "cube is at rest")

    err = np.max(np.abs(data.qpos[:7] - q_home))
    chk.check(err < cfg["validation"]["home_joint_tolerance"], "arm holds home pose under gravity", f"max err={err:.4f} rad")

    grip = sl.gripper_position(model, data)
    above = grip[2] - cfg["scene"]["table_top_z"]
    chk.check(0.1 < above < 0.4, "gripper hovers above the table", f"{above:.3f} m above surface")

    # Target marker must not collide with anything (visual only).
    tgt_geom = model.geom("target_geom").id
    touching = any(tgt_geom in (data.contact[i].geom1, data.contact[i].geom2) for i in range(data.ncon))
    chk.check(not touching, "target marker is non-colliding")


def check_random_spawns(cfg, model, data, chk: Checker) -> None:
    print("3. Randomised cube / target spawns")
    rng = np.random.default_rng(cfg["seed"])
    rest_z = sl.cube_rest_height(cfg)
    n = cfg["validation"]["spawn_trials"]
    good = 0
    for _ in range(n):
        sl.reset_to_home(model, data, cfg)
        cube_xy, tgt_xy = sl.sample_cube_and_target(cfg, rng)
        sl.set_cube_pose(model, data, [cube_xy[0], cube_xy[1], rest_z + 0.01])  # drop from 1 cm
        sl.set_target_position(model, data, tgt_xy, cfg)
        simulate(model, data, 1.0)
        c = sl.cube_position(model, data)
        on_table = abs(c[2] - rest_z) < cfg["validation"]["cube_rest_tolerance"] and np.allclose(c[:2], cube_xy, atol=0.005)
        if on_table:
            good += 1
    chk.check(good == n, f"{n}/{n} random spawns settle on the table at the spawn xy", f"{good}/{n}")


def check_grasp_physics(cfg, model, data, chk: Checker) -> None:
    """Smoke test of finger/cube contact + friction. NOT a controller test.

    The cube is placed between the open fingertips and pinned in place while the
    gripper closes (otherwise it would free-fall before the fingers reach it).
    It is then released with the arm holding its pose: if friction/contact are
    sane the cube stays pinched in mid-air; if not, it drops to the table.
    """
    print("4. Gripper-cube contact physics (grasp smoke test)")
    sl.reset_to_home(model, data, cfg)
    simulate(model, data, 0.5)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.body("hand").xmat.flatten())   # align cube faces with finger axes
    pin_pos = sl.gripper_position(model, data)
    sl.set_cube_pose(model, data, pin_pos, quat)

    adr = model.joint(sl.CUBE_JOINT).qposadr[0]
    vadr = model.joint(sl.CUBE_JOINT).dofadr[0]
    data.ctrl[7] = 0.0                                              # close gripper
    for _ in range(int(0.4 / model.opt.timestep)):                  # pinned while fingers close
        data.qpos[adr:adr + 3] = pin_pos
        data.qpos[adr + 3:adr + 7] = quat
        data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_step(model, data)
    simulate(model, data, cfg["validation"]["hold_seconds"])        # cube released: friction must hold it

    cube = sl.cube_position(model, data)
    grip = sl.gripper_position(model, data)
    offset = np.linalg.norm(cube - grip)
    lifted = cube[2] > cfg["scene"]["table_top_z"] + 0.1
    chk.check(lifted, "cube held in mid-air by the closed gripper", f"cube z={cube[2]:.3f}")
    chk.check(offset < 0.01, "cube stays between the fingertips (no slip)", f"offset={offset * 1000:.1f} mm")
    finger = data.qpos[model.joint("finger_joint1").qposadr[0]]
    expected = cfg["scene"]["cube_half_size"]                       # each finger travels to half the cube width
    chk.check(abs(finger - expected) < 0.003, "fingers blocked by the 5 cm cube", f"finger joint={finger * 1000:.1f} mm, expected {expected * 1000:.1f} mm")
    data.ctrl[7] = 255.0                                            # release
    simulate(model, data, 1.5)
    cube = sl.cube_position(model, data)
    chk.check(abs(cube[2] - sl.cube_rest_height(cfg)) < cfg["validation"]["cube_rest_tolerance"], "released cube falls and rests on the table", f"z={cube[2]:.4f}")


def render_snapshot(cfg, model, data, chk: Checker) -> None:
    print("5. Offscreen render")
    out_dir = sl.PROJECT_ROOT / cfg["paths"]["graphs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "phase1_scene.png"
    try:
        import matplotlib.pyplot as plt

        sl.reset_to_home(model, data, cfg)
        simulate(model, data, 1.0)
        renderer = mujoco.Renderer(model, cfg["validation"]["render_height"], cfg["validation"]["render_width"])
        renderer.update_scene(data, camera="overview")
        frame = renderer.render()
        renderer.close()
        plt.imsave(out, frame)
        chk.check(out.is_file() and frame.std() > 5, "overview camera frame rendered", str(out))
    except Exception as exc:  # no GL context etc. - report but do not mask other results
        chk.check(False, "overview camera frame rendered", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the MuJoCo pick-and-place scene (Phase 1).")
    parser.add_argument("--no-render", action="store_true", help="skip the offscreen PNG snapshot")
    parser.add_argument("--config", default=None, help="path to a custom config.yaml")
    args = parser.parse_args()

    cfg = sl.load_config(args.config)
    model, data = sl.load_model(cfg)
    print(f"MuJoCo {mujoco.__version__} | nq={model.nq} nv={model.nv} nu={model.nu} ngeom={model.ngeom}")

    chk = Checker()
    check_model_structure(cfg, model, data, chk)
    check_settle(cfg, model, data, chk)
    check_random_spawns(cfg, model, data, chk)
    check_grasp_physics(cfg, model, data, chk)
    if not args.no_render:
        render_snapshot(cfg, model, data, chk)

    passed = sum(ok for ok, _ in chk.results)
    print(f"\n{passed}/{len(chk.results)} checks passed")
    return 0 if chk.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
