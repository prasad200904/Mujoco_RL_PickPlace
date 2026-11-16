"""Scene loading and low-level scene helpers shared by every phase.

Everything that touches raw MuJoCo model/data layout lives here so later modules
(controller, perception, PickPlaceEnv) never hard-code body/geom names themselves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Names defined in assets/*.xml
CUBE_BODY = "cube"
CUBE_JOINT = "cube_joint"
CUBE_GEOM = "cube_geom"
TARGET_BODY = "target"
GRIP_SITE = "grip_site"
TABLE_GEOM = "table_top"
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read configs/config.yaml (or a custom path) into a dict."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_model(config: dict[str, Any]) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the scene XML and return (model, data) reset to the home keyframe."""
    xml_path = PROJECT_ROOT / config["paths"]["scene_xml"]
    if not xml_path.is_file():
        raise FileNotFoundError(f"Scene XML not found: {xml_path}")
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:  # MuJoCo raises ValueError with the XML error text
        raise RuntimeError(f"Failed to compile {xml_path}: {exc}") from exc
    data = mujoco.MjData(model)
    reset_to_home(model, data, config)
    return model, data


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData, config: dict[str, Any]) -> None:
    """Reset simulation state to the named home keyframe and recompute kinematics."""
    key_name = config["scene"]["home_keyframe"]
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id < 0:
        raise KeyError(f"Keyframe '{key_name}' not found in model")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


# ----------------------------------------------------------------- state access
def cube_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Cube centre [x, y, z] from the simulation state (world frame)."""
    return data.body(CUBE_BODY).xpos.copy()


def gripper_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Grasp-centre [x, y, z] between the fingertip pads (world frame)."""
    return data.site(GRIP_SITE).xpos.copy()


def target_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Target marker centre [x, y, z] on the table surface (world frame)."""
    return data.body(TARGET_BODY).xpos.copy()


# ----------------------------------------------------------------- state setting
def set_cube_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position: np.ndarray,
    quat: np.ndarray | None = None,
) -> None:
    """Teleport the cube (zero velocity). Works for any free-joint cube."""
    jnt = model.joint(CUBE_JOINT)
    adr, vadr = jnt.qposadr[0], jnt.dofadr[0]
    data.qpos[adr:adr + 3] = position
    data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0] if quat is None else quat
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def set_target_position(model: mujoco.MjModel, data: mujoco.MjData, xy: np.ndarray, config: dict[str, Any]) -> None:
    """Move the (mocap) target marker to table position xy; z stays on the table surface."""
    mocap_id = model.body(TARGET_BODY).mocapid[0]
    marker_half_height = float(model.geom("target_geom").size[2])
    z = config["scene"]["table_top_z"] + marker_half_height
    data.mocap_pos[mocap_id] = [xy[0], xy[1], z]
    mujoco.mj_forward(model, data)


def cube_rest_height(config: dict[str, Any]) -> float:
    """z of the cube centre when resting on the table."""
    return config["scene"]["table_top_z"] + config["scene"]["cube_half_size"]


def sample_cube_and_target(config: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random cube / target xy positions that respect the configured minimum separation."""
    sc = config["scene"]
    for _ in range(1000):
        cube_xy = np.array([rng.uniform(*sc["cube_spawn"]["x"]), rng.uniform(*sc["cube_spawn"]["y"])])
        tgt_xy = np.array([rng.uniform(*sc["target_spawn"]["x"]), rng.uniform(*sc["target_spawn"]["y"])])
        if np.linalg.norm(cube_xy - tgt_xy) >= sc["min_cube_target_distance"]:
            return cube_xy, tgt_xy
    raise RuntimeError("Could not sample cube/target with the configured minimum distance; check config ranges.")


def cube_table_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """True if the cube geom is currently in contact with the table top."""
    cube_g = model.geom(CUBE_GEOM).id
    table_g = model.geom(TABLE_GEOM).id
    for i in range(data.ncon):
        c = data.contact[i]
        if {c.geom1, c.geom2} == {cube_g, table_g}:
            return True
    return False


def finger_cube_contact(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, bool]:
    """Returns (left_finger_contact, right_finger_contact) with the cube."""
    cube_g = model.geom(CUBE_GEOM).id
    left_id = model.body("left_finger").id
    right_id = model.body("right_finger").id

    left_touch, right_touch = False, False
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == cube_g or c.geom2 == cube_g:
            other_geom = c.geom2 if c.geom1 == cube_g else c.geom1
            other_body = model.geom_bodyid[other_geom]
            if other_body == left_id:
                left_touch = True
            elif other_body == right_id:
                right_touch = True
    return left_touch, right_touch
