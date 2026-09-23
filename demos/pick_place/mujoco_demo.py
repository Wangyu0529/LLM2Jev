"""One-stage LLM2Jev control of a physical Panda pick-and-place scene."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

try:
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
except ImportError as error:  # pragma: no cover - requires optional demo dependency
    raise SystemExit(
        "Install the MuJoCo demo dependencies: pip install mujoco imageio imageio-ffmpeg pillow"
    ) from error


AXIS_CHOICES = ("negative", "zero", "positive")
HOME = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
STEP_M = 0.01
PHYSICS_DT = 0.002

ACTION_CHOICES = tuple(
    [f"move_{axis}_{value}" for axis in "xyz" for value in ("negative", "positive")]
    + ["open_gripper", "close_gripper"]
)

ACTION_RULES = """Select exactly one immediate atomic action from CURRENT measured facts.
+X/+Y/+Z are robot-base axes and +Z is up. Choose an action listed in state.available_actions.
Every unlisted action is invalid. A move translates the TCP by 1 cm and keeps the current gripper command.
"""


def direction(value: float, tolerance: float = 0.006) -> str:
    return "positive" if value > tolerance else "negative" if value < -tolerance else "zero"


def relation(delta: np.ndarray) -> dict[str, object]:
    return {
        "delta_m": delta.tolist(),
        "directions": dict(zip("xyz", map(direction, delta))),
        "xy_aligned": bool(np.max(abs(delta[:2])) <= 0.006),
    }


def action_criteria() -> dict[str, str]:
    return {
        name: f"Select exactly when state.available_actions contains {name}."
        for name in ACTION_CHOICES
    }


class JevClient:
    def __init__(self, api_base: str, model: str):
        url = urlsplit(api_base)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("--api-base must be an http(s) URL")
        connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        self.connection = connection_type(url.hostname, url.port, timeout=120)
        self.path = f"{url.path.rstrip('/')}/v1/systemone"
        self.model = model

    def choose(self, state: dict[str, object]) -> dict[str, str]:
        questions = {
            "action": {
                "type": "choice",
                "criteria": action_criteria(),
                "instructions": {
                    "rules": ACTION_RULES,
                    "question": "Which single atomic action should the robot execute now?",
                },
            }
        }
        decision_state = {
            "task": state["task"],
            "available_actions": state["available_actions"],
        }
        body = json.dumps(
            {"state": decision_state, "model": self.model, "questions": questions},
            separators=(",", ":"),
        )
        self.connection.request("POST", self.path, body,
                                {"Content-Type": "application/json", "Accept": "application/json"})
        response = self.connection.getresponse()
        payload = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"LLM2Jev HTTP {response.status}: {payload}")
        choice = payload.get("answers", {}).get("action", {}).get("choice")
        if choice not in ACTION_CHOICES:
            raise RuntimeError(f"model selected invalid action: {choice!r}")
        available = state.get("available_actions", [])
        if choice not in available:
            raise RuntimeError(f"model selected unavailable action {choice!r}; available: {available}")
        action = {"x": "zero", "y": "zero", "z": "zero", "gripper": "hold"}
        if choice.startswith("move_"):
            _, axis, value = choice.split("_")
            action[axis] = value
        else:
            action["gripper"] = choice.removesuffix("_gripper")
        action["primitive"] = choice
        return action

    def close(self) -> None:
        self.connection.close()


class PandaPickPlace:
    def __init__(self, panda_dir: Path):
        self.model = mujoco.MjModel.from_xml_string(self._scene_xml(panda_dir))
        self.data = mujoco.MjData(self.model)
        self.site_id = self.model.site("tcp").id
        self.arm_q = np.array([self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_v = np.array([self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.arm_act = np.array([self.model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.limits = np.array([self.model.joint(f"joint{i}").range for i in range(1, 8)])
        self.finger_q = [self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.grip_act = self.model.actuator("actuator8").id
        self.cube_id = self.model.body("cube").id
        self.cube_geom = self.model.geom("cube_geom").id
        self.cube_q = self.model.joint("cube_free").qposadr[0]
        self.finger_bodies = [self.model.body(f"{side}_finger").id for side in ("left", "right")]
        self.target_mocap = self.model.body("target").mocapid[0]
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.reference_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.gripper_target = "open"
        self.target_pos = np.zeros(3)
        self.held = self.released = False
        self.stable_steps = 0
        self.next_frame_time = 0.0
        self.history: list[dict[str, object]] = []
        self.last_result = "reset"
        self.overlay_font = ImageFont.load_default(size=24)

    @staticmethod
    def _scene_xml(panda_dir: Path) -> str:
        manifest = panda_dir / "manifest.json"
        if not manifest.is_file() or not (panda_dir / "panda.xml").is_file():
            raise RuntimeError("Panda assets missing; run fetch_panda.py in this demo directory")
        for relative, digest in json.loads(manifest.read_text())["files"].items():
            if hashlib.sha256((panda_dir / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Panda asset checksum mismatch: {relative}")
        root = ET.parse(panda_dir / "panda.xml").getroot()
        root.find("compiler").set("meshdir", str(panda_dir / "assets"))
        option = root.find("option")
        for key, value in {"timestep": str(PHYSICS_DT), "gravity": "0 0 -9.81", "iterations": "100",
                           "cone": "elliptic", "impratio": "10", "noslip_iterations": "5"}.items():
            option.set(key, value)
        root.remove(root.find("keyframe"))
        gripper = root.find("actuator/general[@name='actuator8']")
        gripper.set("gainprm", str(.04 * 1000 / 255)); gripper.set("biasprm", "0 -1000 -40")
        gripper.set("forcerange", "-40 40")
        world = root.find("worldbody")
        for body in world.iter("body"):
            body.set("gravcomp", "1")
        hand = world.find(".//body[@name='hand']")
        ET.SubElement(hand, "site", name="tcp", pos="0 0 0.107", size="0.004")
        for side in ("left", "right"):
            finger = world.find(f".//body[@name='{side}_finger']")
            for index, geom in enumerate(finger.findall("geom")):
                if geom.get("class") != "visual":
                    geom.set("name", f"{side}_finger_geom_{index}")
                    geom.set("friction", "1 0.005 0.0001"); geom.set("condim", "4")
                    geom.set("solref", "0.005 1"); geom.set("solimp", "0.95 0.99 0.001")
        ET.SubElement(world, "geom", name="table", type="box", size=".45 .4 .025",
                      pos=".45 0 -.025", rgba=".12 .15 .20 1", friction="1 .005 .0001")
        cube = ET.SubElement(world, "body", name="cube", pos=".50 -.10 .021")
        ET.SubElement(cube, "freejoint", name="cube_free")
        ET.SubElement(cube, "geom", name="cube_geom", type="box", size=".02 .02 .02", mass=".05",
                      rgba="1 .55 .12 1", friction="1 .005 .0001", condim="4")
        target = ET.SubElement(world, "body", name="target", mocap="true", pos=".50 .12 .0005")
        ET.SubElement(target, "geom", name="target_marker", type="box", size=".06 .06 .0005",
                      rgba=".05 .70 .80 1", contype="0", conaffinity="0")
        ET.SubElement(world, "camera", name="overview", pos="1.08 -1.10 .96",
                      xyaxes=".879 .477 0 -.267 .492 .829", fovy="57")
        ET.SubElement(world, "light", pos=".4 -.5 1.5", dir="0 0 -1", diffuse=".8 .8 .8")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth="1280", offheight="720")
        return ET.tostring(root, encoding="unicode")

    def reset(self) -> dict[str, object]:
        mujoco.mj_resetData(self.model, self.data)
        self.gripper_target = "open"; self.held = self.released = False; self.stable_steps = 0
        self.next_frame_time = 0.0
        self.history.clear(); self.last_result = "reset"
        self.data.qpos[self.arm_q] = HOME; self.data.qpos[self.finger_q] = .04
        self.data.ctrl[self.arm_act] = HOME; self.data.ctrl[self.grip_act] = 255
        self.data.qpos[self.cube_q:self.cube_q + 7] = [.50, -.10, .021, 1, 0, 0, 0]
        self.target_pos = np.array([.50, .12, 0.0]); self.data.mocap_pos[self.target_mocap] = self.target_pos + [.0, .0, .0005]
        mujoco.mj_forward(self.model, self.data); self.reference_quat = self._quat()
        return self.state()

    def _quat(self) -> np.ndarray:
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id]); return quat

    def _contacts(self) -> list[bool]:
        forces = [0.0, 0.0]
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.cube_geom not in (contact.geom1, contact.geom2): continue
            other = contact.geom2 if contact.geom1 == self.cube_geom else contact.geom1
            body = self.model.geom_bodyid[other]
            if body not in self.finger_bodies: continue
            force = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, index, force)
            forces[self.finger_bodies.index(body)] += max(0.0, float(force[0]))
        return [value > .1 for value in forces]

    def state(self) -> dict[str, object]:
        eef = self.data.site_xpos[self.site_id].copy(); cube = self.data.xpos[self.cube_id].copy()
        xy_delta = self.target_pos[:2] - cube[:2]
        grasp_delta = cube + [0.0, 0.0, .008] - eef
        placement = self.target_pos.copy()
        placement[2] = .025 - (cube[2] - eef[2])
        placement_delta = placement - eef
        contacts = self._contacts(); self.held = all(contacts) and .005 < self.data.qpos[self.finger_q].sum() < .065
        inside = np.max(abs(xy_delta)) <= .06; resting = abs(cube[2] - .02) <= .006
        cube_v = self.model.joint("cube_free").dofadr[0]
        speed = np.linalg.norm(self.data.qvel[cube_v:cube_v + 3])
        self.stable_steps = self.stable_steps + 1 if self.released and inside and resting and speed < .02 else 0
        grasp_relation = {**relation(grasp_delta), "z_aligned": bool(abs(grasp_delta[2]) <= .006)}
        target_relation = relation(np.append(xy_delta, 0.0))
        placement_relation = {**relation(placement_delta), "z_aligned": bool(abs(placement_delta[2]) <= .006)}
        clear = bool(self.held and cube[2] - .02 >= .05)
        if self.released:
            available_actions = ["open_gripper"]
        elif self.held and not inside:
            available_actions = (["move_z_positive"] if not clear else [
                f"move_{axis}_{target_relation['directions'][axis]}" for axis in "xy"
                if target_relation["directions"][axis] != "zero"
            ])
        elif self.held:
            z_direction = placement_relation["directions"]["z"]
            available_actions = ([f"move_z_{z_direction}"] if z_direction != "zero" else ["open_gripper"])
        elif self.gripper_target == "close":
            available_actions = ["open_gripper"]
        else:
            available_actions = [
                f"move_{axis}_{grasp_relation['directions'][axis]}" for axis in "xy"
                if grasp_relation["directions"][axis] != "zero"
            ]
            if not available_actions:
                z_direction = grasp_relation["directions"]["z"]
                available_actions = ([f"move_z_{z_direction}"] if z_direction != "zero" else ["close_gripper"])
        return {"task": "Pick up the amber cube, carry it to the cyan target region, and release it on the table.",
                "task_id": "pick_place", "robot": {"tcp_position": eef.tolist(), "gripper_target": self.gripper_target,
                "finger_object_contacts": contacts, "held_object": "cube" if self.held else None},
                "objects": [{"id": "cube", "label": "amber cube", "position": cube.tolist()}],
                "target": {"id": "target", "label": "cyan target region", "position": self.target_pos.tolist()},
                "relations": {"grasp_tcp_from_tcp": grasp_relation,
                "target_from_cube": target_relation,
                "placement_tcp_from_tcp": placement_relation,
                "cube_clear_of_table_for_transport": clear,
                "cube_inside_target_xy": bool(inside), "cube_resting_height": bool(resting)},
                "available_actions": available_actions,
                "recent_actions": self.history[-4:], "frame": "robot_base", "units": "m",
                "quaternion_order": "wxyz"}

    def _control(self, target: np.ndarray) -> None:
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
        current = self._quat(); inverse = current * [1, -1, -1, -1]; difference = np.zeros(4)
        mujoco.mju_mulQuat(difference, self.reference_quat, inverse)
        if difference[0] < 0: difference *= -1
        orientation = np.zeros(3); mujoco.mju_quat2Vel(orientation, difference, 1.0)
        error = np.concatenate([target - self.data.site_xpos[self.site_id], orientation])
        jac = np.vstack([self.jacp[:, self.arm_v], self.jacr[:, self.arm_v]])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + .01**2 * np.eye(6), error); q = self.data.qpos[self.arm_q]
        self.data.ctrl[self.arm_act] = np.clip(q + np.clip(dq, -.05, .05), self.limits[:, 0] + .005, self.limits[:, 1] - .005)
        self.data.ctrl[self.grip_act] = 255 if self.gripper_target == "open" else 0

    def execute(self, action: dict[str, str], renderer, writer, step: int, model_ms: float) -> bool:
        state = self.state(); current = np.asarray(state["robot"]["tcp_position"])
        vector = np.array([AXIS_CHOICES.index(action[axis]) - 1 for axis in "xyz"], dtype=float)
        delta = vector * (STEP_M / np.linalg.norm(vector)) if vector.any() else vector; target = current + delta
        if np.any(target < [.2, -.35, .018]) or np.any(target > [.75, .35, .65]):
            self._remember(action, "workspace_rejected", np.zeros(3))
            return False
        if action["gripper"] != "hold": self.gripper_target = action["gripper"]
        for tick in range(round(.5 / PHYSICS_DT)):
            self._control(target); mujoco.mj_step(self.model, self.data); mujoco.mj_forward(self.model, self.data)
            contacts = self._contacts()
            if action["gripper"] == "close" and all(contacts): self.held = True
            if action["gripper"] == "open" and self.held: self.held, self.released = False, True
            self.state()
            if self.data.time + 1e-9 >= self.next_frame_time:
                renderer.update_scene(self.data, camera="overview")
                writer.append_data(self._annotate_frame(renderer.render(), step, model_ms))
                self.next_frame_time += 1 / 30
            if tick >= round(.2 / PHYSICS_DT) and np.linalg.norm(self.data.site_xpos[self.site_id] - target) < .004: break
            if self.stable_steps >= round(.5 / PHYSICS_DT):
                self._remember(action, "task_success", self.data.site_xpos[self.site_id] - current)
                return True
        self._remember(action, "executed", self.data.site_xpos[self.site_id] - current)
        return False

    def _annotate_frame(self, frame: np.ndarray, step: int, model_ms: float) -> np.ndarray:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        label = f"Step {step:03d}  |  Model latency {model_ms:.1f} ms"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=self.overlay_font)
        draw.rounded_rectangle(
            (24, 24, 48 + right - left, 40 + bottom - top),
            radius=6,
            fill=(0, 0, 0, 180),
        )
        draw.text((36, 32), label, font=self.overlay_font, fill=(255, 255, 255, 255))
        return np.asarray(image)

    def _remember(self, action: dict[str, str], result: str, measured: np.ndarray) -> None:
        self.last_result = result
        self.history.append({
            "action": dict(action),
            "result": result,
            "delta_measured_m": measured.tolist(),
            "gripper_target": self.gripper_target,
        })
        self.history = self.history[-4:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model name exposed by llm2jev-serve")
    parser.add_argument("--api-base", default="http://127.0.0.1:30000")
    parser.add_argument("--panda-dir", type=Path, default=Path(__file__).with_name("panda"))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--video", type=Path, default=Path("runs/robojev-pick-place.mp4"))
    args = parser.parse_args()
    import imageio.v2 as imageio
    arm = PandaPickPlace(args.panda_dir); client = JevClient(args.api_base, args.model)
    renderer = mujoco.Renderer(arm.model, height=720, width=1280); args.video.parent.mkdir(parents=True, exist_ok=True)
    success = False
    rejected_streak = 0
    try:
        arm.reset()
        with imageio.get_writer(args.video, fps=30, codec="libx264", macro_block_size=16) as writer:
            for step in range(args.steps):
                state = arm.state()
                started = time.perf_counter()
                action = client.choose(state)
                model_ms = (time.perf_counter() - started) * 1000
                success = arm.execute(action, renderer, writer, step + 1, model_ms)
                rejected_streak = rejected_streak + 1 if arm.last_result == "workspace_rejected" else 0
                print(
                    f"step={step + 1} model_ms={model_ms:.1f} "
                    f"action={json.dumps(action)} result={arm.last_result}",
                    flush=True,
                )
                if success: break
                if rejected_streak >= 3:
                    raise RuntimeError("model repeated a workspace-rejected action three times")
    finally:
        client.close(); renderer.close()
    if not success: raise SystemExit(f"pick & place did not settle within {args.steps} decisions; video written to {args.video}")
    print(f"SUCCESS; wrote {args.video}")


if __name__ == "__main__":
    main()
