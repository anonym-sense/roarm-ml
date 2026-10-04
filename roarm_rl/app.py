"""Interactive PyBullet GUI for the RoArm-M2 -- keyboard controlled.

PyBullet's native debug-panel sliders turned out to be unreliable to drag
precisely (values snapping back when the mouse left the tiny hit-box), so
control here is entirely keyboard-based: hold a key to move continuously,
tap a key to toggle a mode. Current state is shown as on-screen debug text
(which, unlike sliders, *can* be updated from code every frame).

Controls
--------
M            toggle Joint mode <-> IK-XYZ mode
LEFT/RIGHT   Joint mode: base   | IK mode: target X
UP/DOWN      Joint mode: shoulder | IK mode: target Y
PAGE UP/DOWN Joint mode: elbow  | IK mode: target Z
,  .         gripper close / open
P            toggle "Mirror to Real Robot"
H            send HOME pose (sim + hardware if mirroring)
T            toggle hardware torque
ESC          quit
Mouse: click+drag the arm itself to move the hand in 3D (switches to IK
mode automatically). Drag happens in the plane facing the camera at the
depth you grabbed -- orbit/zoom the camera first to reach a different
depth, then grab again.

If a RoArmHardware is attached and mirroring is on, every joint target is
throttled and mirrored to the physical arm via roarm_sdk, so whatever the
sim is doing is replayed on hardware.
"""

import time

import pybullet as p

from roarm_rl.sim import RoArmSim
from roarm_rl.picking import camera_ray, intersect_plane

MIRROR_MIN_INTERVAL = 0.08  # seconds; avoid flooding the servo bus
MIRROR_EPSILON = 0.01  # radians; skip resend if target barely changed

JOINT_STEP = 0.008  # rad / frame while a joint key is held
IK_STEP = 0.0025  # m / frame while an IK key is held

LEFT, RIGHT = p.B3G_LEFT_ARROW, p.B3G_RIGHT_ARROW
UP, DOWN = p.B3G_UP_ARROW, p.B3G_DOWN_ARROW
PGUP, PGDN = p.B3G_PAGE_UP, p.B3G_PAGE_DOWN
COMMA, PERIOD = ord(","), ord(".")
KEY_M, KEY_P, KEY_H, KEY_T = ord("m"), ord("p"), ord("h"), ord("t")
ESC = 27

MOUSE_MOVE_EVENT = 1
MOUSE_BUTTON_EVENT = 2
LEFT_BUTTON = 0


def _held(keys, code):
    return code in keys and (keys[code] & p.KEY_IS_DOWN)


def _tapped(keys, code):
    return code in keys and (keys[code] & p.KEY_WAS_TRIGGERED)


class _MouseDrag:
    """Click-and-drag picking: grab the arm, drag in the camera-facing
    plane at the depth you grabbed, release to let go."""

    def __init__(self, sim):
        self.sim = sim
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self.active = False
        self.plane_point = None
        self.plane_normal = None

    def update(self, mouse_events, client_id):
        target = None
        for e in mouse_events:
            event_type, x, y, button, state = e[0], e[1], e[2], e[3], e[4]
            self.mouse_x, self.mouse_y = x, y
            if event_type != MOUSE_BUTTON_EVENT or button != LEFT_BUTTON:
                continue
            if state & p.KEY_WAS_TRIGGERED:
                self._try_start(client_id)
            elif state & p.KEY_WAS_RELEASED:
                self.active = False

        if self.active:
            ray_from, ray_to, _ = camera_ray(self.mouse_x, self.mouse_y, client_id)
            target = intersect_plane(ray_from, ray_to, self.plane_point, self.plane_normal)
        return target

    def _try_start(self, client_id):
        ray_from, ray_to, cam_fwd = camera_ray(self.mouse_x, self.mouse_y, client_id)
        hits = p.rayTest(ray_from, ray_to, physicsClientId=client_id)
        hit = hits[0] if hits else None
        if hit and hit[0] == self.sim.robot and hit[1] >= 0:
            self.active = True
            self.plane_point = list(hit[3])
            self.plane_normal = list(cam_fwd)


def run(hardware=None):
    sim = RoArmSim(gui=True)
    limits = sim.joint_limits

    joints = list(sim.home_radians)
    ik_target = [0.2, 0.0, 0.2]
    mode_ik = False
    mirroring = False

    last_mirror_sent = None
    last_mirror_time = 0.0
    status_text_id = None
    drag = _MouseDrag(sim)

    legend = (
        "M: joint/IK mode   P: mirror on/off   H: home   T: torque   Esc: quit\n"
        "Joint mode:  <-/-> base   up/down shoulder   PgUp/PgDn elbow   ,/. gripper\n"
        "IK mode:     <-/-> X      up/down Y          PgUp/PgDn Z       ,/. gripper\n"
        "Mouse: click+drag the arm to move the hand (drags on the plane you grabbed)"
    )
    p.addUserDebugText(legend, [-0.25, 0.0, 0.62], textColorRGB=[0.85, 0.9, 1.0], textSize=1.0)
    marker = sim.add_target_marker()

    print(__doc__)
    if hardware is not None and hardware.connected:
        print(f"[hardware] connected via {hardware.mode}; press P to start mirroring.")
    elif hardware is not None:
        print("[hardware] requested but not connected -- mirroring disabled.")

    try:
        while p.isConnected():
            keys = p.getKeyboardEvents()

            if _tapped(keys, KEY_M):
                mode_ik = not mode_ik
                print(f"[mode] {'IK-XYZ' if mode_ik else 'Joint'}")
            if _tapped(keys, KEY_P):
                mirroring = not mirroring
                print(f"[mirror] {'ON' if mirroring else 'OFF'}")
            if _tapped(keys, KEY_H):
                joints = list(sim.home_radians)
                ik_target = list(sim.get_ee_pose()[0])
                print("[action] home")
                if hardware is not None and hardware.connected:
                    try:
                        hardware.set_joint_targets(joints)
                        print("[hardware] sent home pose")
                    except Exception as e:
                        print(f"[hardware] home failed: {e}")
            if _tapped(keys, KEY_T) and hardware is not None and hardware.connected:
                hardware._torque_on = not getattr(hardware, "_torque_on", True)
                try:
                    hardware.set_torque(hardware._torque_on)
                    print(f"[hardware] torque {'ON' if hardware._torque_on else 'OFF'}")
                except Exception as e:
                    print(f"[hardware] torque toggle failed: {e}")
            if ESC in keys and (keys[ESC] & p.KEY_WAS_TRIGGERED):
                break

            drag_target = drag.update(p.getMouseEvents(), sim.client_id)
            if drag_target is not None:
                if not mode_ik:
                    mode_ik = True
                    print("[mode] IK-XYZ (grabbed by mouse)")
                ik_target = drag_target
                joints = sim.solve_ik(ik_target)
            elif mode_ik:
                if _held(keys, LEFT):
                    ik_target[0] -= IK_STEP
                if _held(keys, RIGHT):
                    ik_target[0] += IK_STEP
                if _held(keys, UP):
                    ik_target[1] += IK_STEP
                if _held(keys, DOWN):
                    ik_target[1] -= IK_STEP
                if _held(keys, PGUP):
                    ik_target[2] += IK_STEP
                if _held(keys, PGDN):
                    ik_target[2] -= IK_STEP
                joints = sim.solve_ik(ik_target)
            else:
                if _held(keys, LEFT):
                    joints[0] -= JOINT_STEP
                if _held(keys, RIGHT):
                    joints[0] += JOINT_STEP
                if _held(keys, UP):
                    joints[1] += JOINT_STEP
                if _held(keys, DOWN):
                    joints[1] -= JOINT_STEP
                if _held(keys, PGUP):
                    joints[2] += JOINT_STEP
                if _held(keys, PGDN):
                    joints[2] -= JOINT_STEP

            if _held(keys, PERIOD):
                joints[3] += JOINT_STEP
            if _held(keys, COMMA):
                joints[3] -= JOINT_STEP

            joints = sim.set_joint_targets(joints)
            sim.step()

            if mode_ik:
                p.resetBasePositionAndOrientation(marker, ik_target, [0, 0, 0, 1],
                                                  physicsClientId=sim.client_id)
            else:
                p.resetBasePositionAndOrientation(marker, [0, 0, -5], [0, 0, 0, 1],
                                                  physicsClientId=sim.client_id)

            status = (
                f"{'IK-XYZ' if mode_ik else 'JOINT'} MODE    MIRROR {'ON ' if mirroring else 'OFF'}\n"
                f"base {joints[0]:+.2f}   shoulder {joints[1]:+.2f}\n"
                f"elbow {joints[2]:+.2f}   gripper {joints[3]:+.2f}"
            )
            status_text_id = p.addUserDebugText(
                status, [-0.25, 0.0, 0.42],
                textColorRGB=[0.45, 1.0, 0.6] if mirroring else [0.9, 0.9, 0.9],
                textSize=1.1,
                replaceItemUniqueId=status_text_id if status_text_id is not None else -1,
            )

            if hardware is not None and hardware.connected:
                now = time.time()
                changed = (
                    last_mirror_sent is None
                    or max(abs(a - b) for a, b in zip(joints, last_mirror_sent)) > MIRROR_EPSILON
                )
                if mirroring and changed and (now - last_mirror_time) > MIRROR_MIN_INTERVAL:
                    try:
                        hardware.set_joint_targets(joints)
                        last_mirror_sent = joints
                    except Exception as e:
                        print(f"[hardware] mirror failed: {e}")
                    last_mirror_time = now

            time.sleep(sim.timestep)
    except p.error:
        pass  # GUI window closed
    finally:
        sim.close()
        if hardware is not None and hardware.connected:
            hardware.disconnect()
