"""Keyframe gestures for the RoArm-M2, animated with 12-principles-style timing.

Each gesture is a list of (duration_s, [base, shoulder, elbow, gripper]) keyframes
in radians. Motion is a Catmull-Rom spline through the keyframes, so velocity
stays continuous across each pose (no stop-and-go between keys). Keyframes
carry the animation principles:

  - anticipation: a small wind-up pose just before a big move
  - overshoot and settle: the pose goes past its target and comes back
  - follow-through: the gripper trails the wrist a beat later

    python -m roarm_rl.gesture dance             # preview in the simulator only
    python -m roarm_rl.gesture dance --hardware COM9   # run on the real arm over USB
    python -m roarm_rl.gesture --list            # every name, including roarm_rl.library variants
"""

import argparse
import time
from collections import deque

GESTURES = {
    # Disney-style hop-and-wiggle: anticipation dip, springy overshoot, gripper follow-through.
    "dance": [
        (0.8, [0.00, 0.00, 1.57, 0.00]),    # rest
        (0.4, [0.00, -0.08, 1.45, 0.00]),   # anticipation: crouch
        (0.5, [0.00, 0.22, 1.90, 0.30]),    # launch up
        (0.4, [0.30, 0.30, 2.05, 0.55]),    # overshoot the arc to the right
        (0.5, [0.15, 0.12, 1.75, 0.35]),    # settle back
        (0.4, [-0.30, 0.18, 1.95, 0.60]),   # swing left, gripper trails
        (0.5, [-0.20, 0.08, 1.65, 0.20]),   # settle
        (0.8, [0.00, 0.00, 1.57, 0.00]),    # return to rest
    ],
    # Gentle wave: arcs (joints move on offset timing) with a slight overshoot.
    "wave": [
        (0.8, [0.00, 0.00, 1.57, 0.00]),
        (0.5, [0.25, 0.15, 1.75, 0.20]),
        (0.5, [0.45, 0.20, 1.80, 0.50]),    # overshoot
        (0.4, [0.40, 0.16, 1.72, 0.30]),    # settle
        (0.5, [-0.25, 0.10, 1.65, 0.10]),
        (0.5, [-0.45, 0.15, 1.75, 0.55]),   # overshoot
        (0.4, [-0.38, 0.12, 1.70, 0.30]),   # settle
        (0.8, [0.00, 0.00, 1.57, 0.00]),
    ],
}

# Range gestures may use, well inside the firmware limits (see
# roarm_rl.sim._FIRMWARE_LIMITS_M2). The shoulder-forward / elbow-down corner
# is what sets it: at (0.55, 2.1) the gripper is still about 5 cm above the
# table, so any pose inside the box clears the surface the arm stands on.
SAFE_BOUNDS = [(-1.2, 1.2), (-0.6, 0.55), (0.7, 2.1), (0.0, 1.2)]
COMMAND_HZ = 25
LEAD_IN_SPEED = 0.8  # rad/s when moving from the current pose to a gesture's first pose


def _check_bounds(keyframes):
    names = ["base", "shoulder", "elbow", "gripper"]
    for _, pose in keyframes:
        for value, (lo, hi), name in zip(pose, SAFE_BOUNDS, names):
            if not lo <= value <= hi:
                raise ValueError(f"{name}={value} outside safe range {lo}..{hi}")


def _catmull_rom(p0, p1, p2, p3, s):
    s2, s3 = s * s, s * s * s
    return [
        0.5 * (2 * b + (-a + c) * s + (2 * a - 5 * b + 4 * c - d) * s2 + (-a + 3 * b - 3 * c + d) * s3)
        for a, b, c, d in zip(p0, p1, p2, p3)
    ]


def duration(keyframes):
    """Total play time in seconds (the first keyframe's duration is unused)."""
    return sum(dur for dur, _ in keyframes[1:])


def pose_at(keyframes, t):
    """Pose at time t on a Catmull-Rom spline through the keyframes."""
    poses = [kf[1] for kf in keyframes]
    if len(poses) == 1:
        return [min(max(v, lo), hi) for v, (lo, hi) in zip(poses[0], SAFE_BOUNDS)]
    start = 0.0
    i = 1  # poses index of segment end point
    for i, (dur, _) in enumerate(keyframes[1:], start=1):
        if t <= start + dur or i == len(poses) - 1:
            break
        start += dur
    dur = keyframes[i][0]
    s = min(max((t - start) / dur, 0.0), 1.0) if dur > 0 else 1.0
    p0 = poses[max(i - 2, 0)]
    p1 = poses[i - 1]
    p2 = poses[i]
    p3 = poses[min(i + 1, len(poses) - 1)]
    pose = _catmull_rom(p0, p1, p2, p3, s)
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(pose, SAFE_BOUNDS)]


def sample_times(keyframes, hz=COMMAND_HZ):
    """Yield (t, pose) at hz along a Catmull-Rom spline through the keyframes."""
    total = duration(keyframes)
    dt = 1.0 / hz
    t = 0.0
    while t <= total + 1e-9:
        yield t, pose_at(keyframes, t)
        t += dt


class GesturePlayer:
    """Plays queued gestures in real time; call update() once per frame."""

    def __init__(self):
        self._queue = deque()
        self._traj = None
        self._t0 = 0.0
        self.label = None  # what is playing now, or None when idle

    @property
    def busy(self):
        return self._traj is not None or bool(self._queue)

    def enqueue(self, label, keyframes):
        _check_bounds(keyframes)
        self._queue.append((label, keyframes))

    def cancel(self):
        self._queue.clear()
        self._traj = None
        self.label = None

    def update(self, current):
        """Pose to command this frame, or None when nothing is playing."""
        now = time.monotonic()
        if self._traj is None:
            if not self._queue:
                return None
            self.label, keyframes = self._queue.popleft()
            first = keyframes[0][1]
            gap = max(abs(a - b) for a, b in zip(current, first))
            lead_in = max(0.05, gap / LEAD_IN_SPEED)
            self._traj = [(0.0, list(current)), (lead_in, first)] + list(keyframes[1:])
            self._t0 = now
        t = now - self._t0
        pose = pose_at(self._traj, t)
        if t >= duration(self._traj):
            self._traj = None
            if not self._queue:
                self.label = None
        return pose


def lookup(name):
    """Keyframes for a hand-written gesture or any roarm_rl.library variant."""
    if name in GESTURES:
        return GESTURES[name]
    from roarm_rl.library import catalog

    variants = catalog()
    if name not in variants:
        raise KeyError(f"unknown gesture '{name}' (try --list)")
    return variants[name]


def preview(name):
    kf = lookup(name)
    _check_bounds(kf)
    from roarm_rl.sim import RoArmSim

    sim = RoArmSim(gui=True)
    try:
        for _, pose in sample_times(kf):
            sim.set_joint_targets(pose, instant=True)
            sim.step()
            time.sleep(1.0 / COMMAND_HZ)
        print(f"preview of '{name}' finished")
    finally:
        sim.close()


def play_on_hardware(name, port, speed=600, acc=40):
    kf = lookup(name)
    _check_bounds(kf)
    from roarm_sdk.roarm import roarm

    arm = roarm(roarm_type="roarm_m2", port=port, baudrate=115200, timeout=0.5)
    try:
        arm.torque_set(1)
        time.sleep(0.3)
        current = arm.joints_radian_get()
        if not isinstance(current, list):
            raise RuntimeError(f"could not read current pose: {current}")
        print("current pose:", [round(v, 3) for v in current])

        lead_in = [(0.0, current), (1.5, kf[0][1])]
        t0 = time.monotonic()
        for t, pose in sample_times(lead_in):
            wait = t0 + t - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            arm.joints_radian_ctrl(radians=pose, speed=speed, acc=acc)

        t0 = time.monotonic()
        for t, pose in sample_times(kf):
            wait = t0 + t - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            arm.joints_radian_ctrl(radians=pose, speed=speed, acc=acc)
        print(f"'{name}' finished")
    finally:
        arm.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="gesture name, e.g. dance or nod_fast_big")
    ap.add_argument("--list", action="store_true", help="print every gesture name and exit")
    ap.add_argument("--hardware", metavar="PORT", default=None,
                    help="serial port of the real arm (e.g. COM9). Omit to preview in sim only.")
    ap.add_argument("--speed", type=int, default=600)
    args = ap.parse_args()

    if args.list or not args.name:
        from roarm_rl.library import catalog

        names = sorted(set(GESTURES) | set(catalog()))
        print("\n".join(names))
        print(f"{len(names)} gestures")
        return
    try:
        lookup(args.name)
    except KeyError as e:
        ap.error(e.args[0])

    if args.hardware:
        play_on_hardware(args.name, args.hardware, speed=args.speed)
    else:
        preview(args.name)


if __name__ == "__main__":
    main()
