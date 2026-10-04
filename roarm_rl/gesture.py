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
"""

import argparse
import time

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

# Firmware limits intersected with URDF limits (see roarm_rl.sim._FIRMWARE_LIMITS_M2).
SAFE_BOUNDS = [(-0.6, 0.6), (-0.3, 0.4), (1.0, 2.2), (0.0, 0.8)]
COMMAND_HZ = 25


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


def sample_times(keyframes, hz=COMMAND_HZ):
    """Yield (t, pose) at hz along a Catmull-Rom spline through the keyframes."""
    poses = [kf[1] for kf in keyframes]
    seg_times = []
    t_acc = 0.0
    for dur, _ in keyframes[1:]:
        seg_times.append((t_acc, t_acc + dur))
        t_acc += dur
    total = t_acc

    dt = 1.0 / hz
    t = 0.0
    while t <= total + 1e-9:
        seg = len(seg_times) - 1
        for i, (start, end) in enumerate(seg_times):
            if t <= end:
                seg = i
                break
        start, end = seg_times[seg]
        s = min(max((t - start) / (end - start), 0.0), 1.0)
        i = seg + 1  # poses index of segment end point
        p0 = poses[max(i - 2, 0)]
        p1 = poses[i - 1]
        p2 = poses[i]
        p3 = poses[min(i + 1, len(poses) - 1)]
        pose = _catmull_rom(p0, p1, p2, p3, s)
        yield t, [min(max(v, lo), hi) for v, (lo, hi) in zip(pose, SAFE_BOUNDS)]
        t += dt


def preview(name):
    kf = GESTURES[name]
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
    kf = GESTURES[name]
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
    ap.add_argument("name", choices=sorted(GESTURES))
    ap.add_argument("--hardware", metavar="PORT", default=None,
                    help="serial port of the real arm (e.g. COM9). Omit to preview in sim only.")
    ap.add_argument("--speed", type=int, default=600)
    args = ap.parse_args()

    if args.hardware:
        play_on_hardware(args.name, args.hardware, speed=args.speed)
    else:
        preview(args.name)


if __name__ == "__main__":
    main()
