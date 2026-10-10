"""Measure how closely a real RoArm-M2 follows what the web app tells it to do.

    python -m roarm_rl.benchmark static      # each joint to fixed targets, from both directions
    python -m roarm_rl.benchmark poses       # whole-arm poses, visited twice
    python -m roarm_rl.benchmark step        # step responses at the web app's three servo profiles
    python -m roarm_rl.benchmark sine        # sine tracking per joint, 0.2 to 3 Hz
    python -m roarm_rl.benchmark stream      # one reference sent at different rates and accelerations
    python -m roarm_rl.benchmark limits      # a quick sine at different accelerations and speed caps
    python -m roarm_rl.benchmark hold        # stand still at many poses and watch for hunting
    python -m roarm_rl.benchmark ik          # reach Cartesian targets through the simulator's IK
    python -m roarm_rl.benchmark animations  # gestures and traced shapes through the web app's own loop
    python -m roarm_rl.benchmark iksim       # IK residuals in the simulator only (no arm needed)
    python -m roarm_rl.benchmark park        # go back to a stored pose and release the motors

Every run writes one JSON log to docs/paper/data/. roarm_rl.benchmark_report turns
the logs into the figures and tables used by docs/paper.

"Measured" always means the servos' own encoders, read back over USB. That
captures what the servo loop did (lag, speed limit, sag under load, dead band)
but not flex or play between the encoder and the hand.

All commands stay inside roarm_rl.gesture.SAFE_BOUNDS. The arm is brought to
the home pose slowly before each run and left there, holding.
"""

import argparse
import json
import math
import os
import random
import tempfile
import time

# The web app's Robot keeps a chat log and preferences; a benchmark must not write into the real ones.
_SCRATCH = tempfile.mkdtemp(prefix="roarm_bench_")
os.environ.setdefault("ROARM_DATA_DIR", _SCRATCH)
os.environ.setdefault("ROARM_LEARNED_PATH", os.path.join(_SCRATCH, "learned.json"))

from roarm_rl import library  # noqa: E402
from roarm_rl.gesture import SAFE_BOUNDS, fit_to_speed  # noqa: E402
from roarm_rl.hardware import RoArmHardware  # noqa: E402
from roarm_rl.server import (ARM_FAST, ARM_GENTLE, ARM_MAX_SPEED, ARM_PLANNED, MANUAL_SPEED,  # noqa: E402
                             MIRROR_EPSILON, MIRROR_MIN_INTERVAL, REACH_SPEED, Robot, find_arm_port)
from roarm_rl.sim import RoArmSim  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "docs", "paper", "data")
HOME = [0.0, 0.0, 1.5708, 0.0]
JOINTS = ["base", "shoulder", "elbow", "gripper"]
COUNT = 2 * math.pi / 4096  # one encoder count of the ST3215 servo, in radians


def clamp(q):
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(q, SAFE_BOUNDS)]


class Link(RoArmHardware):
    """The web app's hardware wrapper, plus a time-stamped read of everything the arm reports."""

    def __init__(self):
        super().__init__()
        self.sent = []  # (time, [4 radians], speed, acc) of every command, for the logs

    def set_joint_targets(self, radians, speed=None, acc=None):
        for value, (lo, hi), name in zip(radians, SAFE_BOUNDS, JOINTS):
            if not lo - 1e-6 <= value <= hi + 1e-6:
                raise ValueError(f"{name}={value} outside safe range {lo}..{hi}")
        super().set_joint_targets(radians, speed, acc)
        self.sent.append((time.monotonic(), list(radians), speed, acc))

    def read(self):
        """(time, joints rad, load, firmware xyz mm) or None. Takes about 25 ms."""
        arm = self._arm
        with arm.lock:
            port = arm._serial_port
            port.reset_input_buffer()
            asked = time.monotonic()
            port.write(b'{"T":105}\n')
            port.flush()
            while time.monotonic() - asked < 0.25:
                line = port.readline()
                if line.startswith(b'{"T":1051'):
                    got = time.monotonic()
                    try:
                        d = json.loads(line)
                    except ValueError:
                        return None
                    # the servos are read somewhere between the request and the reply
                    return ((asked + got) / 2,
                            [d["b"], d["s"], d["e"], math.pi - d["t"]],
                            [d["torB"], d["torS"], d["torE"], d["torH"]],
                            [d["x"], d["y"], d["z"]])
        return None


class Kin:
    """Forward and inverse kinematics from the same simulator model the web app uses."""

    def __init__(self):
        self.sim = RoArmSim(gui=False)

    def fk(self, q):
        self.sim.set_joint_targets(list(q), instant=True)
        return list(self.sim.get_ee_pose()[0])

    def ik(self, xyz, seed=HOME, passes=4):
        """Joint angles for a hand position, solved the way the hand-over code does."""
        self.sim.set_joint_targets(list(seed), instant=True)
        solved = list(seed[:3])
        for _ in range(passes):
            solved = clamp(self.sim.solve_ik(xyz))[:3]
            self.sim.set_joint_targets(solved + [seed[3]], instant=True)
        return solved + [seed[3]]


class Log:
    """One recording: what was wanted at each sample, and what the arm reported."""

    def __init__(self, **meta):
        self.meta = meta
        self.t, self.q, self.load, self.xyz, self.ref, self.sent = [], [], [], [], [], []

    def add(self, t, sample, ref):
        self.t.append(round(t, 4))
        self.q.append([round(v, 5) for v in sample[1]])
        self.load.append(sample[2])
        self.xyz.append([round(v, 2) for v in sample[3]])
        self.ref.append([round(v, 5) for v in ref])

    def as_dict(self):
        return {**self.meta, "t": self.t, "q": self.q, "load": self.load, "xyz": self.xyz,
                "ref": self.ref, "sent": self.sent}


def stream(link, ref, seconds, profile=ARM_FAST, interval=MIRROR_MIN_INTERVAL, tail=0.8, **meta):
    """Send ref(t) to the arm the way the web app's mirror does, reading the arm back in between."""
    log = Log(profile=list(profile), interval=interval, seconds=seconds, **meta)
    last, last_at = None, -1.0
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        if t > seconds + tail:
            return log
        target = clamp(ref(min(t, seconds)))
        if (last is None or max(abs(a - b) for a, b in zip(target, last)) > MIRROR_EPSILON) \
                and t - last_at > interval:
            link.set_joint_targets(target, *profile)
            log.sent.append([round(t, 4)] + [round(v, 5) for v in target])
            last, last_at = target, t
        sample = link.read()
        if sample is not None:
            at = sample[0] - t0
            log.add(at, sample, clamp(ref(min(max(at, 0.0), seconds))))


def measure(link, n=5):
    """Mean of n encoder reads."""
    reads = [s[1] for s in (link.read() for _ in range(n)) if s is not None]
    return [sum(v) / len(reads) for v in zip(*reads)]


def go_gently(link, goal, speed=0.5):
    """Move to a pose slowly along a straight line in joint space and wait until it is there."""
    goal = clamp(goal)
    start = clamp(measure(link, 2))
    seconds = max(0.3, max(abs(a - b) for a, b in zip(start, goal)) / speed)
    stream(link, lambda t: [a + (b - a) * t / seconds for a, b in zip(start, goal)], seconds,
           profile=ARM_GENTLE if speed <= 0.4 else ARM_PLANNED, tail=0.6)


def ramp(start, goal, speed):
    """The web app's slider motion: every joint heads for its goal at a fixed speed."""
    def ref(t):
        step = speed * t
        return [a + max(-step, min(step, b - a)) for a, b in zip(start, goal)]
    return ref, max(abs(a - b) for a, b in zip(start, goal)) / speed


def save(name, payload):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) // 1024} kB)")


# --- experiments ------------------------------------------------------------------

def static(link, cycles=3, points=7, dwell=1.0):
    """Each joint alone to evenly spaced targets, going up then down, as a slider drag would."""
    runs = []
    for j, name in enumerate(JOINTS):
        lo, hi = SAFE_BOUNDS[j]
        margin = 0.05 * (hi - lo)
        targets = [lo + margin + k * (hi - lo - 2 * margin) / (points - 1) for k in range(points)]
        at = list(HOME)
        at[j] = targets[0]
        go_gently(link, at)
        for cycle in range(cycles):
            for direction, sweep in (("up", targets[1:]), ("down", targets[-2::-1])):
                for target in sweep:
                    goal = list(at)
                    goal[j] = target
                    ref, seconds = ramp(at, goal, MANUAL_SPEED)
                    log = stream(link, ref, seconds, tail=dwell, joint=name, target=target,
                                 direction=direction, cycle=cycle)
                    runs.append(log.as_dict())
                    at = goal
            print(f"static {name} cycle {cycle + 1}/{cycles}")
        go_gently(link, HOME)
    save("static", {"runs": runs})


def poses(link, count=24, visits=2, dwell=1.0, seed=7):
    """Random whole-arm poses, each visited more than once from a different previous pose."""
    rng = random.Random(seed)
    targets = [[round(rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)), 4)
                for lo, hi in SAFE_BOUNDS] for _ in range(count)]
    order = [i for _ in range(visits) for i in rng.sample(range(count), count)]
    at, runs = list(HOME), []
    for n, i in enumerate(order):
        ref, seconds = ramp(at, targets[i], MANUAL_SPEED)
        runs.append(stream(link, ref, seconds, tail=dwell, pose=i, target=targets[i]).as_dict())
        at = targets[i]
        if n % 8 == 7:
            print(f"poses {n + 1}/{len(order)}")
    go_gently(link, HOME)
    save("poses", {"runs": runs})


STEP_CENTRE = [0.0, 0.0, 1.45, 0.6]


def step(link):
    """One command, then watch: dead time, speed, overshoot and settling for each servo profile."""
    runs = []
    plans = [(size, ARM_FAST) for size in (0.05, 0.2, 0.5)] + [(0.5, ARM_PLANNED), (0.5, ARM_GENTLE)]
    for j, name in enumerate(JOINTS):
        if name == "base":
            joint_plans = plans + [(1.0, ARM_FAST)]
        else:
            joint_plans = plans
        for size, profile in joint_plans:
            low, high = list(STEP_CENTRE), list(STEP_CENTRE)
            low[j] -= size / 2
            high[j] += size / 2
            go_gently(link, low)
            time.sleep(0.4)
            for direction, a, b in (("up", low, high), ("down", high, low)):
                seconds = 0.6 + size / (0.35 if profile == ARM_GENTLE else 1.2)
                log = stream(link, lambda t, b=b: b, seconds, profile=profile, tail=0.6,
                             joint=name, size=size, direction=direction, start=a, target=b)
                runs.append(log.as_dict())
        print(f"step {name}")
    go_gently(link, HOME)
    save("step", {"runs": runs})


def _sine(j, centre, amplitude, hz):
    def ref(t):
        q = list(centre)
        q[j] += amplitude * math.sin(2 * math.pi * hz * t)
        return q
    return ref


def sine(link):
    """A sine on one joint at a time, streamed like a gesture. Shows gain, lag and where it gives up."""
    runs = []
    plans = [(0.15, hz) for hz in (0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0)] + \
            [(0.4, hz) for hz in (0.2, 0.4, 0.7, 1.0)]
    go_gently(link, STEP_CENTRE)
    for j, name in enumerate(JOINTS):
        for amplitude, hz in plans:
            seconds = max(4.0, 5.0 / hz)
            seconds = round(seconds * hz) / hz  # whole cycles, so it ends where it began
            log = stream(link, _sine(j, STEP_CENTRE, amplitude, hz), seconds, tail=0.5,
                         joint=name, amplitude=amplitude, hz=hz, centre=STEP_CENTRE)
            runs.append(log.as_dict())
        print(f"sine {name}")
    go_gently(link, HOME)
    save("sine", {"runs": runs})


def stream_shapes(link):
    """The same two-joint sine sent at different command rates, accelerations and speed caps."""
    def ref(t):
        q = list(STEP_CENTRE)
        q[0] += 0.3 * math.sin(2 * math.pi * 0.5 * t)
        q[2] += 0.25 * math.sin(2 * math.pi * 0.5 * t + math.pi / 2) - 0.25
        return q
    runs = []
    go_gently(link, ref(0))
    variants = [("web app (25 Hz, 1500/60)", MIRROR_MIN_INTERVAL, ARM_FAST),
                ("5 Hz", 0.2, ARM_FAST), ("10 Hz", 0.1, ARM_FAST), ("as fast as possible", 0.0, ARM_FAST),
                ("acc 0", MIRROR_MIN_INTERVAL, (1500, 0)), ("acc 10", MIRROR_MIN_INTERVAL, (1500, 10)),
                ("acc 254", MIRROR_MIN_INTERVAL, (1500, 254)),
                ("speed 880", MIRROR_MIN_INTERVAL, ARM_PLANNED), ("speed 300", MIRROR_MIN_INTERVAL, ARM_GENTLE),
                ("speed 4000", MIRROR_MIN_INTERVAL, (4000, 60))]
    for label, interval, profile in variants:
        runs.append(stream(link, ref, 8.0, profile=profile, interval=interval, tail=0.6,
                           label=label).as_dict())
        print(f"stream {label}")
    go_gently(link, HOME)
    save("stream", {"runs": runs})


def limits(link):
    """A sine too quick for the default acceleration, repeated with other accelerations and speeds."""
    runs = []
    go_gently(link, STEP_CENTRE)
    for hz in (1.5, 2.0):
        for profile in ((1500, 60), (1500, 150), (1500, 254), (1500, 0), (4000, 254), (4000, 0)):
            seconds = round(4.0 * hz) / hz
            log = stream(link, _sine(0, STEP_CENTRE, 0.15, hz), seconds, profile=profile, tail=0.5,
                         joint="base", amplitude=0.15, hz=hz, centre=STEP_CENTRE)
            runs.append(log.as_dict())
            print(f"limits {hz} Hz {profile}")
    go_gently(link, HOME)
    save("limits", {"runs": runs})


def hold(link, seconds=5.0):
    """Stand still at a spread of shoulder and elbow angles and watch whether the joints keep quiet."""
    runs = []
    at = list(HOME)
    for elbow in (0.9, 1.5708, 2.05):
        for k in range(8):
            goal = [0.0, -0.55 + k * 1.05 / 7, elbow, 0.0]
            ref, move = ramp(at, goal, 0.6)
            stream(link, ref, move, tail=1.0)
            runs.append(stream(link, lambda t, goal=goal: goal, seconds, tail=0.0, target=goal).as_dict())
            at = goal
        print(f"hold elbow {elbow}")
    go_gently(link, HOME)
    save("hold", {"runs": runs})


def ik_targets(kin, count, seed=11):
    """Hand positions that are reachable inside SAFE_BOUNDS, made by forward kinematics."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        q = [rng.uniform(lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo)) for lo, hi in SAFE_BOUNDS[:3]] + [0.0]
        out.append((kin.fk(q), q))
    return out


def iksim(count=2000):
    """How well the simulator's IK lands on a target, with the web app's two ways of calling it."""
    kin = Kin()
    rng = random.Random(3)
    rows = []
    at = list(HOME)
    for xyz, truth in ik_targets(kin, count):
        # dragging the hand: one solve from wherever the arm is now
        kin.sim.set_joint_targets(at, instant=True)
        t0 = time.perf_counter()
        once = clamp(kin.sim.solve_ik(xyz)[:3] + [0.0])
        took = time.perf_counter() - t0
        # hand-over: four solves starting from home
        four = kin.ik(xyz)
        rows.append({"xyz": [round(v, 5) for v in xyz], "truth": [round(v, 5) for v in truth],
                     "once": [round(v, 5) for v in once], "four": [round(v, 5) for v in four],
                     "err_once": math.dist(kin.fk(once), xyz), "err_four": math.dist(kin.fk(four), xyz),
                     "ms": 1000 * took})
        at = once
    # targets nobody checked for reachability: a box around the arm, as a drag could ask for
    box = []
    for _ in range(count // 2):
        xyz = [rng.uniform(-0.1, 0.55), rng.uniform(-0.45, 0.45), rng.uniform(0.04, 0.65)]
        four = kin.ik(xyz)
        box.append({"xyz": [round(v, 4) for v in xyz], "err_four": math.dist(kin.fk(four), xyz)})
    save("iksim", {"reachable": rows, "box": box})


def ik(link, count=24, dwell=1.0):
    """Drag-the-hand moves on the real arm: IK in the simulator, a ramp, then read where it ended."""
    kin = Kin()
    runs, at = [], list(HOME)
    for n, (xyz, _) in enumerate(ik_targets(kin, count, seed=23)):
        kin.sim.set_joint_targets(at, instant=True)
        goal = clamp(kin.sim.solve_ik(xyz)[:3] + [at[3]])
        ref, seconds = ramp(at, goal, min(REACH_SPEED, ARM_MAX_SPEED))
        log = stream(link, ref, seconds, tail=dwell, hand=[round(v, 5) for v in xyz], goal=goal,
                     ik_error=math.dist(kin.fk(goal), xyz))
        runs.append(log.as_dict())
        at = goal
        if n % 8 == 7:
            print(f"ik {n + 1}/{count}")
    go_gently(link, HOME)
    save("ik", {"runs": runs})


# --- animations -------------------------------------------------------------------

SHAPE_CENTRE = (0.30, 0.0, 0.36)  # metres; close to where the hand is at home


def _shape_points(name, n):
    """n points of a closed path around the origin, in metres: (forward, left, up)."""
    pts = []
    for k in range(n + 1):
        a = 2 * math.pi * k / n
        u = k / n
        if name == "circle_front":  # a circle facing the arm, like drawing on a wall
            pts.append((0.0, 0.06 * math.sin(a), 0.06 * math.cos(a)))
        elif name == "circle_flat":  # a circle parallel to the table
            pts.append((0.05 * math.cos(a), 0.06 * math.sin(a), 0.0))
        elif name == "circle_side":  # a circle in the arm's own plane: shoulder and elbow only
            pts.append((0.05 * math.sin(a), 0.0, 0.05 * math.cos(a)))
        elif name == "figure_eight_front":
            pts.append((0.0, 0.08 * math.sin(a), 0.04 * math.sin(2 * a)))
        elif name == "spiral_front":  # two turns, growing from 1.5 cm to 7 cm
            r = 0.015 + 0.055 * u
            pts.append((0.0, r * math.sin(2 * a), r * math.cos(2 * a)))
        elif name in ("square_front", "triangle_front"):
            if name == "square_front":
                corners = [(-.05, .05), (.05, .05), (.05, -.05), (-.05, -.05)]
            else:
                corners = [(0.0, .06), (.06, -.04), (-.06, -.04)]
            side, f = divmod(u * len(corners), 1.0)
            (y0, z0), (y1, z1) = corners[int(side) % len(corners)], corners[(int(side) + 1) % len(corners)]
            pts.append((0.0, y0 + (y1 - y0) * f, z0 + (z1 - z0) * f))
        elif name == "line_forward":  # out and back twice
            pts.append((0.06 * math.sin(2 * a), 0.0, 0.0))
        elif name == "line_up":
            pts.append((0.0, 0.0, 0.07 * math.sin(2 * a)))
        else:
            raise KeyError(name)
    return pts


SHAPES = {"circle_front": (16, 5.0), "circle_flat": (16, 5.0), "circle_side": (16, 5.0),
          "figure_eight_front": (24, 6.5), "spiral_front": (32, 7.0), "square_front": (24, 6.5),
          "triangle_front": (24, 6.0), "line_forward": (16, 5.0), "line_up": (16, 5.0)}

GESTURES = [("nod", "normal", "normal"), ("nod", "fast", "big"), ("wave", "normal", "normal"),
            ("bow", "normal", "normal"), ("shake", "normal", "normal"), ("circle", "normal", "normal"),
            ("figure_eight", "normal", "normal"), ("dance", "normal", "normal"),
            ("celebrate", "normal", "big"), ("stir", "slow", "normal")]


def shape_keyframes(kin, name):
    """(keyframes, ideal path) for a traced shape: Cartesian points turned into poses by IK."""
    points, seconds = SHAPES[name]
    path = [[c + d for c, d in zip(SHAPE_CENTRE, p)] for p in _shape_points(name, points)]
    seed, frames = list(HOME), [(0.5, list(HOME))]
    for k, xyz in enumerate(path):
        seed = kin.ik(xyz, seed)
        if math.dist(kin.fk(seed), xyz) > 0.002:
            raise ValueError(f"{name}: point {k} is out of reach inside SAFE_BOUNDS")
        frames.append((0.8 if k == 0 else seconds / points, list(seed)))
    frames.append((0.8, list(HOME)))
    return frames, path


def animation_set(kin):
    """{name: {keyframes, path}} for every animation in the benchmark."""
    out = {}
    for name, speed, size in GESTURES:
        label = library.variant_name(name, speed, size)
        out[label] = {"keyframes": library.build(name, speed, size), "kind": "gesture"}
    for name in SHAPES:
        frames, path = shape_keyframes(kin, name)
        out[name] = {"keyframes": frames, "path": path, "kind": "shape"}
    return out


def animations(link, repeats=2):
    """Play every animation through the web app's Robot with mirroring on, and log picture and arm."""
    kin = Kin()
    todo = animation_set(kin)
    robot = Robot(link)
    robot.composer = None
    robot.start()
    runs = []
    try:
        time.sleep(0.5)
        robot.set_mirror(True)
        time.sleep(4.0)  # the web app closes the gap to the picture slowly first
        for name, item in todo.items():
            for repeat in range(repeats):
                log = Log(name=name, kind=item["kind"], repeat=repeat,
                          designed=[[d, list(p)] for d, p in item["keyframes"]],
                          played=[[d, list(p)] for d, p in fit_to_speed(item["keyframes"], ARM_MAX_SPEED)],
                          path=item.get("path"))
                first = len(link.sent)
                with robot._lock:
                    robot.player.enqueue(name, item["keyframes"])
                t0 = time.monotonic()
                picture, ended = [], None
                while True:
                    now = time.monotonic()
                    state = robot.state
                    picture.append((state["t"] - t0, state["q"]))
                    if ended is None and now - t0 > 0.3 and not robot.player.busy:
                        ended = now
                    if ended is not None and now - ended > 1.0:
                        break
                    sample = link.read()
                    if sample is not None:
                        log.add(sample[0] - t0, sample, state["q"])
                    # A read holds the serial port for about 25 ms. Reading back to back would
                    # starve the web app's own commands, so leave it the port most of the time.
                    time.sleep(0.03)
                log.meta["picture"] = [[round(t, 4)] + q for t, q in picture]
                log.sent = [[round(t - t0, 4)] + [round(v, 5) for v in q] + [s, a]
                            for t, q, s, a in link.sent[first:]]
                runs.append(log.as_dict())
                print(f"animation {name} run {repeat + 1}: {log.t[-1]:.1f} s, {len(log.t)} samples")
    finally:
        robot.set_mirror(False)
        robot._stop.set()
        robot._thread.join(timeout=2)
    go_gently(link, HOME)
    save("animations", {"runs": runs})


def park(link, pose, release):
    """Go to a stored pose slowly (it may lie outside SAFE_BOUNDS) and optionally let go."""
    start = measure(link, 2)
    seconds = max(1.0, max(abs(a - b) for a, b in zip(start, pose)) / 0.3)
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds + 1.0:
        u = min(1.0, (time.monotonic() - t0) / seconds)
        RoArmHardware.set_joint_targets(link, [a + (b - a) * u for a, b in zip(start, pose)], *ARM_GENTLE)
        time.sleep(0.05)
    print("parked at", [round(v, 3) for v in measure(link)])
    if release:
        link.set_torque(False)
        print("motors released")


def main():
    ap = argparse.ArgumentParser(description="RoArm-M2 tracking benchmark")
    ap.add_argument("what", choices=["static", "poses", "step", "sine", "stream", "limits", "hold", "ik",
                                     "animations", "iksim", "park", "read"])
    ap.add_argument("--port", default=None, help="serial port (default: find the arm's USB adapter)")
    ap.add_argument("--pose", type=float, nargs=4, default=None, help="pose for 'park', in radians")
    ap.add_argument("--release", action="store_true", help="'park': switch the motors off afterwards")
    args = ap.parse_args()

    if args.what == "iksim":
        iksim()
        return
    link = Link()
    link.connect_serial(args.port or find_arm_port())
    try:
        time.sleep(0.3)
        if args.what == "read":
            print(link.read())
            return
        link.set_torque(True)
        time.sleep(0.3)
        if args.what == "park":
            park(link, args.pose or HOME, args.release)
            return
        start = measure(link, 2)
        if any(not lo <= v <= hi for v, (lo, hi) in zip(start[:3], SAFE_BOUNDS)):
            park(link, HOME, False)  # outside the gesture range (folded away, say): come out slowly
        go_gently(link, HOME, speed=0.4)
        {"static": static, "poses": poses, "step": step, "sine": sine, "stream": stream_shapes,
         "limits": limits, "hold": hold, "ik": ik, "animations": animations}[args.what](link)
    finally:
        link.disconnect()


if __name__ == "__main__":
    main()
