"""Taking things from a hand and giving them back, guided by a camera.

The browser tracks a hand in its camera picture and estimates where the hand
is in the camera's own frame (metres; x right, y down, z away from the lens).
This module turns that into arm coordinates and runs the hand-over:

  take    wait for a steady hand, reach to the pinch point between thumb and
          index finger with the gripper open, close, carry the object back
  give    reach above an open palm, open the gripper, back away
  follow  hover near the hand as it moves

Camera-to-arm calibration: the arm visits a few poses and each time you pinch
its gripper tip. Each visit pairs a camera-frame point with a known arm-frame
point; a map (rotation, scale, shift and a depth stretch) is fitted through them, which also soaks up most of the
error of judging distance from one camera. Until that is done a rough guess is used
(camera 75 cm in front of the arm, looking at it), good enough to try things
in the simulator but not to hand anything to the real arm.

The arm never chases the live hand position, because a hand seen by one
camera jitters by centimetres and chasing that makes the arm shake. Instead
it looks, then moves: once the hand has been still for a moment its position
is averaged and locked, and the arm makes one planned move to that fixed
point. If the hand then wanders more than a few centimetres from the lock,
the arm waits for it to settle and locks again.

HandOver.update() is called on every simulator tick and returns a Command
(where to put the hand, what the gripper should do) or None when idle. The
target in a Command only changes when a new lock is taken.
"""

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from roarm_rl.brain import DATA_DIR

CAL_PATH = os.path.join(DATA_DIR, "camera.json")

# arm = A @ [X, Y, Z, 1]: camera 0.75 m in front of the arm at 0.30 m height, facing it
DEFAULT_MAP = [[0.0, 0.0, -1.0, 0.75], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.30]]

OPEN, CLOSED = 1.0, 0.0  # gripper radians
REST = [0.0, 0.0, 1.5708]
# [base, shoulder, elbow] for the calibration stops. The hand visits the corners
# of a box about 14 cm deep, 12 cm wide and 14 cm tall in front of the arm:
# small enough to stay in the camera's view, and deliberately varied in depth
# (toward the camera), which is the axis the fit most needs.
CAL_POSES = [
    [0.255, -0.305, 1.861], [-0.255, -0.188, 2.1], [0.161, 0.358, 1.55],
    [-0.161, 0.284, 1.215], [0.0, 0.089, 1.95], [0.0, -0.042, 1.566],
]

FRESH_S = 0.5  # a hand report older than this is not trusted
LOST_HOLD_S, LOST_ABORT_S = 1.5, 6.0
STEADY_S, STEADY_M = 0.6, 0.025  # the hand must stay within this for this long
ARRIVE_M = 0.02
PALM_OPEN_M = 0.06  # thumb-to-index gap that counts as an open hand
PINCH_M = 0.05
ABOVE_PALM_M = 0.06
FOLLOW_GAP_M = 0.09
RELOCK_M = 0.045  # the hand must wander this far from the locked point to count as moved
RELOCK_S = 0.35  # ...and stay away this long
FOLLOW_STEP_M = 0.03  # follow mode only re-aims for moves bigger than this
CAL_VERSION = 2  # bump when the browser's distance estimate changes; old files are ignored
MAX_REACH_ERROR_M = 0.035


@dataclass
class Command:
    xyz: Optional[list] = None  # where the hand should go (arm frame, metres)
    joints: Optional[list] = None  # or an explicit [base, shoulder, elbow]
    gripper: Optional[float] = None
    stop: bool = False  # coast to a halt where the arm is


def _similarity(cam, arm):
    """Best rotation, uniform scale and shift taking cam points onto arm points (Umeyama)."""
    mu_c, mu_a = cam.mean(axis=0), arm.mean(axis=0)
    c, a = cam - mu_c, arm - mu_a
    u, d, vt = np.linalg.svd(a.T @ c / len(cam))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ flip @ vt
    scale = np.trace(np.diag(d) @ flip) / (c ** 2).sum() * len(cam)
    return scale * rotation, mu_a - scale * rotation @ mu_c


def fit_camera(camera_points, arm_points):
    """3x4 map from camera points to arm points, and its RMS error in metres.

    The map is a rotation, one overall scale, a shift, and a separate stretch
    of the camera's depth axis, since depth judged from hand size is the least
    trustworthy number. That is 8 unknowns, few enough for six pinches to pin
    down even when each is a centimetre off.
    """
    cam, arm = np.asarray(camera_points, float), np.asarray(arm_points, float)
    best = None
    for depth in np.geomspace(0.25, 4.0, 121):
        stretch = np.diag([1.0, 1.0, depth])
        linear, shift = _similarity(cam @ stretch, arm)
        error = float(np.sqrt(np.mean(np.sum((cam @ stretch @ linear.T + shift - arm) ** 2, axis=1))))
        if best is None or error < best[0]:
            best = (error, np.hstack([linear @ stretch, shift[:, None]]))
    return best[1].tolist(), best[0]


class HandOver:
    def __init__(self, say=print):
        self._say = say
        self.mode = "idle"  # idle, follow, take, give, calibrate
        self.step = ""
        self.holding = False
        self.reach_error = 0.0  # set by the caller: how far the last target was from reachable
        self._hand = None  # {"t", "pinch", "palm", "open", "raw"} in the arm frame
        self._trail = deque()  # (t, pinch, palm) in the arm frame
        self._raw = deque()  # (t, pinch in the camera frame), for calibration
        self._since = 0.0
        self._lost_at = None
        self._told = None
        self._capture = False
        self._pairs = []
        self._lock = None  # {"pinch", "palm"}: the averaged hand position the arm is aiming at
        self._last_tcp, self._still_since = None, 0.0
        self._strayed_at = None
        self.map, self.cal_error = DEFAULT_MAP, None
        try:
            with open(CAL_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("version") == CAL_VERSION:
                self.map, self.cal_error = saved["map"], saved["error"]
        except (OSError, ValueError, KeyError):
            pass

    # --- camera reports ------------------------------------------------------------

    def _to_arm(self, point):
        return (np.asarray(self.map) @ np.array([*point, 1.0])).tolist()

    def see(self, report):
        """One report from the browser: {"seen", "pinch", "palm", "open"} in the camera frame."""
        now = time.monotonic()
        if not report.get("seen"):
            return
        try:
            raw = [float(v) for v in report["pinch"]]
            palm = [float(v) for v in report["palm"]]
            gap = float(report["open"])
        except (KeyError, TypeError, ValueError):
            return
        if not all(math.isfinite(v) for v in (*raw, *palm, gap)) or not 0.05 < raw[2] < 3.0:
            return
        pinch, palm = self._to_arm(raw), self._to_arm(palm)
        if self._hand is not None and now - self._hand["t"] < FRESH_S:  # smooth the jitter
            pinch = [0.5 * a + 0.5 * b for a, b in zip(pinch, self._hand["pinch"])]
            palm = [0.5 * a + 0.5 * b for a, b in zip(palm, self._hand["palm"])]
        self._hand = {"t": now, "pinch": pinch, "palm": palm, "open": gap, "raw": raw}
        self._raw.append((now, raw))
        while self._raw and now - self._raw[0][0] > 1.2:
            self._raw.popleft()
        self._trail.append((now, pinch, palm))
        while self._trail and now - self._trail[0][0] > 1.2:
            self._trail.popleft()

    def camera_live(self, now=None):
        return self._hand is not None and (now or time.monotonic()) - self._hand["t"] < 2.5

    def _fresh(self, now):
        return self._hand is not None and now - self._hand["t"] < FRESH_S

    def _steady(self, now, seconds=STEADY_S):
        recent = [p for t, p, _ in self._trail if now - t <= seconds]
        if len(recent) < 4 or now - self._trail[0][0] < seconds * 0.9:
            return False
        centre = np.mean(recent, axis=0)
        return float(np.max(np.linalg.norm(np.asarray(recent) - centre, axis=1))) < STEADY_M

    # --- control -------------------------------------------------------------------

    def _tell(self, text):
        if text != self._told:
            self._told = text
            self._say(text)

    def _go(self, step, now):
        self.step, self._since = step, now
        self._strayed_at = None

    def _take_lock(self, now):
        """Freeze the hand position: the average over the last moment of stillness."""
        recent = [(p, palm) for t, p, palm in self._trail if now - t <= STEADY_S]
        self._lock = {"pinch": np.mean([p for p, _ in recent], axis=0).tolist(),
                      "palm": np.mean([palm for _, palm in recent], axis=0).tolist()}
        self._strayed_at = None

    def _strayed(self, now, limit=RELOCK_M):
        """True once the hand has stayed away from the locked point for a while."""
        if math.dist(self._hand["pinch"], self._lock["pinch"]) < limit:
            self._strayed_at = None
            return False
        self._strayed_at = self._strayed_at or now
        return now - self._strayed_at > RELOCK_S

    def jitter(self, now):
        """How much the hand position wobbled over the last second, in metres (None if unseen)."""
        recent = [p for t, p, _ in self._trail if now - t <= 1.0]
        if len(recent) < 5:
            return None
        return float(np.sqrt(np.mean(np.sum((np.asarray(recent) - np.mean(recent, axis=0)) ** 2, axis=1))))

    def start(self, mode):
        now = time.monotonic()
        if mode != "calibrate" and not self.camera_live(now):
            return "Turn the camera on first (Camera tab) and show me your hand."
        self.mode, self._told, self._lost_at, self._lock = mode, None, None, None
        self._go("wait", now)
        if mode == "calibrate":
            self._pairs, self._capture = [], False
            self._go("move0", now)
            return ("Calibrating. I'll go to six positions. At each one, pinch my gripper tip "
                    "between thumb and index finger and hold still.")
        return {"take": "Hold it out and keep your hand still.",
                "give": "Hold out an open hand and keep it still.",
                "follow": "Following your hand."}[mode]

    def cancel(self):
        if self.mode != "idle":
            self.mode, self.step = "idle", ""

    def capture_now(self):
        self._capture = True

    def reset_calibration(self):
        self.map, self.cal_error = DEFAULT_MAP, None
        try:
            os.remove(CAL_PATH)
        except OSError:
            pass

    def status(self):
        fresh = self._fresh(time.monotonic())
        return {
            "mode": self.mode, "step": self.step, "holding": self.holding,
            "camera": self.camera_live(), "calibrated": self.cal_error is not None,
            "error": self.cal_error, "hand": [round(v, 3) for v in self._hand["pinch"]] if fresh else None,
            "progress": len(self._pairs) if self.mode == "calibrate" else None,
            "jitter": self.jitter(time.monotonic()) if fresh else None,
        }

    # --- the hand-over itself ------------------------------------------------------

    def update(self, now, tcp, q):
        """Called every tick with the hand position and joints. Returns a Command or None."""
        if self._last_tcp is None or math.dist(tcp, self._last_tcp) > 0.0004:
            self._still_since = now  # the arm is (still) moving
        self._last_tcp = list(tcp)
        if self.mode == "idle":
            return None
        if self.mode == "calibrate":
            return self._calibrate(now, tcp, q)

        fresh = self._fresh(now)
        if fresh:
            self._lost_at = None
        elif self.step in ("wait", "approach"):
            self._lost_at = self._lost_at or now
            if now - self._lost_at > LOST_ABORT_S:
                self._tell("I lost sight of your hand, so I stopped.")
                self.mode, self.step = "idle", ""
                return Command(joints=REST)
            if now - self._lost_at > LOST_HOLD_S:
                self._tell("I can't see your hand. Waiting.")
                return Command(stop=True)
            return Command()  # a brief dropout: carry on with the planned move

        if self.mode == "follow":
            if self._lock is None or (self._strayed(now, FOLLOW_STEP_M) and self._steady(now, 0.3)):
                self._take_lock(now)
            pinch = self._lock["pinch"]
            reach = math.hypot(pinch[0], pinch[1]) or 1.0
            back = max(0.0, reach - FOLLOW_GAP_M) / reach  # stop short, on the arm's side of the hand
            return Command(xyz=[pinch[0] * back, pinch[1] * back, pinch[2]])

        if self.mode == "take":
            return self._take(now, tcp, q)
        return self._give(now, tcp)

    def _arrived(self, now, tcp, target):
        """At the target and at rest: the gripper only acts once the arm has stopped."""
        return math.dist(tcp, target) < ARRIVE_M and now - self._still_since > 0.15

    def _take(self, now, tcp, q):
        if self.step == "wait":
            if self._steady(now) and q[3] > 0.98 * OPEN:  # set off only once the gripper is fully open
                self._take_lock(now)
                self._go("approach", now)
                self._tell("Coming to take it. Keep still.")
            return Command(gripper=OPEN)
        if self.step == "approach":
            target = self._lock["pinch"]
            if self._strayed(now):
                self._go("wait", now)
                self._tell("You moved. Hold still and I'll come again.")
                return Command(gripper=OPEN, stop=True)
            if self.reach_error > MAX_REACH_ERROR_M and now - self._since > 1.0:
                self._tell("That's out of my reach. Bring it a bit closer.")
            elif self._arrived(now, tcp, target):
                self._go("grip", now)
            return Command(xyz=target, gripper=OPEN)
        if self.step == "grip":
            if now - self._since > 0.9:
                self.holding = True
                self._go("retract", now)
            return Command(xyz=self._lock["pinch"], gripper=CLOSED)
        if now - self._since > 2.2:  # retract
            self._tell("Got it.")
            self.mode, self.step = "idle", ""
        return Command(joints=REST, gripper=CLOSED)

    def _give(self, now, tcp):
        if self.step == "wait":
            if self._steady(now) and self._hand["open"] > PALM_OPEN_M:
                self._take_lock(now)
                self._go("approach", now)
                self._tell("Here it comes. Keep your hand still.")
            return Command(gripper=CLOSED)
        palm = self._lock["palm"]
        target = [palm[0], palm[1], palm[2] + ABOVE_PALM_M]
        if self.step == "approach":
            if self._strayed(now):
                self._go("wait", now)
                self._tell("You moved. Hold still and I'll come again.")
                return Command(gripper=CLOSED, stop=True)
            if self.reach_error > MAX_REACH_ERROR_M and now - self._since > 1.0:
                self._tell("Your hand is out of my reach. Bring it a bit closer.")
            elif self._arrived(now, tcp, target):
                self._go("release", now)
            return Command(xyz=target, gripper=CLOSED)
        if self.step == "release":
            if now - self._since > 0.9:
                self.holding = False
                self._go("retract", now)
            return Command(xyz=target, gripper=OPEN)
        if now - self._since > 2.2:  # retract
            self._tell("There you go.")
            self.mode, self.step = "idle", ""
            return Command(joints=REST, gripper=CLOSED)
        return Command(joints=REST, gripper=OPEN)

    def _calibrate(self, now, tcp, q):
        index = int(self.step[4:]) if self.step.startswith("move") else int(self.step[7:])
        pose = CAL_POSES[index]
        if self.step.startswith("move"):
            if max(abs(a - b) for a, b in zip(q[:3], pose)) < 0.02 and now - self._since > 0.8:
                self._go(f"capture{index}", now)
                self._capture = False
                self._tell(f"Position {index + 1} of {len(CAL_POSES)}: pinch my gripper tip and hold still.")
            return Command(joints=pose, gripper=CLOSED)

        pinching = self._fresh(now) and self._hand["open"] < PINCH_M and self._steady(now, 0.8)
        if self._capture and not self._fresh(now):
            self._capture = False
            self._tell("I can't see your hand, so I couldn't capture that one.")
        if (pinching and now - self._since > 1.0) or (self._capture and self._fresh(now)):
            recent = [raw for t, raw in self._raw if now - t <= 0.8] or [self._hand["raw"]]
            self._pairs.append((np.mean(recent, axis=0).tolist(), list(tcp)))
            self._capture = False
            if index + 1 < len(CAL_POSES):
                self._go(f"move{index + 1}", now)
                self._tell(f"Got position {index + 1}. Let go, I'm moving on.")
            else:
                self.map, self.cal_error = fit_camera(*zip(*self._pairs))
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(CAL_PATH, "w", encoding="utf-8") as f:
                    json.dump({"version": CAL_VERSION, "map": self.map, "error": self.cal_error}, f)
                note = "" if self.cal_error < 0.03 else " That is loose; try again with a steadier pinch."
                self._tell(f"Calibrated. Typical error {self.cal_error * 100:.1f} cm.{note}")
                self.mode, self.step = "idle", ""
                return Command(joints=REST, gripper=CLOSED)
        return Command(joints=pose, gripper=CLOSED)
