"""Gesture library: hand-designed motions, each with speed/size/side variants.

Every motion is written once in normalized units and turned into radian
keyframes by build(), so the same definition yields a slow small nod and a
fast big one. Normalized keyframe fields:

    b   base      -1 .. 1   + is the arm's left
    s   shoulder  -1 .. 1   + leans forward, - leans back
    u   forearm   -1 .. 1   + raises the hand, - lowers it
    g   gripper    0 .. 1   1 is fully open

1.0 maps to the edge of roarm_rl.gesture.SAFE_BOUNDS, so no variant can leave
the range that is safe to send to the real arm.

Motions added at run time (roarm_rl.composer) use the same units and live in
gestures/learned.json; they are loaded on import.

catalog() names every variant (nod, nod_fast, nod_slow_big, peek_right, ...)
for roarm_rl.gesture's CLI; roarm_rl.intent picks variants from chat text.
"""

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from roarm_rl.gesture import SAFE_BOUNDS

REST_ELBOW = 1.57

SPEEDS = {"slow": 1.6, "normal": 1.0, "fast": 0.65}  # duration multiplier
SIZES = {"small": 0.5, "normal": 0.8, "big": 1.0}  # amplitude multiplier
MAX_REPS = 8


def K(dur, b=0.0, s=0.0, u=0.0, g=0.0):
    return (dur, (b, s, u, g))


def R(dur=0.5):
    """Rest pose keyframe."""
    return K(dur)


def _loop(n, points, dur, fn):
    """n closed loops of `points` keyframes each, fn(angle) -> dict of K fields."""
    out = []
    for k in range(1, n * points + 1):
        out.append(K(dur, **fn(2 * math.pi * k / points)))
    return out


# --- Motions. n = repeat count, side = +1 (left) or -1 (right). ----------------

def _nod(n, side):
    # Whole arm: rear back with the hand high, then throw shoulder and forearm down.
    return ([R(), K(.22, s=-.35, u=.5)]
            + [K(.24, s=.8, u=-.9, g=.25), K(.24, s=-.3, u=.45)] * n + [R(.4)])


def _shake(n, side):
    return ([R(), K(.2, s=-.2, u=.3)]
            + [K(.3, b=.55, s=.15, u=.25), K(.3, b=-.55, s=.15, u=.25)] * n + [R(.4)])


def _wave(n, side):
    return ([R(), K(.5, b=.5 * side, u=.7, g=.6)]
            + [K(.25, b=.85 * side, u=.75, g=.9), K(.25, b=.3 * side, u=.75, g=.9)] * n
            + [R(.6)])


def _hello(n, side):
    return ([R(), K(.5, u=.7, g=.7)]
            + [K(.25, b=.35, u=.75, g=.9), K(.25, b=-.35, u=.75, g=.9)] * n
            + [K(.3, u=.5, g=.4), K(.25, u=-.3), R(.5)])


def _goodbye(n, side):
    return ([R(), K(.6, u=.9, g=1)]
            + [K(.4, b=.7, u=.85, g=1), K(.4, b=-.7, u=.85, g=1)] * n
            + [K(.5, u=.3, g=.3), R(.6)])


def _bow(n, side):
    return [R(), K(.3, s=-.3, u=.2), K(.7, s=.9, u=-.8), K(.5, s=.95, u=-.85),
            K(.6, s=-.15, u=.1), R(.4)]


def _shrug(n, side):
    return [R()] + [K(.25, s=-.6, u=.6, g=.8), K(.35, s=-.5, u=.5, g=.8), R(.3)] * n


def _point(n, side):
    return [R(), K(.2, b=-.1 * side), K(.45, b=.95 * side, s=.4, u=-.1),
            K(.2, b=.85 * side, s=.35, u=-.1), K(.6, b=.85 * side, s=.35, u=-.1), R(.6)]


def _point_forward(n, side):
    return [R(), K(.2, s=-.3), K(.4, s=.9, u=.1), K(.2, s=.8, u=.05), K(.6, s=.8, u=.05), R(.5)]


def _point_up(n, side):
    return [R(), K(.2, u=-.2), K(.4, s=-.8, u=1), K(.2, s=-.7, u=.9), K(.6, s=-.7, u=.9), R(.5)]


def _point_down(n, side):
    return [R(), K(.2, u=.2), K(.4, s=.5, u=-1), K(.2, s=.45, u=-.9), K(.6, s=.45, u=-.9), R(.5)]


def _look(n, side):
    return [R(), K(.6, b=.8 * side, u=.2), K(.7, b=.8 * side, u=.2), R(.6)]


def _look_up(n, side):
    return [R(), K(.6, s=-.5, u=.8), K(.7, s=-.5, u=.8), R(.6)]


def _look_down(n, side):
    return [R(), K(.6, s=.3, u=-.8), K(.7, s=.3, u=-.8), R(.6)]


def _look_around(n, side):
    return ([R()] + [K(.7, b=.9, u=.3), K(.4, b=.9, u=.3),
                     K(.9, b=-.9, u=.3), K(.4, b=-.9, u=.3)] * n + [R(.7)])


def _clap(n, side):
    return [R(), K(.3, u=.4, g=.9)] + [K(.15, u=.4, g=.1), K(.15, u=.4, g=.9)] * n + [R(.4)]


def _grab(n, side):
    return [R(), K(.4, s=.2, u=.2, g=1), K(.5, s=.9, u=-.3, g=1), K(.3, s=.9, u=-.3),
            K(.5, s=-.2, u=.3), R(.5)]


def _give(n, side):
    return [R(), K(.3, s=-.3), K(.6, s=.9, u=.1), K(.4, s=.9, u=.1, g=1),
            K(.7, s=.9, u=.1, g=1), R(.6)]


def _stretch(n, side):
    return [R(), K(.8, s=-1, u=1, g=1), K(.6, s=-1, u=1, g=1), K(.8, s=1, u=-.4, g=1),
            K(.5, s=1, u=-.4, g=1), R(.8)]


def _sleep(n, side):
    return [R(), K(1.2, s=.5, u=-.7), K(1.2, s=.9, u=-1), K(1.0, s=.95, u=-1)]


def _wake(n, side):
    return [K(0, s=.95, u=-1), K(.5, s=.9, u=-.9), K(.6, s=-.3, u=.6, g=.6),
            K(.3, s=.1, u=-.1), R(.4)]


def _yawn(n, side):
    return [R(), K(.9, s=-.6, u=.8, g=1), K(.8, s=-.7, u=.9, g=1), K(.9, u=-.2, g=.2), R(.6)]


def _celebrate(n, side):
    return ([R(), K(.2, u=-.3), K(.3, s=-.6, u=1, g=1)]
            + [K(.2, b=.4, s=-.6, u=1, g=.2), K(.2, b=-.4, s=-.6, u=1, g=1)] * n
            + [R(.5)])


def _sad(n, side):
    return [R(), K(1.0, s=.5, u=-.8), K(.5, b=.2, s=.6, u=-.9), K(.5, b=-.2, s=.6, u=-.9),
            K(.6, s=.6, u=-.9), R(1.0)]


def _angry(n, side):
    return ([R(), K(.2, s=-.4, u=.3)]
            + [K(.15, s=.7, u=-.2, g=.8), K(.15, s=-.2, u=.2)] * n + [R(.4)])


def _happy(n, side):
    return ([R(), K(.25, u=.5, g=.6)]
            + [K(.18, b=.3, u=.7, g=.8), K(.18, b=-.3, u=.4, g=.4)] * n + [R(.4)])


def _scared(n, side):
    return ([R(), K(.15, s=-1, u=.8, g=1)]
            + [K(.08, b=.08, s=-1, u=.8, g=1), K(.08, b=-.08, s=-1, u=.8, g=1)] * n
            + [K(.8, s=-.5, u=.4, g=.5), R(.8)])


def _curious(n, side):
    return [R(), K(.8, s=.6, u=.2, g=.3), K(.5, b=.3 * side, s=.7, u=.1, g=.5),
            K(.6, b=.3 * side, s=.7, u=.1, g=.5), R(.7)]


def _think(n, side):
    b = .4 * side
    return [R(), K(.7, b=b, s=-.4, u=.6), K(.4, b=b, s=-.4, u=.45), K(.4, b=b, s=-.4, u=.6),
            K(.4, b=b, s=-.4, u=.45), K(.5, b=b, s=-.4, u=.6), R(.6)]


def _confused(n, side):
    return [R(), K(.4, b=.3, u=.3), K(.4, b=-.3, u=.4, g=.5), K(.3, b=.2, u=.2),
            K(.5, s=-.3, u=.5, g=.7), R(.5)]


def _circle(n, side):
    return [R()] + _loop(n, 8, .18, lambda a: dict(
        b=.6 * side * math.sin(a), u=.5 * (1 - math.cos(a)))) + [R(.3)]


def _figure_eight(n, side):
    return [R()] + _loop(n, 12, .15, lambda a: dict(
        b=.7 * math.sin(a), u=.4 * math.sin(2 * a))) + [R(.3)]


def _stir(n, side):
    return ([R(), K(.5, s=.7, u=-.7)]
            + _loop(n, 8, .15, lambda a: dict(
                b=.25 * side * math.sin(a), s=.5 + .2 * math.cos(a), u=-.7))
            + [R(.6)])


def _tap(n, side):
    return [R(), K(.4, s=.5, u=-.5)] + [K(.12, s=.6, u=-.9), K(.12, s=.5, u=-.5)] * n + [R(.5)]


def _sweep(n, side):
    return [R(), K(.6, b=-.95 * side, s=.4, u=-.5), K(1.4, b=.95 * side, s=.4, u=-.5), R(.7)]


def _swipe(n, side):
    return [R(), K(.3, b=-.5 * side, u=.3), K(.25, b=.8 * side, u=.3),
            K(.3, b=.6 * side, u=.2), R(.5)]


def _salute(n, side):
    return [R(), K(.35, b=.3 * side, s=-.5, u=.9), K(.15, b=.4 * side, s=-.5, u=.95),
            K(.6, b=.4 * side, s=-.5, u=.95), K(.2, b=.1 * side, s=.1, u=.3), R(.4)]


def _beckon(n, side):
    return ([R(), K(.4, s=.7, u=-.1, g=.8)]
            + [K(.3, s=.3, u=.6, g=.2), K(.3, s=.7, u=-.1, g=.8)] * n + [R(.5)])


def _halt(n, side):
    return [R(), K(.3, s=.5, u=.8, g=1), K(.15, s=.6, u=.85, g=1), K(1.0, s=.6, u=.85, g=1), R(.6)]


def _push(n, side):
    return [R()] + [K(.2, s=-.5, u=.2), K(.2, s=1), K(.3, s=.2)] * n + [R(.4)]


def _pull(n, side):
    return [R(), K(.4, s=.9, u=-.1, g=1), K(.2, s=.9, u=-.1), K(.35, s=-.8, u=.4), R(.5)]


def _laugh(n, side):
    return ([R(), K(.25, s=-.4, u=.5, g=.8)]
            + [K(.1, s=-.3, u=.25, g=.6), K(.1, s=-.4, u=.55, g=.9)] * n + [R(.5)])


def _cry(n, side):
    return ([R(), K(.8, s=.7, u=-.9)]
            + [K(.3, s=.75, u=-.75), K(.3, s=.7, u=-1)] * n + [R(1.0)])


def _surprise(n, side):
    return [R(), K(.12, s=-.9, u=.9, g=1), K(.15, s=-.7, u=.75, g=1), K(.8, s=-.75, u=.8, g=1), R(.7)]


def _approve(n, side):
    return [R(), K(.4, s=-.3, u=.8, g=.7), K(.15, s=-.3, u=.8), K(.15, s=-.3, u=.8, g=.7),
            K(.15, s=-.3, u=.8), K(.4, s=-.3, u=.8), R(.5)]


def _bounce(n, side):
    return [R()] + [K(.2, s=-.3, u=.6), K(.2, s=.3, u=-.4)] * n + [R(.3)]


def _sway(n, side):
    return [R()] + [K(.7, b=.5, s=.3, u=.2), K(.7, b=-.5, s=.3, u=.2)] * n + [R(.7)]


def _zigzag(n, side):
    return ([R()] + [K(.25, b=.6, u=.6), K(.25, b=.2, u=-.5),
                     K(.25, b=-.2, u=.6), K(.25, b=-.6, u=-.5)] * n + [R(.4)])


def _heartbeat(n, side):
    return ([R(), K(.3, u=.3, g=.3)]
            + [K(.1, u=.4, g=.9), K(.12, u=.3, g=.3), K(.1, u=.38, g=.8), K(.4, u=.3, g=.3)] * n
            + [R(.4)])


def _peek(n, side):
    return [R(), K(1.0, b=.7 * side, s=.4, u=.3), K(.6, b=.75 * side, s=.45, u=.3),
            K(.15, s=-.2), R(.4)]


def _wiggle(n, side):
    return [R()] + [K(.12, b=.2), K(.12, b=-.2)] * n + [R(.3)]


def _dance(n, side):
    return ([R(), K(.4, s=-.25, u=-.2)]
            + [K(.4, b=.5, s=.5, u=.6, g=.7), K(.35, b=.25, s=.2, u=.2, g=.4),
               K(.4, b=-.5, s=.4, u=.7, g=.8), K(.35, b=-.3, s=.1, u=.1, g=.2)] * n
            + [R(.6)])


def _twist(n, side):
    return [R()] + [K(.5, b=1, u=.3), K(.8, b=-1, u=.3)] * n + [K(.4, b=.4), R(.4)]


def _flex(n, side):
    return [R(), K(.5, s=-.6, u=1), K(.2, s=-.65, u=1), K(.5, s=-.65, u=1), R(.5)]


def _proud(n, side):
    return [R(), K(.8, s=-.8, u=.6), K(1.0, s=-.8, u=.6), R(.7)]


def _shy(n, side):
    b = .6 * side
    return [R(), K(.7, b=b, s=.4, u=-.6), K(.8, b=b, s=.4, u=-.6), K(.5, b=.4 * side, s=.3, u=-.4),
            K(.4, b=b, s=.4, u=-.6), R(.8)]


def _bored(n, side):
    return [R(), K(1.0, s=.4, u=-.5), K(.6, b=.3, s=.4, u=-.5), K(.6, b=-.3, s=.4, u=-.5),
            K(.8, s=.4, u=-.5), R(.8)]


def _hug(n, side):
    return [R(), K(.6, s=.8, u=.2, g=1), K(.5, s=.3, u=.5), K(.6, s=.3, u=.5), R(.6)]


def _high_five(n, side):
    return [R(), K(.3, s=-.4, u=.9, g=1), K(.15, s=.3, u=.9, g=1), K(.2, s=-.2, u=.9, g=1), R(.5)]


def _fist_bump(n, side):
    return [R(), K(.3, s=-.4), K(.2, s=1), K(.25, s=.5), R(.4)]


def _handshake(n, side):
    return ([R(), K(.5, s=.9, g=.8), K(.2, s=.9, g=.2)]
            + [K(.18, s=.9, u=.3, g=.2), K(.18, s=.9, u=-.3, g=.2)] * n
            + [K(.3, s=.9, g=.8), R(.5)])


def _lean_forward(n, side):
    return [R(), K(.7, s=1), K(.8, s=1), R(.7)]


def _lean_back(n, side):
    return [R(), K(.7, s=-1), K(.8, s=-1), R(.7)]


def _open(n, side):
    return [R(), K(.5, g=1)]


def _close(n, side):
    return [K(0, g=1), R(.5)]


def _home(n, side):
    return [R(), R(.3)]


@dataclass(frozen=True)
class Motion:
    fn: Callable
    reps: Optional[int] = None  # default repeat count; None = not a repeating motion
    sided: bool = False  # has distinct left/right versions
    about: str = ""
    learned: bool = False  # came from gestures/learned.json


MOTIONS = {
    "nod": Motion(_nod, 2, about="yes"),
    "shake": Motion(_shake, 2, about="no"),
    "wave": Motion(_wave, 3, sided=True),
    "hello": Motion(_hello, 2, about="wave, then a small nod"),
    "goodbye": Motion(_goodbye, 3, about="big slow wave"),
    "bow": Motion(_bow, about="thanks, sorry"),
    "shrug": Motion(_shrug, 1, about="don't know"),
    "point": Motion(_point, sided=True),
    "point_forward": Motion(_point_forward),
    "point_up": Motion(_point_up),
    "point_down": Motion(_point_down),
    "look": Motion(_look, sided=True),
    "look_up": Motion(_look_up),
    "look_down": Motion(_look_down),
    "look_around": Motion(_look_around, 1),
    "clap": Motion(_clap, 4, about="gripper claps"),
    "grab": Motion(_grab, about="reach, close, retract"),
    "give": Motion(_give, about="reach out and open"),
    "pull": Motion(_pull),
    "push": Motion(_push, 1),
    "stretch": Motion(_stretch),
    "sleep": Motion(_sleep, about="droops and stays down"),
    "wake": Motion(_wake),
    "yawn": Motion(_yawn),
    "celebrate": Motion(_celebrate, 4),
    "happy": Motion(_happy, 4),
    "laugh": Motion(_laugh, 6),
    "sad": Motion(_sad),
    "cry": Motion(_cry, 3),
    "angry": Motion(_angry, 3),
    "scared": Motion(_scared, 5),
    "surprise": Motion(_surprise),
    "curious": Motion(_curious, sided=True),
    "think": Motion(_think, sided=True),
    "confused": Motion(_confused),
    "proud": Motion(_proud),
    "shy": Motion(_shy, sided=True),
    "bored": Motion(_bored),
    "approve": Motion(_approve, about="raised gripper pinch, like a thumbs up"),
    "heartbeat": Motion(_heartbeat, 3, about="love"),
    "hug": Motion(_hug),
    "high_five": Motion(_high_five),
    "fist_bump": Motion(_fist_bump),
    "handshake": Motion(_handshake, 3),
    "salute": Motion(_salute, sided=True),
    "beckon": Motion(_beckon, 3, about="come here"),
    "halt": Motion(_halt, about="raised open hand"),
    "circle": Motion(_circle, 2, sided=True),
    "figure_eight": Motion(_figure_eight, 2),
    "stir": Motion(_stir, 3, sided=True),
    "tap": Motion(_tap, 3, about="knock"),
    "sweep": Motion(_sweep, sided=True),
    "swipe": Motion(_swipe, sided=True),
    "peek": Motion(_peek, sided=True),
    "bounce": Motion(_bounce, 3),
    "sway": Motion(_sway, 2),
    "zigzag": Motion(_zigzag, 2),
    "wiggle": Motion(_wiggle, 4),
    "dance": Motion(_dance, 2),
    "twist": Motion(_twist, 2),
    "flex": Motion(_flex),
    "lean_forward": Motion(_lean_forward),
    "lean_back": Motion(_lean_back),
    "open": Motion(_open, about="open the gripper"),
    "close": Motion(_close, about="close the gripper"),
    "home": Motion(_home, about="rest pose"),
}


# --- Learned motions: added at run time, stored in gestures/learned.json --------

LEARNED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "gestures", "learned.json")
LEARNED_KEYWORDS = {}  # motion name -> trigger phrases, read by roarm_rl.intent
MAX_KEYFRAMES = 60
MAX_SECONDS = 15.0


def clean_keyframes(raw):
    """Validate [[dur, b, s, u, g], ...] and clamp it into the normalized range."""
    if not isinstance(raw, list) or not 2 <= len(raw) <= MAX_KEYFRAMES:
        raise ValueError(f"need 2..{MAX_KEYFRAMES} keyframes")
    out = []
    for i, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != 5:
            raise ValueError(f"keyframe {i} must be [dur, b, s, u, g]")
        dur, b, s, u, g = (float(v) for v in row)
        if not all(math.isfinite(v) for v in (dur, b, s, u, g)):
            raise ValueError(f"keyframe {i} has a non-finite value")
        dur = 0.0 if i == 0 else max(0.08, min(3.0, dur))
        b, s, u = (max(-1.0, min(1.0, v)) for v in (b, s, u))
        out.append([round(dur, 3), round(b, 3), round(s, 3), round(u, 3),
                    round(max(0.0, min(1.0, g)), 3)])
    if sum(row[0] for row in out) > MAX_SECONDS:
        raise ValueError(f"longer than {MAX_SECONDS:.0f} s")
    return out


def _register(name, about, keywords, rows):
    norm = [K(*row) for row in rows]
    MOTIONS[name] = Motion(lambda n, side, norm=norm: list(norm), about=about, learned=True)
    LEARNED_KEYWORDS[name] = list(keywords)


def _read_learned():
    try:
        with open(LEARNED_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"motions": {}, "aliases": {}}


def _write_learned(data):
    os.makedirs(os.path.dirname(LEARNED_PATH), exist_ok=True)
    tmp = LEARNED_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        text = json.dumps(data, indent=1)
        # keep each keyframe on one line
        text = re.sub(r"\[\s+(-?[\d.]+(?:,\s+-?[\d.]+)*)\s+\]",
                      lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
        f.write(text + "\n")
    os.replace(tmp, LEARNED_PATH)


def load_learned():
    data = _read_learned()
    for name, m in data.get("motions", {}).items():
        try:
            _register(name, m.get("about", ""), m.get("keywords", []), clean_keyframes(m["keyframes"]))
        except (KeyError, ValueError, TypeError) as e:
            print(f"[library] skipping learned motion '{name}': {e}")
    for phrase, name in data.get("aliases", {}).items():
        if name in MOTIONS:
            LEARNED_KEYWORDS.setdefault(name, []).append(phrase)


def add_learned(name, about, keywords, keyframes):
    """Validate, register and save a new motion. Returns the name it was stored under."""
    name = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")[:40] or "motion"
    base, k = name, 2
    while name in MOTIONS:
        name, k = f"{base}_{k}", k + 1
    rows = clean_keyframes(keyframes)
    keywords = sorted({str(kw).lower().strip() for kw in keywords if str(kw).strip()})
    _register(name, str(about)[:120], keywords, rows)
    data = _read_learned()
    data.setdefault("motions", {})[name] = {"about": str(about)[:120], "keywords": keywords,
                                            "keyframes": rows}
    _write_learned(data)
    return name


def add_alias(phrase, name):
    """Remember that `phrase` means the existing motion `name`."""
    if name not in MOTIONS:
        raise KeyError(name)
    phrase = str(phrase).lower().strip()
    LEARNED_KEYWORDS.setdefault(name, []).append(phrase)
    data = _read_learned()
    data.setdefault("aliases", {})[phrase] = name
    _write_learned(data)


def _to_radians(norm, amp):
    b, s, u, g = norm
    b, s, u = (max(-1.0, min(1.0, v * amp)) for v in (b, s, u))
    (b_lo, b_hi), (s_lo, s_hi), (e_lo, e_hi), (g_lo, g_hi) = SAFE_BOUNDS
    return [
        b * (b_hi if b > 0 else -b_lo),
        s * (s_hi if s > 0 else -s_lo),
        # raising the hand lowers the elbow angle
        REST_ELBOW - u * ((REST_ELBOW - e_lo) if u > 0 else (e_hi - REST_ELBOW)),
        g_lo + max(0.0, min(1.0, g)) * (g_hi - g_lo),
    ]


def build(name, speed="normal", size="normal", side="left", reps=None):
    """Radian keyframes [(duration_s, [base, shoulder, elbow, gripper]), ...]."""
    motion = MOTIONS[name]
    sign = -1 if (motion.sided and side == "right") else 1
    if motion.reps is None:
        norm = motion.fn(1, sign)
        if reps and reps > 1:
            norm = norm + norm[1:] * (min(reps, 3) - 1)
    else:
        n = motion.reps if reps is None else max(1, min(MAX_REPS, reps))
        norm = motion.fn(n, sign)
    tempo, amp = SPEEDS[speed], SIZES[size]
    return [(dur * tempo, _to_radians(pose, amp)) for dur, pose in norm]


def variant_name(name, speed="normal", size="normal", side="left"):
    parts = [name]
    if MOTIONS[name].sided and side == "right":
        parts.append("right")
    parts += [m for m in (speed, size) if m != "normal"]
    return "_".join(parts)


def catalog():
    """{variant name: keyframes} for every speed/size/side combination."""
    out = {}
    for name, motion in MOTIONS.items():
        for side in (("left", "right") if motion.sided else ("left",)):
            for speed in SPEEDS:
                for size in SIZES:
                    out[variant_name(name, speed, size, side)] = build(name, speed, size, side)
    return out


load_learned()
