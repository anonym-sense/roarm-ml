"""Invent a gesture for a phrase the library has no match for.

Hands the phrase to an external language-model command, which either picks
an existing motion or writes new keyframes in roarm_rl.library's normalized
units. The answer is validated, clamped to the safe range, and saved to
gestures/learned.json, so the same phrase is an instant lookup from then on.

The command is whatever you configure: any program that reads a prompt on
stdin and prints the model's reply on stdout. Put it in designer.local.json
next to this package's parent folder (the file is git-ignored):

    {"command": ["my-llm-cli", "--some-flag"]}

or set ROARM_DESIGNER_CMD to the same thing as one string. Nothing is
configured by default, and no key is read or stored by this code.

    python -m roarm_rl.composer "act like a cat stretching after a nap"
    python -m roarm_rl.composer --add my_motion.json     # add a hand-written motion

In the GUI this runs on a worker thread: Composer.request() returns at once
and Composer.poll() hands back finished results.
"""

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading

from roarm_rl import intent, library

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "designer.local.json")
TIMEOUT_S = 90


def _config():
    """{"command": [...], "drop_env_prefixes": [...]} from the env var or the local file."""
    env_cmd = os.environ.get("ROARM_DESIGNER_CMD")
    if env_cmd:
        return {"command": shlex.split(env_cmd)}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return {}
    return cfg if isinstance(cfg.get("command"), list) and cfg["command"] else {}

PROMPT = """You design gestures for a small 4-joint desktop robot arm (Waveshare RoArm-M2). \
It has no face: the gripper at the end reads as its head/hand, the whole arm as its body.

Someone said to the robot: "{phrase}"

A gesture is a list of keyframes [dur, b, s, u, g]:
  dur  seconds to reach this pose from the previous one (0.08 to 3.0; ignored for the first)
  b    base turn      -1 .. 1   + is the arm's left, - its right
  s    shoulder lean  -1 .. 1   + leans forward and down, - leans back
  u    forearm        -1 .. 1   + raises the hand high, - lowers it toward the table
  g    gripper         0 .. 1   0 closed, 1 wide open
All zeros is the rest pose: upright upper arm, forearm level, hand out in front.
Poses are joined by a smooth spline, so velocity carries through each keyframe.

Make it read clearly from across a room:
- Use the whole arm. Combine shoulder and forearm; reach values of 0.7 to 1.0 on the main beats.
- Anticipation: a short move the opposite way before a big move.
- Overshoot, then settle back.
- Let the gripper follow or punctuate the motion.
- Timing carries the mood: 0.1-0.25 s per keyframe is snappy or angry, 0.6-1.2 s is calm or sad.
- Start at rest and end at rest ([.., 0, 0, 0, 0]). 6 to 30 keyframes, under 12 seconds total.

Examples:
  strong nod:  [[0,0,0,0,0],[0.22,0,-0.35,0.5,0],[0.24,0,0.8,-0.9,0.25],[0.24,0,-0.3,0.45,0],\
[0.24,0,0.8,-0.9,0.25],[0.24,0,-0.3,0.45,0],[0.4,0,0,0,0]]
  bow:         [[0,0,0,0,0],[0.3,0,-0.3,0.2,0],[0.7,0,0.9,-0.8,0],[0.5,0,0.95,-0.85,0],\
[0.6,0,-0.15,0.1,0],[0.4,0,0,0,0]]
  celebrate:   [[0,0,0,0,0],[0.2,0,0,-0.3,0],[0.3,0,-0.6,1,1],[0.2,0.4,-0.6,1,0.2],\
[0.2,-0.4,-0.6,1,1],[0.2,0.4,-0.6,1,0.2],[0.2,-0.4,-0.6,1,1],[0.5,0,0,0,0]]

Motions that already exist: {existing}

If one of those already expresses what was said, reuse it. Otherwise design a new one.
Reply with one JSON object and nothing else, in one of these two forms:
  {{"use": "<existing motion name>"}}
  {{"name": "<short_snake_case_name>", "about": "<one line>", \
"keywords": ["<2-5 short phrases someone might say to ask for this>"], \
"keyframes": [[dur, b, s, u, g], ...]}}
"""


def available():
    command = _config().get("command")
    return bool(command) and shutil.which(command[0]) is not None


def _ask(phrase):
    prompt = PROMPT.format(
        phrase=phrase.replace('"', "'"),
        existing=", ".join(sorted(library.MOTIONS)),
    )
    cfg = _config()
    drop = tuple(cfg.get("drop_env_prefixes", []))
    env = {k: v for k, v in os.environ.items() if not (drop and k.startswith(drop))}
    proc = subprocess.run(
        cfg["command"], input=prompt, capture_output=True, text=True, encoding="utf-8",
        timeout=TIMEOUT_S, env=env, cwd=os.path.expanduser("~"),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "designer command failed")
    answer = proc.stdout
    try:  # some CLIs wrap the reply in a JSON envelope with the text under "result"
        envelope = json.loads(answer)
        if isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
            answer = envelope["result"]
    except ValueError:
        pass
    found = re.search(r"\{.*\}", answer, re.DOTALL)
    if not found:
        raise ValueError("no JSON in the reply")
    return json.loads(found.group(0))


def _ends_at_rest(rows):
    return all(abs(v) < 1e-6 for v in rows[-1][1:])


def learn(phrase, spec):
    """Store a designer reply. Returns (motion name, True if newly created)."""
    if "use" in spec:
        name = str(spec["use"]).strip().lower().replace(" ", "_")
        library.add_alias(phrase, name)
        intent.reindex()
        return name, False
    rows = library.clean_keyframes(spec["keyframes"])
    if not _ends_at_rest(rows):
        rows.append([0.6, 0, 0, 0, 0])
    keywords = list(spec.get("keywords", [])) + [phrase]
    name = library.add_learned(spec.get("name", "motion"), spec.get("about", phrase), keywords, rows)
    intent.reindex()
    return name, True


def compose(phrase):
    """Blocking: design (or find) a motion for `phrase` and remember it."""
    return learn(phrase, _ask(phrase))


class Composer:
    """Runs compose() off the GUI thread, one phrase at a time."""

    def __init__(self):
        self._done = queue.Queue()
        self._busy = False

    @property
    def busy(self):
        return self._busy

    def request(self, phrase):
        if self._busy or not available():
            return False
        self._busy = True
        threading.Thread(target=self._work, args=(phrase,), daemon=True).start()
        return True

    def _work(self, phrase):
        try:
            name, created = compose(phrase)
            self._done.put((phrase, name, created, None))
        except Exception as e:
            self._done.put((phrase, None, False, str(e)[:200]))

    def poll(self):
        """Finished jobs as (phrase, motion name or None, created, error or None)."""
        out = []
        while True:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                break
        if out:
            self._busy = False
        return out


def main():
    ap = argparse.ArgumentParser(description="Teach the arm a new gesture.")
    ap.add_argument("phrase", nargs="?", help="what the gesture should express")
    ap.add_argument("--add", metavar="FILE",
                    help="add a hand-written motion: JSON with name, about, keywords, keyframes")
    args = ap.parse_args()

    if args.add:
        with open(args.add, encoding="utf-8") as f:
            spec = json.load(f)
        name, _ = learn(spec.get("keywords", [spec.get("name", "motion")])[0], spec)
    elif args.phrase:
        if not available():
            ap.error("no designer command configured; see the top of roarm_rl/composer.py")
        name, created = compose(args.phrase)
        print("designed a new motion" if created else "matched an existing motion")
    else:
        ap.error("give a phrase or --add FILE")
    print(f"{name}: preview with  python -m roarm_rl.gesture {name}")


if __name__ == "__main__":
    main()
