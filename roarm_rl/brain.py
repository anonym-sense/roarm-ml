"""What the arm learns from being used.

Three kinds of learning, all from ordinary operation:

  preferences  Every thumbs up / down (and every "stop" mid-gesture) is a
               reward for the speed and size that was played. A bandit keeps
               a success estimate per (motion, speed, size) and overall, and
               when the words leave speed or size open it usually plays the
               best-rated variant and sometimes tries a neighbouring one.
  skills       Named sequences: "remember that as greeting", or
               "learn greeting: wave, then bow". Saying the name runs them.
  corrections  "no, I meant wave" re-plays, and remembers that the previous
               phrase means wave.

Skills and phrase corrections are stored with the learned motions in
gestures/learned.json. The interaction log and the preference counts live in
data/, which is git-ignored because it records what was said.
"""

import json
import os
import random
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from roarm_rl import intent, library
from roarm_rl.library import SIZES, SPEEDS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("ROARM_DATA_DIR") or os.path.join(_ROOT, "data")
LOG_PATH = os.path.join(DATA_DIR, "experience.jsonl")
PREFS_PATH = os.path.join(DATA_DIR, "preferences.json")

EXPLORE = 0.1  # chance of trying a neighbouring variant instead of the best one
PRIOR_DOWNS = 1.0  # before any feedback an arm counts one dislike and 1 to 5 likes (see _rate)
GLOBAL_WEIGHT = 0.5  # how much the overall taste counts next to the per-motion record

_TEACH = [
    re.compile(r"^(?:learn|teach yourself|new skill)\s+(?:the skill\s+)?(.+?)\s*[:=]\s*(.+)$", re.I),
    re.compile(r"^when i say\s+(.+?)[, ]+(?:you\s+)?(?:do|then)\s+(.+)$", re.I),
]
_REMEMBER = re.compile(r"^(?:remember|save|call|name)\s+(?:that|this|it)\s+(?:as\s+)?(.+)$", re.I)
_CORRECT = re.compile(
    r"^(?:no|nope|wrong)[,.! ]+(?:i (?:meant|mean|said)|do|it should be|that should be|"
    r"that means|it means)\s+(.+)$", re.I)
_FORGET = re.compile(r"^forget (?:the skill\s+)?(.+)$", re.I)


@dataclass
class Event:
    """One thing the arm was told, and what it did about it."""
    id: int
    text: str
    reply: str
    gestures: list = field(default_factory=list)  # intent.Gesture
    pending: str = ""  # phrase handed to the composer, if any
    rating: int = 0
    kind: str = "command"  # command, skill, learn, correction, info


def _clean(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("'", "")))


class Brain:
    def __init__(self):
        self._lock = threading.RLock()
        self._next_id = 1
        self._events = {}  # id -> Event, recent only
        self._recent = deque(maxlen=200)
        self._prefs = self._load_prefs()
        self._last_command = None  # last Event that played gestures

    # --- storage ---------------------------------------------------------------

    def _load_prefs(self):
        try:
            with open(PREFS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"motions": {}, "overall": {}}

    def _save_prefs(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = PREFS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._prefs, f, indent=1)
        os.replace(tmp, PREFS_PATH)

    def _log(self, kind, **fields):
        os.makedirs(DATA_DIR, exist_ok=True)
        row = {"t": round(time.time(), 1), "kind": kind, **fields}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _read_log(self):
        try:
            with open(LOG_PATH, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except (OSError, ValueError):
            return []

    # --- preferences (bandit) ----------------------------------------------------

    @staticmethod
    def _key(speed, size):
        return f"{speed}/{size}"

    def _rate(self, table, speed, size):
        ups, downs = table.get(self._key(speed, size), (0, 0))
        # with no feedback yet, "normal" wins on each axis the words left open
        prior_up = 1.0 + 2 * (speed == "normal") + 2 * (size == "normal")
        return (ups + prior_up) / (ups + downs + prior_up + PRIOR_DOWNS)

    def _score(self, motion, speed, size):
        own = self._rate(self._prefs["motions"].get(motion, {}), speed, size)
        overall = self._rate(self._prefs["overall"], speed, size)
        return own + GLOBAL_WEIGHT * (overall - 0.5)

    def choose(self, motion, speed, size):
        """Pick the speed/size the words left open: mostly the best-rated, sometimes a neighbour."""
        with self._lock:
            arms = [(sp, sz) for sp in ([speed] if speed else SPEEDS)
                    for sz in ([size] if size else SIZES)]
            arms.sort(key=lambda arm: (arm[0] != "normal") + (arm[1] != "normal"))  # ties go to normal
            best = max(arms, key=lambda arm: self._score(motion, *arm))
            if len(arms) > 1 and random.random() < EXPLORE:
                order_sp, order_sz = list(SPEEDS), list(SIZES)
                near = [a for a in arms if a != best
                        and abs(order_sp.index(a[0]) - order_sp.index(best[0]))
                        + abs(order_sz.index(a[1]) - order_sz.index(best[1])) == 1]
                if near:
                    return random.choice(near)
            return best

    def _reward(self, gesture, value):
        for table in (self._prefs["motions"].setdefault(gesture.motion, {}), self._prefs["overall"]):
            ups, downs = table.get(self._key(gesture.speed, gesture.size), (0, 0))
            table[self._key(gesture.speed, gesture.size)] = (
                [ups + value, downs] if value > 0 else [ups, downs - value])

    def feedback(self, event_id, value):
        """value: +1 liked, -1 disliked. Returns a short acknowledgement."""
        with self._lock:
            event = self._events.get(event_id)
            if event is None or not event.gestures:
                return "Nothing to rate there."
            value = 1 if value > 0 else -1
            for g in event.gestures:
                self._reward(g, value)
            event.rating = value
            self._save_prefs()
            self._log("feedback", text=event.text, value=value,
                      gestures=[[g.motion, g.speed, g.size] for g in event.gestures])
            return "Noted, more like that." if value > 0 else "Noted, I'll do that differently."

    def stopped(self, label):
        """A gesture was cancelled mid-way: count it as a mild dislike."""
        with self._lock:
            event = self._last_command
            if event is None:
                return
            for g in event.gestures:
                if g.label == label:
                    self._reward(g, -0.5)
                    self._save_prefs()
                    self._log("stopped", text=event.text, gesture=[g.motion, g.speed, g.size])

    # --- skills ------------------------------------------------------------------

    @staticmethod
    def skills():
        return library._read_learned().get("skills", {})

    def _save_skill(self, name, steps):
        name = _clean(name)
        if not name or not steps:
            return None
        data = library._read_learned()
        data.setdefault("skills", {})[name] = {"steps": steps, "learned": time.strftime("%Y-%m-%d")}
        library._write_learned(data)
        self._log("skill", name=name, steps=steps)
        return name

    def forget_skill(self, name):
        data = library._read_learned()
        if data.get("skills", {}).pop(_clean(name), None) is None:
            return False
        library._write_learned(data)
        self._log("forget", name=_clean(name))
        return True

    def _match_skill(self, cleaned):
        skills = self.skills()
        for name in sorted(skills, key=len, reverse=True):
            if cleaned == name or cleaned in (f"do {name}", f"run {name}", f"do the {name}",
                                              f"show me {name}", f"{name} please"):
                return name, skills[name]["steps"]
        return None, None

    # --- the main entry point ------------------------------------------------------

    def _event(self, text, reply, gestures=(), kind="command", pending=""):
        event = Event(self._next_id, text, reply, list(gestures), pending, kind=kind)
        self._next_id += 1
        self._events[event.id] = event
        self._recent.append(event)
        if len(self._events) > 400:
            for old in sorted(self._events)[:200]:
                del self._events[old]
        if event.gestures and kind in ("command", "skill", "correction"):
            self._last_command = event
        self._log(kind, text=text, reply=reply,
                  gestures=[[g.motion, g.speed, g.size, g.chosen] for g in event.gestures])
        return event

    def _interpret_steps(self, steps):
        gestures = []
        for step in steps:
            gestures += intent.interpret(step, self.choose).gestures
        return gestures

    def handle(self, text, can_compose=False):
        """Text in -> Event out. The caller plays event.gestures and, when
        event.pending is set, asks the composer to invent that phrase."""
        with self._lock:
            text = text.strip()
            cleaned = _clean(text)

            for pattern in _TEACH:
                m = pattern.match(text)
                if m:
                    steps = [s.strip() for s in re.split(r"\bthen\b|,|;", m.group(2)) if s.strip()]
                    gestures = self._interpret_steps(steps)
                    if not gestures:
                        return self._event(text, "I couldn't make sense of those steps.", kind="info")
                    name = self._save_skill(m.group(1), steps)
                    return self._event(text, f"Learned the skill \"{name}\": "
                                       + "  >  ".join(g.label for g in gestures), gestures, "learn")

            m = _REMEMBER.match(text)
            if m:
                last = self._last_command
                if last is None:
                    return self._event(text, "Do something first, then tell me to remember it.",
                                       kind="info")
                name = self._save_skill(m.group(1), [last.text])
                return self._event(text, f"Saved \"{last.text}\" as the skill \"{name}\".", kind="learn")

            m = _FORGET.match(text)
            if m and _clean(m.group(1)) in self.skills():
                self.forget_skill(m.group(1))
                return self._event(text, f"Forgot the skill \"{_clean(m.group(1))}\".", kind="learn")

            m = _CORRECT.match(text)
            if m and self._last_command is not None:
                result = intent.interpret(m.group(1), self.choose)
                if len(result.gestures) == 1:
                    wrong = self._last_command
                    for g in wrong.gestures:
                        self._reward(g, -1)
                    self._save_prefs()
                    library.add_alias(_clean(wrong.text), result.gestures[0].motion)
                    intent.reindex()
                    return self._event(
                        text, f"Got it: \"{wrong.text}\" means {result.gestures[0].motion.replace('_', ' ')}.",
                        result.gestures, "correction")

            name, steps = self._match_skill(cleaned)
            if name:
                gestures = self._interpret_steps(steps)
                return self._event(text, f"{name}: " + "  >  ".join(g.label for g in gestures),
                                   gestures, "skill")

            result = intent.interpret(text, self.choose)
            if result.abort:
                return self._event(text, result.reply, kind="stop")
            missing = result.unknown[0] if result.unknown else ""
            if missing and len(missing.split()) >= 2 and can_compose:
                lead = result.reply.splitlines()[0] + "\n" if result.gestures else ""
                return self._event(
                    text, f"{lead}I don't know \"{missing}\" yet. Give me a few seconds to invent it.",
                    result.gestures, pending=missing)
            if missing in result.guesses:
                gestures = result.gestures + [result.guesses[missing]]
                return self._event(text, "  >  ".join(g.label for g in gestures), gestures)
            return self._event(text, result.reply, result.gestures,
                               "command" if result.gestures else "info")

    def composed(self, phrase, name, created):
        """The composer finished: build the new gesture and record the lesson."""
        with self._lock:
            g = intent._make(name, intent._modifiers([]), self.choose)
            shown = name.replace("_", " ")
            reply = (f"Learned a new gesture: {shown}" if created
                     else f"That sounds like my {shown}. I'll remember that.")
            return self._event(phrase, reply, [g], "learn")

    # --- reporting ---------------------------------------------------------------

    def summary(self):
        """Everything the Learning page shows."""
        with self._lock:
            log = self._read_log()
            learned = library._read_learned()
            today = time.strftime("%Y-%m-%d")
            days = Counter()
            lessons = Counter()
            motions = Counter()
            ups = downs = 0
            for row in log:
                day = time.strftime("%Y-%m-%d", time.localtime(row["t"]))
                if row["kind"] in ("command", "skill", "correction", "learn"):
                    days[day] += 1
                    for g in row.get("gestures", []):
                        motions[g[0]] += 1
                if row["kind"] in ("learn", "correction", "skill", "feedback"):
                    lessons[day] += 1
                if row["kind"] == "feedback":
                    ups += row["value"] > 0
                    downs += row["value"] < 0
            last14 = [time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400 * i))
                      for i in range(13, -1, -1)]

            def taste(options, index):
                out = {}
                for option in options:
                    up = down = 0.0
                    for key, (u, d) in self._prefs["overall"].items():
                        if key.split("/")[index] == option:
                            up, down = up + u, down + d
                    out[option] = {"up": up, "down": down}
                return out

            return {
                "today": today,
                "counts": {
                    "motions": sum(1 for m in library.MOTIONS.values() if not m.learned),
                    "learned": sum(1 for m in library.MOTIONS.values() if m.learned),
                    "skills": len(learned.get("skills", {})),
                    "phrases": len(learned.get("aliases", {})),
                    "interactions": sum(days.values()),
                    "up": ups, "down": downs,
                },
                "days": [{"day": d, "used": days[d], "lessons": lessons[d]} for d in last14],
                "top": motions.most_common(8),
                "taste": {"speed": taste(SPEEDS, 0), "size": taste(SIZES, 1)},
                "skills": [{"name": n, **s} for n, s in sorted(learned.get("skills", {}).items())],
                "learned": [{"name": n, "about": m.get("about", "")}
                            for n, m in sorted(learned.get("motions", {}).items())],
                "lessons": [
                    {"t": row["t"], "kind": row["kind"],
                     "text": row.get("reply") or row.get("name") or row.get("text", "")}
                    for row in log if row["kind"] in ("learn", "correction", "skill")
                ][-12:][::-1],
            }
