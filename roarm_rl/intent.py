"""Turn a line of chat text into gestures from roarm_rl.library.

Offline keyword matching, no model involved: each motion lists the words and
phrases that should trigger it, the longest matching phrase wins, and words
like "slowly", "big", "right" or "three times" pick the variant. Sentences
are split on "then" / "and" / commas, so "nod twice then wave slowly" queues
two gestures.

    interpret("thanks, that was great!")  ->  bow, then celebrate
"""

import difflib
import re
from dataclasses import dataclass, field

from roarm_rl import library
from roarm_rl.library import MOTIONS, build, variant_name

KEYWORDS = {
    "nod": ["nod", "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "agree", "agreed",
            "correct", "exactly", "of course", "indeed", "affirmative"],
    "shake": ["shake", "shake your head", "no", "nope", "nah", "disagree", "wrong", "never",
              "negative", "no way jose", "refuse"],
    "wave": ["wave"],
    "hello": ["hello", "hi", "hey", "hiya", "howdy", "greetings", "greet", "good morning",
              "good afternoon", "good evening", "whats up", "sup", "nice to meet you", "welcome"],
    "goodbye": ["goodbye", "bye", "farewell", "see you", "see ya", "later", "wave goodbye",
                "take care", "cya", "so long"],
    "bow": ["bow", "thanks", "thank you", "thx", "thank", "grateful", "respect", "sorry",
            "apologize", "apologies", "forgive me", "my bad", "you are welcome", "curtsy"],
    "shrug": ["shrug", "dont know", "do not know", "idk", "whatever", "no idea", "maybe",
              "who knows", "not sure", "dunno", "meh", "perhaps"],
    "point": ["point", "pointing", "over there", "that one", "this one", "show me", "indicate"],
    "look": ["look", "watch", "see", "glance", "gaze", "turn", "face"],
    "look_around": ["look around", "scan", "search", "explore", "survey", "patrol", "find",
                    "where is it", "check around", "look left and right"],
    "clap": ["clap", "applaud", "applause", "bravo", "snap", "clack", "click"],
    "grab": ["grab", "pick", "pick up", "take", "grasp", "catch", "seize", "hold", "pinch", "get it"],
    "give": ["give", "offer", "hand over", "here you go", "release", "drop", "pass", "share",
             "present", "deliver"],
    "pull": ["pull", "tug", "drag", "retract", "yank"],
    "push": ["push", "shove", "poke", "jab", "punch", "nudge", "press"],
    "stretch": ["stretch", "warm up", "limber", "exercise", "workout"],
    "sleep": ["sleep", "nap", "good night", "goodnight", "night night", "bedtime", "shut down",
              "power down", "lie down", "rest now", "doze", "snooze"],
    "wake": ["wake", "wake up", "get up", "rise and shine", "arise", "boot up"],
    "yawn": ["yawn", "tired", "sleepy", "exhausted", "drowsy"],
    "celebrate": ["celebrate", "cheer", "hooray", "hurray", "yay", "woohoo", "party", "victory",
                  "we won", "well done", "good job", "great job", "congrats", "congratulations",
                  "awesome", "amazing", "great", "fantastic", "excellent", "brilliant", "win",
                  "success", "it works"],
    "happy": ["happy", "glad", "joy", "joyful", "cheerful", "smile", "delighted", "pleased",
              "excited", "thrilled", "good mood", "fun"],
    "laugh": ["laugh", "haha", "hahaha", "lol", "lmao", "rofl", "funny", "hilarious", "joke",
              "giggle", "chuckle", "hehe"],
    "sad": ["sad", "unhappy", "depressed", "disappointed", "bad news", "gloomy", "miserable",
            "failed", "failure", "lost", "lonely", "upset", "oh no", "hurt"],
    "cry": ["cry", "sob", "weep", "tears", "crying", "heartbroken"],
    "angry": ["angry", "mad", "furious", "annoyed", "rage", "irritated", "grumpy", "hate",
              "frustrated", "attack", "fight"],
    "scared": ["scared", "afraid", "fear", "frightened", "terrified", "boo", "spider", "monster",
               "ghost", "scary", "tremble", "shiver", "panic", "nervous", "cold"],
    "surprise": ["surprise", "surprised", "wow", "whoa", "omg", "oh my god", "no way", "shock",
                 "shocked", "astonished", "unbelievable", "really", "gasp", "jump"],
    "curious": ["curious", "interesting", "whats that", "what is that", "what", "huh", "inspect",
                "examine", "investigate", "sniff", "closer look", "tell me more"],
    "think": ["think", "thinking", "hmm", "hmmm", "wonder", "ponder", "consider", "let me think",
              "why", "how", "calculate", "compute", "contemplate"],
    "confused": ["confused", "puzzled", "dont understand", "do not understand", "confusing",
                 "baffled", "what do you mean", "error"],
    "proud": ["proud", "pride", "confident", "stand tall", "tall", "brave", "hero"],
    "shy": ["shy", "embarrassed", "bashful", "blush", "hide", "timid", "awkward"],
    "bored": ["bored", "boring", "dull", "waiting", "sigh", "idle"],
    "approve": ["approve", "good", "nice", "like", "thumbs up", "perfect", "cool", "fine",
                "well", "alright", "deal", "sounds good", "love it"],
    "heartbeat": ["love", "heart", "heartbeat", "i love you", "adore", "pulse", "kiss", "cute"],
    "hug": ["hug", "embrace", "cuddle", "squeeze", "comfort"],
    "high_five": ["high five", "highfive", "hi five", "gimme five", "give me five", "slap"],
    "fist_bump": ["fist bump", "fistbump", "bump", "pound it", "knuckles"],
    "handshake": ["handshake", "shake hands", "shake my hand", "shake hand", "pleased to meet you"],
    "salute": ["salute", "yes sir", "aye aye", "attention", "at your service", "captain"],
    "beckon": ["beckon", "come", "come here", "come closer", "over here", "follow me", "this way",
               "approach", "summon"],
    "halt": ["halt", "wait", "hold on", "stop sign", "stay", "dont move", "stand back",
             "hold it", "not so fast", "pause", "talk to the hand"],
    "circle": ["circle", "round", "loop", "orbit", "rotate", "spin around", "draw a circle",
               "ring", "wheel"],
    "figure_eight": ["figure eight", "figure 8", "eight", "infinity", "draw an eight"],
    "stir": ["stir", "mix", "whisk", "cook", "blend", "soup", "coffee"],
    "tap": ["tap", "knock", "peck", "drum", "hammer", "type", "typing", "knock knock", "beat"],
    "sweep": ["sweep", "wipe", "clean", "brush", "dust", "clear the table", "mop"],
    "swipe": ["swipe", "flick", "next", "previous", "dismiss", "swat", "scroll"],
    "peek": ["peek", "peekaboo", "sneak", "spy", "peep"],
    "bounce": ["bounce", "bob", "hop", "bop", "pump", "boing"],
    "sway": ["sway", "rock", "swing", "groove", "drift", "relaxed", "chill", "calm", "music"],
    "zigzag": ["zigzag", "zig zag", "lightning", "slalom", "snake"],
    "wiggle": ["wiggle", "jiggle", "shimmy", "vibrate", "buzz", "shudder", "tickle"],
    "dance": ["dance", "boogie", "disco", "party time", "show me your moves", "moves", "jam"],
    "twist": ["twist", "spin", "swivel", "whirl", "turn around"],
    "flex": ["flex", "strong", "muscle", "muscles", "power", "show off", "mighty"],
    "lean_forward": ["lean forward", "lean in", "forward", "reach", "reach out", "closer", "extend"],
    "lean_back": ["lean back", "back", "back off", "back up", "away", "recoil", "retreat"],
    "open": ["open", "open gripper", "open hand", "open your hand", "let go", "unclench"],
    "close": ["close", "close gripper", "close hand", "shut", "clench", "grip", "fist"],
    "home": ["home", "go home", "reset", "rest", "relax", "neutral", "return", "center",
             "centre", "default", "at ease", "stand by", "standby"],
}

# Motions picked by direction rather than by their own keyword.
DIRECTIONAL = {
    "point": {"up": "point_up", "down": "point_down", "forward": "point_forward", None: "point_forward"},
    "look": {"up": "look_up", "down": "look_down", "forward": "look_around", None: "look_around"},
}

SPEED_WORDS = {
    "slow": ["slow", "slowly", "gently", "gentle", "calmly", "lazily", "carefully", "softly", "soft"],
    "fast": ["fast", "quick", "quickly", "rapidly", "rapid", "hurry", "energetic", "energetically",
             "frantically", "wildly", "speedy", "faster"],
}
SIZE_WORDS = {
    "small": ["small", "little", "tiny", "subtle", "slightly", "slight", "bit", "mini"],
    "big": ["big", "large", "huge", "wide", "exaggerated", "giant", "massive", "dramatic",
            "dramatically", "lot", "enthusiastically", "strong", "strongly", "stronger",
            "powerful", "deep", "deeply", "full", "hard", "bigger"],
}
SIDE_WORDS = {"left": "left", "right": "right"}
VERTICAL_WORDS = {"up": "up", "upward": "up", "upwards": "up", "above": "up", "sky": "up",
                  "ceiling": "up", "down": "down", "downward": "down", "downwards": "down",
                  "below": "down", "floor": "down", "ground": "down", "forward": "forward",
                  "ahead": "forward", "front": "forward", "straight": "forward"}
NUMBER_WORDS = {"once": 1, "one": 1, "twice": 2, "two": 2, "thrice": 3, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "couple": 2, "few": 3}

ABORT = {"stop", "stop it", "stop moving", "cancel", "abort", "freeze", "enough", "quit it",
         "stop that", "thats enough"}
HELP = {"help", "list", "gestures", "commands", "what can you do", "options", "menu"}

_IDIOMS = ["up and down", "back and forth", "side to side", "to and fro"]
_SPLIT = re.compile(r"\b(?:and then|then|and|after that|followed by)\b|[,;.!]+")
_FILLER = {"please", "pls", "robot", "roarm", "arm", "the", "a", "an", "kindly", "just", "now"}

_MODIFIER_VOCAB = (
    {w for ws in SPEED_WORDS.values() for w in ws}
    | {w for ws in SIZE_WORDS.values() for w in ws}
    | set(SIDE_WORDS) | set(VERTICAL_WORDS) | set(NUMBER_WORDS) | {"times", "time", "again"}
)
# Words that say nothing about which motion is meant; ignored when judging
# how much of a sentence a match explains.
_STOP = {"i", "im", "you", "me", "my", "your", "we", "it", "its", "is", "are", "was", "be", "being",
         "do", "can", "could", "would", "will", "to", "for", "of", "in", "on", "at", "with", "that",
         "this", "so", "some", "want", "lets", "let", "us", "thats", "today", "then", "like", "as",
         "if", "there", "here", "have", "has", "am", "really", "very", "too", "again", "try"}
_PHRASES = []  # [(token tuple, motion name)], longest first
_LEARNED_PHRASES = []  # multi-word phrases of learned motions, matched on the whole line
_VOCAB = set()


def _words(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower().replace("'", "")) if t not in _FILLER]


def reindex():
    """Rebuild the phrase tables; call after roarm_rl.library learns a motion."""
    global _PHRASES, _LEARNED_PHRASES, _VOCAB
    pairs = [(tuple(kw.split()), name) for name, kws in KEYWORDS.items() for kw in kws]
    learned = [(tuple(_words(kw)), name) for name, kws in library.LEARNED_KEYWORDS.items()
               for kw in kws if name in MOTIONS]
    learned = [(phrase, name) for phrase, name in learned if phrase]
    _PHRASES = sorted(pairs + learned, key=lambda item: -len(item[0]))
    _LEARNED_PHRASES = sorted((item for item in learned if len(item[0]) > 1),
                              key=lambda item: -len(item[0]))
    _VOCAB = {tok for phrase, _ in _PHRASES for tok in phrase} | _MODIFIER_VOCAB


reindex()


@dataclass
class Gesture:
    label: str  # e.g. "nod (fast, x3)"
    name: str  # library variant name
    keyframes: list


@dataclass
class Result:
    reply: str
    gestures: list = field(default_factory=list)
    abort: bool = False
    unknown: list = field(default_factory=list)  # clauses nothing matched well
    guesses: dict = field(default_factory=dict)  # unknown clause -> weak-match Gesture, if any


def _normalize(token):
    """Map inflections and near-misses onto a known word: waving -> wave."""
    if token in _VOCAB or token.isdigit():
        return token
    stems = []
    for suffix in ("ing", "ed", "es", "s", "ly", "er"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            stem = token[: -len(suffix)]
            stems += [stem, stem + "e"]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stems.append(stem[:-1])
    for stem in stems:
        if stem in _VOCAB:
            return stem
    if len(token) >= 5:
        close = difflib.get_close_matches(token, _VOCAB, n=1, cutoff=0.82)
        if close:
            return close[0]
    return token


def _tokens(text):
    text = text.lower().replace("'", "").replace("’", "")
    return [_normalize(t) for t in re.findall(r"[a-z0-9]+", text) if t not in _FILLER]


def _match_motion(tokens):
    """(best motion or None, weak) by longest keyword phrase.

    weak is True when a single stray word matched inside a longer sentence
    whose other words went unexplained, e.g. "pretend you are a snake being
    charmed" only hitting "snake". Such a sentence deserves a new gesture.
    """
    scores = {}
    covered = set()
    for phrase, name in _PHRASES:
        n = len(phrase)
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i + n]) == phrase:
                covered.update(range(i, i + n))
                # squared so one long phrase beats several stray single words;
                # a lone modifier word ("forward", "five") barely counts
                weight = 0.25 if n == 1 and phrase[0] in _MODIFIER_VOCAB else n * n
                scores[name] = scores.get(name, 0) + weight
                if phrase == tuple(name.split("_")):
                    scores[name] += 0.5  # naming the motion outright breaks ties
                break
    if not scores:
        return None, False
    best = max(scores, key=lambda name: scores[name])
    unexplained = [t for i, t in enumerate(tokens) if i not in covered and t not in _STOP
                   and t not in _MODIFIER_VOCAB and not t.isdigit()]
    return best, scores[best] < 2 and len(unexplained) >= 2


def _modifiers(tokens):
    mods = {"speed": "normal", "size": "normal", "side": None, "vertical": None, "reps": None}
    for i, tok in enumerate(tokens):
        for speed, words in SPEED_WORDS.items():
            if tok in words:
                mods["speed"] = speed
        for size, words in SIZE_WORDS.items():
            if tok in words:
                mods["size"] = size
        if tok in SIDE_WORDS:
            mods["side"] = SIDE_WORDS[tok]
        if tok in VERTICAL_WORDS:
            mods["vertical"] = VERTICAL_WORDS[tok]
        if tok in ("once", "twice", "thrice"):
            mods["reps"] = NUMBER_WORDS[tok]
        if tok in ("times", "time") and i > 0:
            prev = tokens[i - 1]
            if prev.isdigit():
                mods["reps"] = int(prev)
            elif prev in NUMBER_WORDS:
                mods["reps"] = NUMBER_WORDS[prev]
    return mods


def _resolve(name, mods):
    """Apply direction words: 'point' + 'up' -> point_up."""
    if name in DIRECTIONAL and mods["side"] is None:
        return DIRECTIONAL[name].get(mods["vertical"], DIRECTIONAL[name][None])
    return name


def _label(name, mods, side):
    notes = [m for m in (mods["speed"], mods["size"]) if m != "normal"]
    if MOTIONS[name].sided:
        notes.insert(0, side)
    if mods["reps"]:
        notes.append(f"x{mods['reps']}")
    text = name.replace("_", " ")
    return f"{text} ({', '.join(notes)})" if notes else text


def help_text():
    names = ", ".join(sorted(n.replace("_", " ") for n in MOTIONS))
    return (
        f"I know {len(MOTIONS)} motions: {names}.\n"
        "Add slowly / fast, small / big, left / right, or a count like 'three times'. "
        "Chain with 'then'. Say 'stop' to cancel."
    )


def interpret(text):
    """Chat text -> Result(reply, gestures to play in order, abort flag)."""
    cleaned = " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("'", "")))
    if not cleaned:
        return Result("Tell me what to do, or type 'help'.")
    if cleaned in ABORT:
        return Result("Stopped.", abort=True)
    if cleaned in HELP:
        return Result(help_text())

    # A learned phrase may contain "and" or commas, so try the whole line first.
    words = _words(text)
    for phrase, name in _LEARNED_PHRASES:
        n = len(phrase)
        if any(tuple(words[i:i + n]) == phrase for i in range(len(words) - n + 1)):
            mods = _modifiers(_tokens(text))
            g = Gesture(_label(name, mods, "left"), variant_name(name, mods["speed"], mods["size"]),
                        build(name, mods["speed"], mods["size"], reps=mods["reps"]))
            return Result(g.label, [g])

    lowered = text.lower()
    for idiom in _IDIOMS:
        lowered = lowered.replace(idiom, " ")

    picked, unknown, weak_picks = [], [], {}  # picked: [motion name, modifiers]
    for clause in _SPLIT.split(lowered):
        tokens = _tokens(clause)
        if not tokens:
            continue
        mods = _modifiers(tokens)
        name, weak = _match_motion(tokens)
        only_modifiers = all(t in _MODIFIER_VOCAB or t.isdigit() for t in tokens)
        if picked and only_modifiers:
            if mods["side"] or mods["vertical"]:
                # "look left and right": same motion again, other direction
                picked.append([picked[-1][0], mods])
            else:
                # "wave, slowly": the modifier belongs to the previous motion
                defaults = _modifiers([])
                picked[-1][1].update({k: v for k, v in mods.items() if v != defaults[k]})
            continue
        if name is None:
            if "?" in text and not picked:
                name = "think"
            else:
                unknown.append(clause.strip())
                continue
        if weak:
            unknown.append(clause.strip())
            weak_picks[clause.strip()] = [name, mods]
            continue
        picked.append([name, mods])

    def make(name, mods):
        name = _resolve(name, mods)
        side = mods["side"] or "left"
        return Gesture(
            label=_label(name, mods, side),
            name=variant_name(name, mods["speed"], mods["size"], side),
            keyframes=build(name, mods["speed"], mods["size"], side, mods["reps"]),
        )

    gestures = [make(name, mods) for name, mods in picked]
    guesses = {clause: make(name, mods) for clause, (name, mods) in weak_picks.items()}

    if not gestures:
        return Result("I don't know a gesture for that yet. Type 'help' to see what I can do.",
                      unknown=unknown, guesses=guesses)
    reply = "  >  ".join(g.label for g in gestures)
    if unknown:
        reply += f"\n(skipped: {'; '.join(unknown)})"
    return Result(reply, gestures, unknown=unknown, guesses=guesses)
