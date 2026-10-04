"""Web app for the arm: 3D view, chat, voice, gesture library and learning page.

    python -m roarm_rl.server                 # http://127.0.0.1:8000 on this machine
    python -m roarm_rl.server --lan           # also reachable from a phone on the same Wi-Fi
    python -m roarm_rl.server --lan --https   # needed for the phone's microphone
    python -m roarm_rl.server --hw serial     # with the real arm on USB (port is found automatically)

The simulator runs headless on one thread and is the only code that touches
PyBullet. Browsers get the arm's link poses over a WebSocket about 30 times a
second and draw it themselves, so any number of devices can watch and drive
the same arm.

With --lan the server prints a link containing an access code; devices
without the code are refused. Requests from this machine never need it.
"""

import argparse
import asyncio
import io
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import pybullet as p
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from roarm_rl import library
from roarm_rl.brain import Brain
from roarm_rl.composer import Composer, available as composer_available
from roarm_rl.gesture import SAFE_BOUNDS, GesturePlayer
from roarm_rl.hardware import RoArmHardware, RoArmHardwareError
from roarm_rl.sim import RoArmSim

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
MESH_DIR = os.path.join(_ROOT, "assets", "meshes")
URDF_PATH = os.path.join(_ROOT, "assets", "urdf", "roarm_m2.urdf.template")
CERT_DIR = os.path.join(_ROOT, "certs")

TICK_HZ = 60
MANUAL_SPEED = 1.5  # rad/s when following the sliders
REACH_SPEED = 3.0  # rad/s when the hand is dragged in the 3D view
MIN_REACH_HEIGHT = 0.04  # m; a dragged target is kept this far above the table
MIRROR_MIN_INTERVAL = 0.04  # seconds; avoid flooding the servo bus
MIRROR_EPSILON = 0.004  # radians; skip resend if target barely changed
# Measured on a RoArm-M2 over USB: the arm starts moving about 0.2 s after a
# command and tops out near 1.9 rad/s. Gestures are sent that far ahead of the
# picture and stretched to fit that speed, so the real arm stays in step.
ARM_LATENCY = 0.18
ARM_MAX_SPEED = 1.6
ARM_FAST = (1500, 60)  # servo speed, acceleration while following
ARM_GENTLE = (300, 10)  # while catching up after mirroring is switched on
CATCH_UP_SPEED = 0.4  # rad/s the arm manages at ARM_GENTLE
COOKIE = "roarm_access"


def robot_description():
    """Link names and mesh files in kinematic order, for the browser to load."""
    root = ET.parse(URDF_PATH).getroot()
    links = []
    for link in root.findall("link"):
        mesh = link.find("visual/geometry/mesh")
        if mesh is not None:
            links.append({"name": link.get("name"),
                          "mesh": os.path.basename(mesh.get("filename"))})
    return links


class Robot:
    """Owns the simulator thread. Everything else talks to it through these methods."""

    def __init__(self, hardware=None):
        self.hardware = hardware
        self.mirroring = False
        self.torque = True  # False: motors released, the view follows the real arm
        self._torque_request = None
        self._catching_up = False  # mirroring was just switched on
        self.player = GesturePlayer()
        self.brain = Brain()
        self.composer = Composer() if composer_available() else None
        self.messages = []  # chat history shared by every connected browser
        self.state = {}
        self._lock = threading.RLock()
        self._goal = None
        self._goal_speed = MANUAL_SPEED
        self._reach = None
        self._home = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if self.hardware is not None and self.hardware.connected:
            self.hardware.disconnect()

    # --- commands (any thread) -----------------------------------------------------

    def post(self, who, text, event=None, source=""):
        with self._lock:
            self.messages.append({
                "n": len(self.messages) + 1, "who": who, "text": text, "source": source,
                "event": event.id if event is not None and event.gestures else None,
            })
            del self.messages[:-300]

    def say(self, text, source=""):
        """Handle one line of chat or transcribed speech."""
        with self._lock:
            self.post("you", text, source=source)
            busy = self.composer is not None and self.composer.busy
            event = self.brain.handle(text, can_compose=self.composer is not None and not busy)
            if event.kind == "stop":
                self.stop()
            for g in event.gestures:
                self.player.enqueue(g.label, g.keyframes)
            if event.pending and not self.composer.request(event.pending):
                event.reply = "I'm still working on the last new gesture."
            self.post("arm", event.reply, event)
            return event

    def play(self, name, speed="normal", size="normal", side="left", reps=None):
        with self._lock:
            keyframes = library.build(name, speed, size, side, reps)
            label = library.variant_name(name, speed, size, side).replace("_", " ")
            self.player.enqueue(label, keyframes)
            self.post("arm", label)

    def stop(self):
        with self._lock:
            if self.player.label:
                self.brain.stopped(self.player.label)
            self.player.cancel()
            self._goal = None

    def home(self):
        with self._lock:
            self.player.cancel()
            self._home = True

    def set_joints(self, q):
        with self._lock:
            self.player.cancel()
            self._goal, self._goal_speed = [float(v) for v in q[:4]], MANUAL_SPEED

    def reach(self, xyz):
        """Move the hand toward a point (metres, simulator frame); solved on the simulator thread."""
        with self._lock:
            self.player.cancel()
            self._reach = [float(xyz[0]), float(xyz[1]), max(MIN_REACH_HEIGHT, float(xyz[2]))]

    def set_mirror(self, on):
        on = bool(on) and self.torque and self.hardware is not None and self.hardware.connected
        self._catching_up = on and not self.mirroring
        self.mirroring = on
        self.player.max_speed = ARM_MAX_SPEED if on else None
        return self.mirroring

    def set_torque(self, on):
        """Hold (True) or release (False) the real arm's motors; applied on the simulator thread."""
        if self.hardware is None or not self.hardware.connected:
            return False
        self._torque_request = bool(on)
        return True

    # --- simulator thread ----------------------------------------------------------

    def _run(self):
        sim = RoArmSim(gui=False)
        links = [j for j in range(p.getNumJoints(sim.robot, physicsClientId=sim.client_id))]
        q = list(sim.home_radians)
        last_sent, last_sent_at, gentle_until = None, 0.0, 0.0
        dt = 1.0 / TICK_HZ
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                with self._lock:
                    if self.composer is not None:
                        for phrase, name, created, error in self.composer.poll():
                            if error:
                                self.post("arm", f"I couldn't invent a gesture for \"{phrase}\": {error}")
                                continue
                            event = self.brain.composed(phrase, name, created)
                            for g in event.gestures:
                                self.player.enqueue(g.label, g.keyframes)
                            self.post("arm", event.reply, event)
                    if self._torque_request is not None:
                        want, self._torque_request = self._torque_request, None
                        try:
                            self.hardware.set_torque(want)
                            self.torque = want
                            if not want:  # never command a limp arm
                                self.mirroring = False
                        except Exception as e:
                            print(f"[hardware] torque change failed: {e}")
                    if not self.torque:
                        # Motors released: someone is moving the arm by hand. Show where it is.
                        self.player.cancel()
                        self._goal, self._home = None, False
                        try:
                            measured = self.hardware.get_joint_positions()
                        except Exception as e:
                            measured = None
                            print(f"[hardware] read failed: {e}")
                        if measured is not None:
                            q = measured
                    if self._home:
                        self._home, self._goal = False, list(sim.home_radians)
                        self._goal_speed = MANUAL_SPEED
                    if self._reach is not None and self.torque:
                        target, self._reach = self._reach, None
                        solved = sim.solve_ik(target)[:3] + [q[3]]  # the gripper keeps its opening
                        self._goal = [min(max(v, lo), hi) for v, (lo, hi) in zip(solved, SAFE_BOUNDS)]
                        self._goal_speed = min(REACH_SPEED, ARM_MAX_SPEED) if self.mirroring else REACH_SPEED
                    pose = self.player.update(q)
                    if pose is not None:
                        q, self._goal = pose, None
                    elif self._goal is not None:
                        step = self._goal_speed * dt
                        q = [a + max(-step, min(step, b - a)) for a, b in zip(q, self._goal)]
                    label = self.player.label
                    inventing = self.composer is not None and self.composer.busy

                q = sim.set_joint_targets(q, instant=True)
                poses = []
                for j in links:
                    s = p.getLinkState(sim.robot, j, physicsClientId=sim.client_id)
                    poses.append([round(v, 5) for v in (*s[4], *s[5])])
                self.state = {
                    "q": [round(v, 4) for v in q], "links": poses, "gesture": label,
                    "inventing": inventing, "mirror": self.mirroring, "torque": self.torque,
                    "hardware": self.hardware is not None and self.hardware.connected,
                }

                if self.mirroring:
                    now = time.monotonic()
                    if self._catching_up:
                        # The arm may be far from the picture: close the gap slowly first.
                        self._catching_up, last_sent = False, None
                        try:
                            measured = self.hardware.get_joint_positions()
                        except Exception:
                            measured = None
                        gap = (max(abs(a - b) for a, b in zip(q, measured))
                               if measured else 1.5)
                        gentle_until = now + gap / CATCH_UP_SPEED + 0.3
                    target = self.player.peek(ARM_LATENCY) or q
                    moved = last_sent is None or max(
                        abs(a - b) for a, b in zip(target, last_sent)) > MIRROR_EPSILON
                    if moved and now - last_sent_at > MIRROR_MIN_INTERVAL:
                        speed, acc = ARM_GENTLE if now < gentle_until else ARM_FAST
                        try:
                            self.hardware.set_joint_targets(target, speed, acc)
                            last_sent = list(target)
                        except Exception as e:
                            print(f"[hardware] mirror failed: {e}")
                        last_sent_at = now

                next_tick += dt
                time.sleep(max(0.0, next_tick - time.monotonic()))
        finally:
            sim.close()


class Ears:
    """Speech-to-text for audio recorded in the browser. Loads the model in the background."""

    def __init__(self):
        self.ready = False
        self.error = None
        self._transcriber = None
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from roarm_rl.voice import Transcriber

            self._transcriber = Transcriber()
            self.ready = True
        except Exception as e:
            self.error = str(e)

    def transcribe(self, data):
        return self._transcriber.transcribe(io.BytesIO(data))


def create_app(robot, ears=None, access_code=None):
    app = FastAPI(title="RoArm", docs_url=None, redoc_url=None)

    def allowed(client_host, cookies, query):
        if access_code is None or client_host in ("127.0.0.1", "::1"):
            return True
        return secrets.compare_digest(cookies.get(COOKIE, ""), access_code) or \
            secrets.compare_digest(query.get("code", ""), access_code)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        if not allowed(request.client.host, request.cookies, request.query_params):
            return JSONResponse({"error": "access code required: open the link the server printed"},
                                status_code=403)
        if "code" in request.query_params:  # keep the code out of the address bar
            response = RedirectResponse(request.url.path)
            response.set_cookie(COOKIE, access_code, max_age=30 * 86400, httponly=True,
                                samesite="strict")
            return response
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/api/info")
    def info():
        motions = [{"name": name, "about": m.about, "sided": m.sided,
                    "repeats": m.reps is not None, "learned": m.learned}
                   for name, m in library.MOTIONS.items()]
        return {
            "links": robot_description(), "motions": motions,
            "speeds": list(library.SPEEDS), "sizes": list(library.SIZES),
            "bounds": SAFE_BOUNDS, "designer": robot.composer is not None,
            "voice": ears is not None and ears.error is None,
        }

    @app.post("/api/say")
    async def say(body: dict):
        text = str(body.get("text", "")).strip()[:300]
        if not text:
            return JSONResponse({"error": "empty"}, status_code=400)
        event = await asyncio.to_thread(robot.say, text)
        return {"reply": event.reply, "event": event.id}

    @app.post("/api/audio")
    async def audio(request: Request):
        if ears is None or not ears.ready:
            return JSONResponse({"error": ears.error if ears and ears.error
                                 else "the speech model is still loading"}, status_code=503)
        data = await request.body()
        if not 200 < len(data) < 5_000_000:
            return JSONResponse({"error": "no audio"}, status_code=400)
        try:
            text = await asyncio.to_thread(ears.transcribe, data)
        except Exception as e:
            print(f"[voice] {e}")
            return JSONResponse({"error": "could not read that recording"}, status_code=400)
        if not text:
            return {"text": ""}
        await asyncio.to_thread(robot.say, text, "voice")
        return {"text": text}

    @app.post("/api/play")
    def play(body: dict):
        name = body.get("name")
        if name not in library.MOTIONS:
            return JSONResponse({"error": "unknown motion"}, status_code=404)
        speed = body.get("speed") if body.get("speed") in library.SPEEDS else "normal"
        size = body.get("size") if body.get("size") in library.SIZES else "normal"
        side = "right" if body.get("side") == "right" else "left"
        robot.play(name, speed, size, side)
        return {"ok": True}

    @app.post("/api/joints")
    def joints(body: dict):
        q = body.get("q")
        if not isinstance(q, list) or len(q) != 4:
            return JSONResponse({"error": "q must be four numbers"}, status_code=400)
        # same box the gestures stay in, so the sliders cannot reach the table either
        robot.set_joints([min(max(float(v), lo), hi) for v, (lo, hi) in zip(q, SAFE_BOUNDS)])
        return {"ok": True}

    @app.post("/api/reach")
    def reach(body: dict):
        xyz = body.get("xyz")
        if not isinstance(xyz, list) or len(xyz) != 3:
            return JSONResponse({"error": "xyz must be three numbers"}, status_code=400)
        robot.reach(xyz)
        return {"ok": True}

    @app.post("/api/stop")
    def stop():
        robot.stop()
        return {"ok": True}

    @app.post("/api/home")
    def home():
        robot.home()
        return {"ok": True}

    @app.post("/api/mirror")
    def mirror(body: dict):
        return {"mirror": robot.set_mirror(body.get("on"))}

    @app.post("/api/torque")
    def torque(body: dict):
        return {"ok": robot.set_torque(body.get("on"))}

    @app.post("/api/feedback")
    def feedback(body: dict):
        reply = robot.brain.feedback(int(body.get("event", 0)), int(body.get("value", 0)))
        return {"reply": reply}

    @app.get("/api/brain")
    def brain():
        return robot.brain.summary()

    @app.post("/api/skill")
    def add_skill(body: dict):
        name, steps = str(body.get("name", "")), str(body.get("steps", ""))
        event = robot.say(f"learn {name}: {steps}")
        return {"reply": event.reply}

    @app.delete("/api/skill/{name}")
    def delete_skill(name: str):
        return {"ok": robot.brain.forget_skill(name)}

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        if not allowed(socket.client.host, socket.cookies, socket.query_params):
            await socket.close(code=4403)
            return
        await socket.accept()
        seen = 0
        try:
            while True:
                packet = dict(robot.state)
                fresh = [m for m in robot.messages if m["n"] > seen]
                if fresh:
                    packet["messages"] = fresh
                    seen = fresh[-1]["n"]
                packet["ears"] = bool(ears and ears.ready)
                await socket.send_json(packet)
                await asyncio.sleep(0.025)
        except (WebSocketDisconnect, RuntimeError):
            pass

    app.mount("/meshes", StaticFiles(directory=MESH_DIR), name="meshes")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def find_arm_port():
    """Serial port of the arm's USB adapter (CP210x or CH34x), or None."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    ports = list(list_ports.comports())
    for port in ports:
        if (port.vid, port.pid) in ((0x10C4, 0xEA60), (0x1A86, 0x7523), (0x1A86, 0x55D4)):
            return port.device
    return ports[0].device if len(ports) == 1 else None


def lan_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def self_signed_cert(address):
    """Create certs/roarm.pem + certs/roarm.key with openssl if they are missing."""
    cert, key = os.path.join(CERT_DIR, "roarm.pem"), os.path.join(CERT_DIR, "roarm.key")
    if not (os.path.exists(cert) and os.path.exists(key)):
        os.makedirs(CERT_DIR, exist_ok=True)
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
             "-keyout", key, "-out", cert, "-subj", "/CN=roarm",
             "-addext", f"subjectAltName=IP:{address},IP:127.0.0.1,DNS:localhost"],
            check=True, capture_output=True)
    return cert, key


def main():
    ap = argparse.ArgumentParser(description="RoArm-M2 web app")
    ap.add_argument("--port-http", type=int, default=8000, help="port to serve on (default 8000)")
    ap.add_argument("--lan", action="store_true",
                    help="accept other devices on the network (they need the printed access code)")
    ap.add_argument("--https", action="store_true",
                    help="serve over HTTPS with a self-signed certificate; phones need this for the mic")
    ap.add_argument("--no-voice", action="store_true", help="do not load the speech model")
    ap.add_argument("--hw", choices=["none", "serial", "http"], default="none",
                    help="connect to a real RoArm-M2 over serial or WiFi/HTTP (default: none)")
    ap.add_argument("--port", default=None,
                    help="serial port, e.g. COM5 or /dev/ttyUSB0 (default: find the arm's USB adapter)")
    ap.add_argument("--baudrate", type=int, default=115200)
    ap.add_argument("--host", default="192.168.4.1", help="RoArm IP address for --hw http")
    args = ap.parse_args()

    hardware = None
    if args.hw != "none":
        hardware = RoArmHardware()
        try:
            if args.hw == "serial":
                port = args.port or find_arm_port()
                if port is None:
                    raise RoArmHardwareError("no USB serial adapter found; plug the arm in or pass --port")
                hardware.connect_serial(port, args.baudrate)
                print(f"[hardware] connected on {port}; turn on Mirror in the Control tab to move it")
            else:
                hardware.connect_http(args.host)
        except RoArmHardwareError as e:
            print(f"[hardware] {e}", file=sys.stderr)
            hardware = None

    robot = Robot(hardware)
    robot.start()
    ears = None if args.no_voice else Ears()
    code = secrets.token_urlsafe(6) if args.lan else None
    address = lan_address()
    ssl = {}
    if args.https:
        try:
            cert, key = self_signed_cert(address)
            ssl = {"ssl_certfile": cert, "ssl_keyfile": key}
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"[https] could not create a certificate ({e}); serving plain HTTP", file=sys.stderr)
    scheme = "https" if ssl else "http"

    print(f"\n  On this machine:  {scheme}://127.0.0.1:{args.port_http}")
    if args.lan:
        print(f"  On your phone:    {scheme}://{address}:{args.port_http}/?code={code}")
        if not ssl:
            print("  (the phone's microphone needs --https; everything else works over http)")
        else:
            print("  (the phone will warn about the self-signed certificate once; accept it)")
    print()

    try:
        uvicorn.run(create_app(robot, ears, code), host="0.0.0.0" if args.lan else "127.0.0.1",
                    port=args.port_http, log_level="warning", **ssl)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
