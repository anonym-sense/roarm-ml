// RoArm web app: draws the arm from the poses the server streams, and sends it commands.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const $ = (id) => document.getElementById(id);
const post = (url, body) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) })
    .then((r) => r.json());
const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...kids);
  return node;
};

const JOINTS = ["base", "shoulder", "elbow", "gripper"];
const LINK_COLORS = { base_link: 0x2a2f38, gripper_link: 0xff9a3d };
let info = null;
let state = null;

// ---------------------------------------------------------------- 3D scene
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
camera.position.set(0.86, 0.56, 0.86);
let renderer = null;
let controls = null;
const robot = new THREE.Group();
robot.rotation.x = -Math.PI / 2; // the simulator is z-up, three.js is y-up
scene.add(robot);
const parts = []; // moving links, in the order the server sends their poses
const targets = [];
const trail = { points: [], line: null, max: 70 };

function buildScene() {
  try {
    renderer = new THREE.WebGLRenderer({ canvas: $("view"), antialias: true, alpha: true });
  } catch (e) {
    $("gl-notice").hidden = false;
    $("gl-notice").textContent = "3D view unavailable in this browser. Chat and controls still work.";
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.1, 0.24, 0);
  controls.enableDamping = true;
  controls.minDistance = 0.3;
  controls.maxDistance = 2.2;
  controls.maxPolarAngle = Math.PI * 0.495;

  scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x1a1410, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(0.7, 1.4, 0.9);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -0.7, right: 0.7, top: 0.7, bottom: -0.7, near: 0.1, far: 4 });
  key.shadow.bias = -0.0004;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x5cc8ff, 0.35);
  rim.position.set(-0.9, 0.5, -0.8);
  scene.add(rim);

  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(1.4, 96),
    new THREE.MeshStandardMaterial({ color: 0x18202b, roughness: 0.92, metalness: 0.05, transparent: true, opacity: 0.92 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.PolarGridHelper(1.4, 12, 7, 96, 0x2a3443, 0x1d2531);
  grid.position.y = 0.0006;
  scene.add(grid);

  const trailGeom = new THREE.BufferGeometry();
  trailGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(trail.max * 3), 3));
  trail.line = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0x5cc8ff, transparent: true, opacity: 0.55 }));
  trail.line.frustumCulled = false;
  robot.add(trail.line);

  const resize = () => {
    const { clientWidth: w, clientHeight: h } = $("stage");
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe($("stage"));
  resize();
  enableDragging();
  renderer.setAnimationLoop(frame);
}

// Drag the arm: the hand follows the pointer on a plane facing the camera,
// at the depth where the arm was grabbed.
const ray = new THREE.Raycaster();
const dragPlane = new THREE.Plane();
let drag = null;
let lastReach = 0;
const marker = new THREE.Mesh(new THREE.SphereGeometry(0.012, 20, 20),
  new THREE.MeshBasicMaterial({ color: 0x5cc8ff, transparent: true, opacity: 0.85 }));
marker.visible = false;
scene.add(marker);

function aim(e) {
  const box = $("view").getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - box.left) / box.width) * 2 - 1,
    -((e.clientY - box.top) / box.height) * 2 + 1), camera);
  return ray.intersectObjects(parts, true)[0];
}

function enableDragging() {
  // capture phase on the parent, so this runs before the orbit controls see the press
  $("stage").addEventListener("pointerdown", (e) => {
    if (e.target !== $("view") || !state?.links || state.torque === false) return;
    const hit = aim(e);
    if (!hit) return;
    controls.enabled = false;
    const tip = state.links[state.links.length - 1];
    const hand = robot.localToWorld(new THREE.Vector3(tip[0], tip[1], tip[2]));
    dragPlane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()).negate(), hit.point);
    drag = { offset: hand.sub(hit.point) };
    $("view").classList.add("grabbing");
  }, true);
  window.addEventListener("pointermove", (e) => {
    if (!drag) {
      if (e.target === $("view") && e.pointerType === "mouse") $("view").classList.toggle("grab", !!aim(e));
      return;
    }
    aim(e);
    const point = new THREE.Vector3();
    if (!ray.ray.intersectPlane(dragPlane, point)) return;
    point.add(drag.offset);
    marker.position.copy(point);
    marker.visible = true;
    const now = performance.now();
    if (now - lastReach < 45) return;
    lastReach = now;
    const local = robot.worldToLocal(point.clone());
    post("/api/reach", { xyz: [local.x, local.y, local.z] });
  });
  const release = () => {
    if (!drag) return;
    drag = null;
    controls.enabled = true;
    marker.visible = false;
    $("view").classList.remove("grabbing");
  };
  window.addEventListener("pointerup", release);
  window.addEventListener("pointercancel", release);
}

// Read-only hooks for automated checks of the 3D view.
window.__project = () => {
  if (!state?.links) return null;
  const tip = state.links[state.links.length - 1];
  const p = robot.localToWorld(new THREE.Vector3(tip[0], tip[1], tip[2])).project(camera);
  const box = $("view").getBoundingClientRect();
  return { x: Math.round(box.left + (p.x + 1) / 2 * box.width), y: Math.round(box.top + (1 - p.y) / 2 * box.height) };
};
window.__camera = () => camera.position.toArray().map((v) => +v.toFixed(3)).join(",");

function loadRobot() {
  if (!renderer) return;
  const loader = new STLLoader();
  info.links.forEach((link, i) => {
    const holder = new THREE.Group();
    robot.add(holder);
    if (i > 0) { parts.push(holder); targets.push(null); }
    loader.load(`/meshes/${link.mesh}`, (geometry) => {
      geometry.computeVertexNormals();
      const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
        color: LINK_COLORS[link.name] ?? 0xd9dee6, roughness: 0.42, metalness: 0.35 }));
      mesh.castShadow = mesh.receiveShadow = true;
      holder.add(mesh);
    });
  });
}

function frame() {
  if (state?.links) {
    state.links.forEach((pose, i) => {
      if (!parts[i]) return;
      const pos = new THREE.Vector3(pose[0], pose[1], pose[2]);
      const quat = new THREE.Quaternion(pose[3], pose[4], pose[5], pose[6]);
      if (!targets[i]) { parts[i].position.copy(pos); parts[i].quaternion.copy(quat); targets[i] = true; }
      parts[i].position.lerp(pos, 0.45); // smooth the 30 Hz stream up to the display rate
      parts[i].quaternion.slerp(quat, 0.45);
    });
    const tip = state.links[state.links.length - 1];
    const last = trail.points[trail.points.length - 1];
    if (!last || Math.hypot(tip[0] - last[0], tip[1] - last[1], tip[2] - last[2]) > 0.004) {
      trail.points.push(tip.slice(0, 3));
      if (trail.points.length > trail.max) trail.points.shift();
      const attr = trail.line.geometry.attributes.position;
      trail.points.forEach((pt, k) => attr.setXYZ(k, pt[0], pt[1], pt[2]));
      attr.needsUpdate = true;
      trail.line.geometry.setDrawRange(0, trail.points.length);
    }
  }
  controls.update();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- live link
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => $("link-dot").classList.add("on");
  ws.onclose = () => { $("link-dot").classList.remove("on"); setTimeout(connect, 1500); };
  ws.onmessage = (msg) => {
    state = JSON.parse(msg.data);
    $("chip-gesture").hidden = !state.gesture;
    if (state.gesture) $("chip-gesture").textContent = state.gesture;
    $("chip-inventing").hidden = !state.inventing;
    $("chip-hw").hidden = !state.hardware;
    $("mirror").disabled = !state.hardware || !state.torque;
    $("mirror").checked = !!state.mirror;
    $("torque").disabled = !state.hardware;
    if ($("torque-confirm").hidden) $("torque").checked = !!state.torque;
    $("chip-limp").hidden = !state.hardware || state.torque;
    $("hw-hint").textContent = !state.hardware
      ? "No real arm connected. Start the server with --hw serial to use these."
      : !state.torque
        ? "Motors released: move the arm by hand and the 3D view follows it. Turn the motors back on before mirroring."
        : "The real arm is connected. Turn mirroring on to move it.";
    ears = state.ears;
    (state.messages || []).forEach(addMessage);
    if (!draggingSlider && state.q) state.q.forEach((v, i) => setSlider(i, v));
  };
}

// ---------------------------------------------------------------- chat
const SUGGESTIONS = ["hello", "strong nod", "wave slowly then bow", "point right", "dance",
  "that was great!", "look around", "remember that as patrol"];

function addMessage(m) {
  const node = el("div", { className: `msg ${m.who}` }, m.text);
  if (m.source === "voice") node.append(el("span", { className: "tag", textContent: "voice" }));
  if (m.who === "arm" && m.event) {
    const rate = el("div", { className: "rate" });
    [[1, "up", "Good"], [-1, "down", "Not like that"]].forEach(([value, cls, label]) => {
      const b = el("button", { className: cls, textContent: label, type: "button" });
      b.onclick = async () => {
        rate.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
        b.classList.add("on");
        await post("/api/feedback", { event: m.event, value });
        learningStale = true;
      };
      rate.append(b);
    });
    node.append(rate);
  }
  $("log").append(node);
  $("log").scrollTop = $("log").scrollHeight;
  if (/learned|saved|got it|forgot/i.test(m.text)) { learningStale = true; refreshInfo(); }
}

function say(text) {
  text = text.trim();
  if (text) post("/api/say", { text });
}

$("say").onsubmit = (e) => { e.preventDefault(); say($("text").value); $("text").value = ""; };
SUGGESTIONS.forEach((s) => $("suggest").append(el("button", { textContent: s, type: "button", onclick: () => say(s) })));
$("btn-home").onclick = () => post("/api/home");
$("btn-stop").onclick = () => post("/api/stop");

// ---------------------------------------------------------------- voice
let ears = false;
let recorder = null;

function micHint(text) { $("mic-hint").hidden = !text; $("mic-hint").textContent = text || ""; }

async function toggleMic() {
  if (recorder) { recorder.stop(); return; }
  if (!navigator.mediaDevices || !window.isSecureContext) {
    micHint("The microphone needs a secure page. Start the server with --https and open the https link.");
    return;
  }
  if (!ears) { micHint("The speech model is still loading. Try again in a moment."); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    micHint("Microphone permission was refused.");
    return;
  }
  micHint("");
  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    audioCtx.close();
    $("mic").classList.remove("live");
    $("mic").classList.add("busy");
    const type = recorder.mimeType;
    recorder = null;
    try {
      const r = await fetch("/api/audio", { method: "POST", body: new Blob(chunks, { type }) });
      const out = await r.json();
      if (out.error) micHint(out.error);
      else if (!out.text) micHint("I didn't catch that.");
    } finally {
      $("mic").classList.remove("busy");
    }
  };

  // stop by itself after a pause in speech
  const audioCtx = new AudioContext();
  const analyser = audioCtx.createAnalyser();
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  const started = performance.now();
  let spoke = false, quietSince = null;
  const watch = () => {
    if (!recorder) return;
    analyser.getFloatTimeDomainData(buf);
    const rms = Math.sqrt(buf.reduce((s, v) => s + v * v, 0) / buf.length);
    const now = performance.now();
    if (rms > 0.02) { spoke = true; quietSince = null; }
    else if (spoke) quietSince ??= now;
    if ((quietSince && now - quietSince > 1100) || now - started > 12000 || (!spoke && now - started > 6000)) {
      recorder.stop();
      return;
    }
    requestAnimationFrame(watch);
  };
  recorder.start();
  $("mic").classList.add("live");
  watch();
}
$("mic").onclick = toggleMic;

// ---------------------------------------------------------------- gestures
const choice = { speed: "normal", size: "normal", side: "left" };

function segment(id, key, options) {
  const box = $(id);
  box.replaceChildren(...options.map((option) => {
    const b = el("button", { textContent: option, type: "button", className: choice[key] === option ? "on" : "" });
    b.onclick = () => { choice[key] = option; segment(id, key, options); };
    return b;
  }));
}

function drawGrid() {
  const q = $("search").value.trim().toLowerCase();
  const shown = info.motions.filter((m) => !q || m.name.replace(/_/g, " ").includes(q) || m.about.toLowerCase().includes(q));
  $("grid").replaceChildren(...shown.map((m) => {
    const card = el("button", { className: `card${m.learned ? " learned" : ""}`, type: "button" },
      el("b", { textContent: m.name.replace(/_/g, " ") }), el("small", { textContent: m.about }));
    card.onclick = () => post("/api/play", { name: m.name, ...choice });
    return card;
  }));
}
$("search").oninput = drawGrid;

async function refreshInfo() {
  info = await (await fetch("/api/info")).json();
  drawGrid();
}

// ---------------------------------------------------------------- learning
let learningStale = true;

async function drawLearning() {
  learningStale = false;
  const b = await (await fetch("/api/brain")).json();
  const c = b.counts;
  const tiles = [[c.motions, "built-in motions"], [c.learned, "invented gestures"], [c.skills, "skills"],
    [c.phrases, "phrases learned"], [c.interactions, "things done"], [`${c.up} / ${c.down}`, "liked / disliked"]];
  $("tiles").replaceChildren(...tiles.map(([n, label]) =>
    el("div", { className: "tile" }, el("b", { textContent: n }), el("span", { textContent: label }))));

  const peak = Math.max(1, ...b.days.map((d) => d.used + d.lessons));
  $("days").replaceChildren(...b.days.map((d) => {
    const col = el("div", { title: `${d.day}: did ${d.used}, learned ${d.lessons}` });
    if (!d.used && !d.lessons) col.append(el("i", { className: "empty" }));
    if (d.used) col.append(el("i", { className: "used", style: `height:${Math.max(3, (d.used / peak) * 80)}px` }));
    if (d.lessons) col.append(el("i", { className: "lesson", style: `height:${Math.max(3, (d.lessons / peak) * 80)}px` }));
    return col;
  }));

  const rated = Object.values(b.taste.speed).some((t) => t.up + t.down > 0);
  const prefs = [];
  for (const group of ["speed", "size"]) {
    for (const [name, t] of Object.entries(b.taste[group])) {
      const total = t.up + t.down;
      prefs.push(el("div", { className: "pref" }, name,
        el("div", { className: "track" }, el("div", { className: "fill", style: `width:${total ? (t.up / total) * 100 : 0}%` })),
        el("span", { className: "count", textContent: total ? `${t.up} of ${total}` : "no votes" })));
    }
  }
  $("taste").replaceChildren(...(rated ? prefs
    : [el("p", { className: "empty-note", textContent: "Nothing yet. Rate what it does with the buttons under each reply and it will lean toward the speed and size you like." })]));

  $("skills").replaceChildren(...(b.skills.length ? b.skills.map((s) => {
    const run = el("button", { textContent: "Run", type: "button", onclick: () => say(s.name) });
    const drop = el("button", { textContent: "Forget", type: "button" });
    drop.onclick = async () => { await fetch(`/api/skill/${encodeURIComponent(s.name)}`, { method: "DELETE" }); drawLearning(); };
    return el("div", { className: "row" },
      el("div", {}, el("b", { textContent: s.name }), el("small", { textContent: s.steps.join("  >  ") })),
      el("div", { className: "acts" }, run, drop));
  }) : [el("p", { className: "empty-note", textContent: "No skills yet. Teach one below, or say \"remember that as …\" after a command." })]));

  $("invented").replaceChildren(...(b.learned.length ? b.learned.map((m) =>
    el("div", { className: "row" },
      el("div", {}, el("b", { textContent: m.name.replace(/_/g, " ") }), el("small", { textContent: m.about })),
      el("div", { className: "acts" }, el("button", { textContent: "Play", type: "button", onclick: () => post("/api/play", { name: m.name }) }))))
    : [el("p", { className: "empty-note", textContent: "None yet." })]));

  $("lessons").replaceChildren(...(b.lessons.length ? b.lessons.map((l) =>
    el("div", { className: "row" }, el("div", {},
      el("small", { textContent: new Date(l.t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) }), l.text)))
    : [el("p", { className: "empty-note", textContent: "Nothing learned yet." })]));
}

$("skill-form").onsubmit = async (e) => {
  e.preventDefault();
  const name = $("skill-name").value.trim(), steps = $("skill-steps").value.trim();
  if (!name || !steps) return;
  await post("/api/skill", { name, steps });
  $("skill-name").value = $("skill-steps").value = "";
  drawLearning();
};

// ---------------------------------------------------------------- manual control
let draggingSlider = false;

function buildSliders() {
  $("sliders").replaceChildren(...JOINTS.map((name, i) => {
    const [lo, hi] = info.bounds[i];
    const out = el("output", { id: `out-${i}`, textContent: "0.00" });
    const input = el("input", { type: "range", id: `q-${i}`, min: lo, max: hi, step: 0.01, value: 0 });
    input.oninput = () => {
      draggingSlider = true;
      out.textContent = (+input.value).toFixed(2);
      post("/api/joints", { q: JOINTS.map((_, k) => +$(`q-${k}`).value) });
    };
    input.onchange = () => { draggingSlider = false; };
    return el("div", { className: "slider" }, el("label", {}, name, out), input);
  }));
}

function setSlider(i, v) {
  const input = $(`q-${i}`);
  if (!input) return;
  input.value = v;
  $(`out-${i}`).textContent = v.toFixed(2);
}

$("mirror").onchange = () => post("/api/mirror", { on: $("mirror").checked });
// Releasing the motors lets the arm drop, so it takes a second, deliberate tap.
$("torque").onchange = () => {
  if ($("torque").checked) post("/api/torque", { on: true });
  else $("torque-confirm").hidden = false;
};
$("torque-release").onclick = () => { $("torque-confirm").hidden = true; post("/api/torque", { on: false }); };
$("torque-cancel").onclick = () => { $("torque-confirm").hidden = true; $("torque").checked = true; };

// ---------------------------------------------------------------- tabs + start
document.querySelectorAll(".tabs button").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tabs button, .pane").forEach((x) => x.classList.remove("active"));
    tab.classList.add("active");
    $(`pane-${tab.dataset.pane}`).classList.add("active");
    if (tab.dataset.pane === "learning" && learningStale) drawLearning();
    if (tab.dataset.pane === "learning") learningStale = false;
  };
});

buildScene();
await refreshInfo();
loadRobot();
segment("seg-speed", "speed", info.speeds);
segment("seg-size", "size", info.sizes);
segment("seg-side", "side", ["left", "right"]);
buildSliders();
if (location.hash === "#learning") document.querySelector('[data-pane="learning"]').click();
if (location.hash === "#gestures") document.querySelector('[data-pane="gestures"]').click();
connect();
