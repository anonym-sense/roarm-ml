// Camera + hand tracking in the browser. Reports where the hand is, in metres, in
// the camera's own frame (x right, y down, z away from the lens).
//
// Distance comes from apparent size: the tracker gives the hand's real proportions
// (metric "world" landmarks) and their pixel positions, so depth = focal * real / pixels.
// The focal length is assumed from a typical webcam field of view; the server's
// calibration absorbs the error in that guess.

const VISION = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const FOV = (62 * Math.PI) / 180; // assumed horizontal field of view
const RIGID = [0, 1, 5, 9, 13, 17]; // wrist and knuckles: the part of the hand that keeps its shape
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

let landmarker = null;

async function loadTracker() {
  if (landmarker) return landmarker;
  const { HandLandmarker, FilesetResolver } = await import(VISION);
  const files = await FilesetResolver.forVisionTasks(`${VISION}/wasm`);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: "VIDEO", numHands: 1,
  });
  try {
    landmarker = await HandLandmarker.createFromOptions(files, options("GPU"));
  } catch (e) {
    landmarker = await HandLandmarker.createFromOptions(files, options("CPU"));
  }
  return landmarker;
}

/** Hand position in the camera frame from one detection. */
export function locate(image, world, width, height) {
  const focal = width / 2 / Math.tan(FOV / 2);
  // Pixels per metre, by least squares over the rigid part of the hand: the one scale
  // that best lays the hand's real outline (as the camera faces it) over its picture.
  const centre = (get) => RIGID.reduce((s, i) => s + get(i), 0) / RIGID.length;
  const wx = centre((i) => world[i].x), wy = centre((i) => world[i].y);
  const px = centre((i) => image[i].x * width), py = centre((i) => image[i].y * height);
  let top = 0, bottom = 0;
  for (const i of RIGID) {
    const ax = world[i].x - wx, ay = world[i].y - wy;
    top += ax * (image[i].x * width - px) + ay * (image[i].y * height - py);
    bottom += ax * ax + ay * ay;
  }
  if (bottom < 1e-5 || top <= 0) return null; // hand edge-on to the camera: no usable size
  const depth = (focal * bottom) / top;
  const point = (i) => {
    const z = depth + world[i].z;
    return [((image[i].x - 0.5) * width * z) / focal, ((image[i].y - 0.5) * height * z) / focal, z];
  };
  const mean = (ids) => ids.map(point).reduce((s, p) => s.map((v, k) => v + p[k] / ids.length), [0, 0, 0]);
  return {
    pinch: mean([4, 8]), // between thumb tip and index tip
    palm: mean([0, 5, 9, 13, 17]),
    open: Math.hypot(world[4].x - world[8].x, world[4].y - world[8].y, world[4].z - world[8].z),
    depth,
  };
}

/** Cameras the browser can see. Names are only filled in once camera permission is granted. */
export async function cameras() {
  const all = await navigator.mediaDevices.enumerateDevices();
  return all.filter((d) => d.kind === "videoinput")
    .map((d, i) => ({ id: d.deviceId, name: d.label || `Camera ${i + 1}` }));
}

export class HandCamera {
  constructor(video, overlay, onHand, onStatus) {
    Object.assign(this, { video, overlay, onHand, onStatus, stream: null, running: false });
  }

  /** deviceId: a camera from cameras(), or empty for the browser's default. */
  async start(deviceId = "") {
    this.onStatus("Starting the camera…");
    const which = deviceId ? { deviceId: { exact: deviceId } } : {};
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { ...which, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
    });
    this.deviceId = this.stream.getVideoTracks()[0].getSettings().deviceId || deviceId;
    this.video.srcObject = this.stream;
    await this.video.play();
    this.onStatus("Loading the hand tracker…");
    const tracker = await loadTracker();
    this.running = true;
    this.onStatus("Show me your hand.");
    let last = -1;
    const tick = () => {
      if (!this.running) return;
      if (this.video.readyState >= 2 && this.video.currentTime !== last) {
        last = this.video.currentTime;
        const found = tracker.detectForVideo(this.video, performance.now());
        this.draw(found.landmarks?.[0]);
        const hand = found.landmarks?.[0]
          ? locate(found.landmarks[0], found.worldLandmarks[0], this.video.videoWidth, this.video.videoHeight)
          : null;
        this.onHand(this.steady(hand));
      }
      this.video.requestVideoFrameCallback ? this.video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
    };
    tick();
  }

  /** Distance is the noisiest number and changes slowly, so it is smoothed harder than the rest. */
  steady(hand) {
    if (!hand) { this.depth = null; return null; }
    const jump = this.depth && Math.abs(hand.depth - this.depth) / this.depth > 0.25;
    this.depth = !this.depth || jump ? hand.depth : this.depth + 0.25 * (hand.depth - this.depth);
    const k = this.depth / hand.depth;
    const scale = (p) => p.map((v) => v * k);
    return { ...hand, pinch: scale(hand.pinch), palm: scale(hand.palm), depth: this.depth };
  }

  stop() {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.draw(null);
  }

  draw(points) {
    const c = this.overlay, g = c.getContext("2d");
    c.width = this.video.videoWidth || 640;
    c.height = this.video.videoHeight || 480;
    g.clearRect(0, 0, c.width, c.height);
    if (!points) return;
    g.lineWidth = 3;
    g.strokeStyle = "#5cc8ff";
    for (const [a, b] of BONES) {
      g.beginPath();
      g.moveTo(points[a].x * c.width, points[a].y * c.height);
      g.lineTo(points[b].x * c.width, points[b].y * c.height);
      g.stroke();
    }
    g.fillStyle = "#ff9a3d"; // the pinch point the arm aims for
    g.beginPath();
    g.arc(((points[4].x + points[8].x) / 2) * c.width, ((points[4].y + points[8].y) / 2) * c.height, 8, 0, 7);
    g.fill();
  }
}
