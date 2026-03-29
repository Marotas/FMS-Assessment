import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ─── State ──────────────────────────────────────────────────────
let poseLandmarker = null;
let lastVideoTime = -1;
let rafId = null;
const EMA = {};          // exponential moving averages for smoothing
const EMA_ALPHA = 0.25;  // smoothing factor (lower = smoother, more lag)

// FPS tracking
let fpsFrames = 0;
let fpsTs    = performance.now();
let currentFps = 0;

// ─── DOM refs ───────────────────────────────────────────────────
const video     = document.getElementById('video');
const canvas    = document.getElementById('canvas');
const ctx       = canvas.getContext('2d');
const loading   = document.getElementById('loading-overlay');
const loadTxt   = document.getElementById('loading-text');
const statusDot = document.getElementById('status-dot');
const statusTxt = document.getElementById('status-text');
const phaseLabel = document.getElementById('phase-label');
const fpsLabel  = document.getElementById('fps-label');
const clock     = document.getElementById('cam-clock');
const placeholder = document.getElementById('cam-placeholder');
const fmsTotalEl = document.getElementById('fms-total');
const fmsDescEl  = document.getElementById('fms-desc');

// ─── Smoothing ──────────────────────────────────────────────────
function ema(key, value) {
  if (EMA[key] === undefined || isNaN(EMA[key])) { EMA[key] = value; return value; }
  EMA[key] = EMA_ALPHA * value + (1 - EMA_ALPHA) * EMA[key];
  return EMA[key];
}

// ─── Angle helpers ──────────────────────────────────────────────
// 3D angle at point b (world landmarks: y is UP)
function angle3D(a, b, c) {
  const ba = [a.x-b.x, a.y-b.y, a.z-b.z];
  const bc = [c.x-b.x, c.y-b.y, c.z-b.z];
  const dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2];
  const mag = Math.hypot(...ba) * Math.hypot(...bc);
  if (mag < 1e-8) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot/mag))) * (180/Math.PI);
}

// 2D angle at point b (screen/normalized landmarks: y is DOWN)
function angle2D(a, b, c) {
  const ba = [a.x-b.x, a.y-b.y];
  const bc = [c.x-b.x, c.y-b.y];
  const dot = ba[0]*bc[0] + ba[1]*bc[1];
  const mag = Math.hypot(...ba) * Math.hypot(...bc);
  if (mag < 1e-8) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot/mag))) * (180/Math.PI);
}

// ─── FMS Scoring ────────────────────────────────────────────────
// direction: 'hi' = higher value is better, 'lo' = lower is better
function fmsScore(value, t3, t2, direction) {
  if (direction === 'hi') {
    if (value >= t3) return 3;
    if (value >= t2) return 2;
    return 1;
  } else {
    if (value <= t3) return 3;
    if (value <= t2) return 2;
    return 1;
  }
}

// Score → CSS class / color
function scoreClass(s) {
  return s === 3 ? 's3' : s === 2 ? 's2' : 's1';
}
function scoreColor(s) {
  return s === 3 ? 'var(--score-3)' : s === 2 ? 'var(--score-2)' : 'var(--score-1)';
}
function scoreLabel(s) {
  return s === 3 ? 'Optimal' : s === 2 ? 'Kompensation' : 'Fehlerhaft';
}

// ─── UI helpers ─────────────────────────────────────────────────
function setCard(id, score, valL, valR, barPct) {
  const card  = document.getElementById(`card-${id}`);
  const badge = document.getElementById(`badge-${id}`);
  const bar   = document.getElementById(`bar-${id}`);
  const dot   = document.getElementById(`dot-${id}`);
  const sc    = scoreClass(score);
  const color = scoreColor(score);

  card.className  = `metric-card ${sc}`;
  badge.className = `score-badge ${sc}`;
  badge.textContent = `${score} / 3`;
  if (dot) { dot.className = `matrix-dot ${sc}`; }
  if (bar) { bar.style.width = `${Math.max(0,Math.min(100,barPct))}%`; bar.style.background = color; }

  function setVal(elId, v, unit='°') {
    const el = document.getElementById(elId);
    if (!el) return;
    el.className = `val-number ${sc}`;
    el.innerHTML = v !== null ? `${Math.round(v)}<span class="val-unit">${unit}</span>` : `--<span class="val-unit">${unit}</span>`;
  }
  if (valL !== null) setVal(`${id}-l`, valL, id==='fh'?'':'°');
  if (valR !== null) setVal(`${id}-r`, valR, id==='fh'?'':'°');

  // single-value metrics
  const singleIds = {'tv':'tv-val','kv':null,'as':'as-val','ts':'ts-val'};
  if (singleIds[id]) setVal(singleIds[id], valL, '°');
}

// ─── MediaPipe init ─────────────────────────────────────────────
async function initMediaPipe() {
  loadTxt.textContent = 'Vision-Modell wird geladen...';
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  loadTxt.textContent = 'Pose Landmarker wird initialisiert...';
  poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence:  0.5,
    minTrackingConfidence:      0.5,
    outputSegmentationMasks: false
  });
  loadTxt.textContent = 'Kamera wird gestartet...';
  await startCamera();
}

// ─── Camera ─────────────────────────────────────────────────────
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = stream;
    video.onloadeddata = () => {
      placeholder.style.display = 'none';
      statusDot.classList.add('live');
      statusTxt.textContent = 'Live · Echtzeit-Analyse aktiv';
      phaseLabel.textContent = 'Pose-Erkennung aktiv';
      loading.classList.add('hidden');
      rafId = requestAnimationFrame(loop);
    };
  } catch (err) {
    loadTxt.textContent = `Kamerafehler: ${err.message}`;
    statusTxt.textContent = 'Kamerazugriff verweigert';
    loading.classList.remove('hidden');
  }
}

// ─── Main loop ──────────────────────────────────────────────────
function loop() {
  // Clock
  const now = new Date();
  clock.textContent = now.toLocaleTimeString('de-DE');

  // FPS
  fpsFrames++;
  const elapsed = performance.now() - fpsTs;
  if (elapsed >= 1000) {
    currentFps = Math.round(fpsFrames * 1000 / elapsed);
    fpsLabel.textContent = `${currentFps} fps`;
    fpsFrames = 0;
    fpsTs = performance.now();
  }

  // Resize canvas to match video
  if (video.videoWidth && (canvas.width !== video.videoWidth)) {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (poseLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const results = poseLandmarker.detectForVideo(video, performance.now());

    if (results.landmarks && results.landmarks.length > 0) {
      const lm  = results.landmarks[0];       // normalized screen coords (y down)
      const wlm = results.worldLandmarks[0];  // world coords in meters (y up)

      // Draw skeleton
      drawSkeleton(lm);

      // Compute and display metrics
      processMetrics(lm, wlm);

      phaseLabel.textContent = 'Analyse läuft';
    } else {
      phaseLabel.textContent = 'Person nicht erkannt';
      resetMetrics();
    }
  }

  rafId = requestAnimationFrame(loop);
}

// ─── Skeleton drawing ────────────────────────────────────────────
const CONNECTIONS = [
  [11,13],[13,15], [12,14],[14,16],  // arms
  [11,12],                            // shoulders
  [23,25],[25,27],[27,29],[27,31],    // left leg + foot
  [24,26],[26,28],[28,30],[28,32],    // right leg + foot
  [11,23],[12,24],[23,24],            // torso
  [29,31],[30,32]                     // feet
];

const JOINT_HIGHLIGHT = new Set([23,24,25,26,27,28,29,30,31,32,11,12]); // body only

function drawSkeleton(lm) {
  const W = canvas.width, H = canvas.height;
  const px = i => lm[i].x * W;
  const py = i => lm[i].y * H;

  // Connections
  ctx.lineCap = 'round';
  for (const [a, b] of CONNECTIONS) {
    if (lm[a].visibility < 0.3 || lm[b].visibility < 0.3) continue;
    ctx.beginPath();
    ctx.moveTo(px(a), py(a));
    ctx.lineTo(px(b), py(b));
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.55)';
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  // Joints
  for (let i = 11; i <= 32; i++) {
    if (lm[i].visibility < 0.3) continue;
    const x = px(i), y = py(i);
    const isKey = JOINT_HIGHLIGHT.has(i);
    ctx.beginPath();
    ctx.arc(x, y, isKey ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle   = isKey ? 'rgba(0,212,255,0.9)' : 'rgba(255,255,255,0.7)';
    ctx.shadowColor = isKey ? 'rgba(0,212,255,0.6)' : 'transparent';
    ctx.shadowBlur  = isKey ? 8 : 0;
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}

// ─── Angle annotations on canvas ─────────────────────────────────
function drawAngleLabel(x, y, text, color) {
  const W = canvas.width, H = canvas.height;
  const px = x * W, py = y * H;
  ctx.font = 'bold 13px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  // background pill
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = 'rgba(6,10,22,0.72)';
  ctx.beginPath();
  ctx.roundRect(px - tw/2 - 6, py - 14, tw + 12, 20, 4);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.fillText(text, px, py);
}

// ─── Metric computation ──────────────────────────────────────────
function processMetrics(lm, wlm) {
  // Landmark shortcuts (world coords, y up)
  const ls = wlm[11], rs = wlm[12];   // shoulders
  const lh = wlm[23], rh = wlm[24];   // hips
  const lk = wlm[25], rk = wlm[26];   // knees
  const la = wlm[27], ra = wlm[28];   // ankles
  const lhe= wlm[29], rhe= wlm[30];   // heels
  const lfi= wlm[31], rfi= wlm[32];   // foot index

  // Screen landmark shortcuts (y down, normalized 0-1)
  const slh= lm[23], srh= lm[24];
  const slk= lm[25], srk= lm[26];
  const sla= lm[27], sra= lm[28];

  // ── 1. Knee Flexion (3D) ──────────────────────────────────────
  // Interior angle at knee: standing ≈ 180°, squatting decreases
  // Flexion = 180 - interior_angle
  const kfL = ema('kfL', 180 - angle3D(lh, lk, la));
  const kfR = ema('kfR', 180 - angle3D(rh, rk, ra));
  const kfMin = Math.min(kfL, kfR);
  const kfScore = fmsScore(kfMin, 120, 90, 'hi');
  setCard('kf', kfScore, kfL, kfR, (kfMin / 150) * 100);

  // Draw angle labels on skeleton
  const kfColor = scoreColor(kfScore);
  drawAngleLabel(lm[25].x, lm[25].y - 0.03, `${Math.round(kfL)}°`, kfColor);
  drawAngleLabel(lm[26].x, lm[26].y - 0.03, `${Math.round(kfR)}°`, kfColor);

  // ── 2. Hip Flexion (3D) ───────────────────────────────────────
  const hfL = ema('hfL', 180 - angle3D(ls, lh, lk));
  const hfR = ema('hfR', 180 - angle3D(rs, rh, rk));
  const hfMin = Math.min(hfL, hfR);
  const hfScore = fmsScore(hfMin, 110, 80, 'hi');
  setCard('hf', hfScore, hfL, hfR, (hfMin / 140) * 100);

  drawAngleLabel(lm[23].x, lm[23].y - 0.03, `${Math.round(hfL)}°`, scoreColor(hfScore));
  drawAngleLabel(lm[24].x, lm[24].y - 0.03, `${Math.round(hfR)}°`, scoreColor(hfScore));

  // ── 3. Ankle Dorsiflexion (3D) ────────────────────────────────
  // Angle between shin (knee→ankle) and foot (ankle→foot_index)
  // Standing ≈ 90°. Shin tilts forward during squat → angle decreases.
  // Dorsiflexion = 90 - angle (when angle < 90)
  const ankleAngleL = angle3D(lk, la, lfi);
  const ankleAngleR = angle3D(rk, ra, rfi);
  const dfL = ema('dfL', Math.max(0, 90 - ankleAngleL));
  const dfR = ema('dfR', Math.max(0, 90 - ankleAngleR));
  const dfMin = Math.min(dfL, dfR);
  const dfScore = fmsScore(dfMin, 30, 20, 'hi');
  setCard('df', dfScore, dfL, dfR, (dfMin / 45) * 100);

  // ── 4. Trunk Forward Lean (3D) ────────────────────────────────
  // Trunk vector: hip_mid → shoulder_mid in world space (y up)
  const smx = (ls.x + rs.x)/2, smy = (ls.y + rs.y)/2, smz = (ls.z + rs.z)/2;
  const hmx = (lh.x + rh.x)/2, hmy = (lh.y + rh.y)/2, hmz = (lh.z + rh.z)/2;
  // Forward lean: angle in sagittal plane (y-z) relative to vertical
  const tvRaw = Math.atan2(Math.abs(smz - hmz), Math.abs(smy - hmy)) * (180/Math.PI);
  const tvVal = ema('tv', tvRaw);
  const tvScore = fmsScore(tvVal, 15, 35, 'lo');
  setCard('tv', tvScore, tvVal, null, Math.max(0, 100 - (tvVal / 45) * 100));

  // ── 5. Knee Valgus (2D frontal plane from screen landmarks) ───
  // 2D angle at knee using x,y screen coords. Perfect alignment = 180°.
  // Valgus = 180 - angle_at_knee_in_frontal_plane
  const frontAngleL = angle2D(slh, slk, sla);
  const frontAngleR = angle2D(srh, srk, sra);
  const kvL = ema('kvL', Math.max(0, 180 - frontAngleL));
  const kvR = ema('kvR', Math.max(0, 180 - frontAngleR));
  const kvMax = Math.max(kvL, kvR);
  const kvScore = fmsScore(kvMax, 5, 12, 'lo');
  setCard('kv', kvScore, kvL, kvR, Math.max(0, 100 - (kvMax / 20) * 100));

  // Draw valgus indicators on knees
  drawAngleLabel(lm[25].x, lm[25].y + 0.05, `${Math.round(kvL)}°`, scoreColor(kvScore));
  drawAngleLabel(lm[26].x, lm[26].y + 0.05, `${Math.round(kvR)}°`, scoreColor(kvScore));

  // ── 6. L/R Asymmetry Knee ────────────────────────────────────
  const asymVal = ema('asym', Math.abs(kfL - kfR));
  const asymScore = fmsScore(asymVal, 5, 10, 'lo');
  setCard('as', asymScore, asymVal, null, Math.max(0, 100 - (asymVal / 15) * 100));

  // ── 7. Trunk Lateral Lean (2D frontal) ────────────────────────
  const slsm = lm[11], srsm = lm[12]; // screen shoulders
  const slhm = lm[23], srhm = lm[24]; // screen hips
  const shoulderMidX = (slsm.x + srsm.x) / 2;
  const shoulderMidY = (slsm.y + srsm.y) / 2;
  const hipMidX = (slhm.x + srhm.x) / 2;
  const hipMidY = (slhm.y + srhm.y) / 2;
  const tsDeltaX = shoulderMidX - hipMidX;
  const tsDeltaY = hipMidY - shoulderMidY; // positive since hips below shoulders
  const tsRaw = Math.atan2(Math.abs(tsDeltaX), Math.max(tsDeltaY, 0.01)) * (180/Math.PI);
  const tsVal = ema('ts', tsRaw);
  const tsScore = fmsScore(tsVal, 3, 8, 'lo');
  setCard('ts', tsScore, tsVal, null, Math.max(0, 100 - (tsVal / 12) * 100));

  // ── 8. Heel Rise (world coords, y up) ─────────────────────────
  // When heel rises, heel.y > foot_index.y (both in world space, y up)
  // At rest: heel.y ≈ foot_index.y or slightly lower due to foot arch
  // Rise in meters → convert to rough mm estimate
  const heelRiseL = ema('heelL', Math.max(0, (lhe.y - lfi.y) * 1000)); // mm
  const heelRiseR = ema('heelR', Math.max(0, (rhe.y - rfi.y) * 1000));
  const heelMax = Math.max(heelRiseL, heelRiseR);
  const fhScore = fmsScore(heelMax, 1, 15, 'lo');

  // Display as OK / Grenz / Hoch
  const fhLabelL = heelRiseL <= 1 ? 'OK' : heelRiseL <= 15 ? 'Grenz' : 'Hoch';
  const fhLabelR = heelRiseR <= 1 ? 'OK' : heelRiseR <= 15 ? 'Grenz' : 'Hoch';

  // Overwrite fh values as text labels (not degrees)
  const card  = document.getElementById('card-fh');
  const badge = document.getElementById('badge-fh');
  const bar   = document.getElementById('bar-fh');
  const dotFh = document.getElementById('dot-fh');
  const sc    = scoreClass(fhScore);
  card.className  = `metric-card ${sc}`;
  badge.className = `score-badge ${sc}`;
  badge.textContent = `${fhScore} / 3`;
  if (dotFh) dotFh.className = `matrix-dot ${sc}`;
  if (bar) { bar.style.width = `${Math.max(0, 100 - (heelMax/20)*100)}%`; bar.style.background = scoreColor(fhScore); }
  const fhL = document.getElementById('fh-l');
  const fhR = document.getElementById('fh-r');
  if (fhL) { fhL.className = `val-number ${sc}`; fhL.textContent = fhLabelL; }
  if (fhR) { fhR.className = `val-number ${sc}`; fhR.textContent = fhLabelR; }

  // ── FMS Gesamtscore (= minimum of all scores) ─────────────────
  const scores = [kfScore, hfScore, dfScore, tvScore, kvScore, asymScore, tsScore, fhScore];
  const totalScore = Math.min(...scores);
  const sc2 = scoreClass(totalScore);
  fmsTotalEl.className = `fms-score-value ${sc2}`;
  fmsTotalEl.innerHTML = `${totalScore}<span class="fms-score-denom"> / 3</span>`;
  fmsDescEl.textContent = scoreLabel(totalScore);
  fmsDescEl.style.color = scoreColor(totalScore);
}

// ─── Reset metrics to pending state ─────────────────────────────
function resetMetrics() {
  const ids = ['kf','hf','df','tv','kv','as','ts','fh'];
  for (const id of ids) {
    const card  = document.getElementById(`card-${id}`);
    const badge = document.getElementById(`badge-${id}`);
    const dot   = document.getElementById(`dot-${id}`);
    if (card)  card.className  = 'metric-card';
    if (badge) { badge.className = 'score-badge'; badge.textContent = '— / 3'; }
    if (dot)   dot.className = 'matrix-dot';
  }
  fmsTotalEl.className = 'fms-score-value';
  fmsTotalEl.innerHTML = `--<span class="fms-score-denom"> / 3</span>`;
  fmsDescEl.textContent = 'Person nicht erkannt';
  fmsDescEl.style.color = '';
  // clear EMA
  for (const k of Object.keys(EMA)) delete EMA[k];
}

// ─── Boot ────────────────────────────────────────────────────────
initMediaPipe().catch(err => {
  console.error(err);
  loadTxt.textContent = `Fehler: ${err.message}`;
});
