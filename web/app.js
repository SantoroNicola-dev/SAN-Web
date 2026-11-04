/**
 * SAN-Web Browser Inference Engine (Microsoft ONNX Runtime Web — CPU WASM)
 * Loads `../models/san_tagger.onnx` and `../models/labels.json`, preprocesses
 * any image into a [1, 3, 224, 224] RGB float32 tensor in [0.0, 1.0], and outputs
 * calibrated multi-label English tags.
 */

const MODEL_URL = "../models/san_tagger.onnx?v=3.0";
const LABELS_URL = "../models/labels.json?v=3.0";
const INPUT_SIZE = 224;

let ortSession = null;
let vocabulary = [];
let lastScores = null;

// DOM Elements
const statusDot = document.getElementById("statusDot");
const statusTitle = document.getElementById("statusTitle");
const statusMeta = document.getElementById("statusMeta");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const inputCanvas = document.getElementById("inputCanvas");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdVal = document.getElementById("thresholdVal");
const topKSlider = document.getElementById("topKSlider");
const topKVal = document.getElementById("topKVal");
const runInferenceBtn = document.getElementById("runInferenceBtn");
const latencyPill = document.getElementById("latencyPill");
const tagCountPill = document.getElementById("tagCountPill");
const tagsContainer = document.getElementById("tagsContainer");

function setStatus(state, title, meta) {
  statusDot.className = `status-indicator ${state}`;
  statusTitle.textContent = title;
  statusMeta.textContent = meta;
}

/**
 * Initialize ONNX Runtime Web on CPU (WASM) and load the 4,660 English labels.
 */
async function initEngine() {
  try {
    setStatus("loading", "Loading Tag Vocabulary...", "Fetching models/labels.json (4,660 tags)");
    const vocabRes = await fetch(LABELS_URL);
    if (!vocabRes.ok) throw new Error(`Failed to fetch labels.json (${vocabRes.status})`);
    const vocabJson = await vocabRes.json();
    vocabulary = vocabJson.labels || [];

    // Configure ONNX Runtime Web for multi-threaded CPU WASM execution
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
    ort.env.wasm.simd = true;

    setStatus("loading", "Loading ONNX Model (344 MB)...", "Initializing CPU WASM InferenceSession");
    const t0 = performance.now();
    ortSession = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    const loadSec = ((performance.now() - t0) / 1000).toFixed(1);

    setStatus(
      "ready",
      "ONNX CPU Engine Ready",
      `${vocabulary.length} tags loaded in ${loadSec}s (Provider: WASM CPU)`
    );
    runInferenceBtn.disabled = false;

    // Draw an initial sample scene on the canvas and run inference automatically
    drawDefaultSampleImage();
    await runInference();
  } catch (err) {
    console.error("[SAN-Web] Engine initialization error:", err);
    setStatus("error", "Initialization Failed", err.message || String(err));
  }
}

/**
 * Draws a default sample scene (mountain sunset landscape with sun and trees)
 * onto the preview and 224x224 canvas so the app works immediately on load.
 */
function drawDefaultSampleImage() {
  const ctx = inputCanvas.getContext("2d");
  // Sky gradient (sunset orange/pink/purple)
  const sky = ctx.createLinearGradient(0, 0, 0, 150);
  sky.addColorStop(0, "#1e1b4b");
  sky.addColorStop(0.5, "#be185d");
  sky.addColorStop(1, "#f97316");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, INPUT_SIZE, 150);

  // Sun
  ctx.fillStyle = "#fde047";
  ctx.beginPath();
  ctx.arc(112, 105, 26, 0, Math.PI * 2);
  ctx.fill();

  // Mountains
  ctx.fillStyle = "#312e81";
  ctx.beginPath();
  ctx.moveTo(0, 150);
  ctx.lineTo(65, 85);
  ctx.lineTo(135, 150);
  ctx.fill();

  ctx.fillStyle = "#1e1b4b";
  ctx.beginPath();
  ctx.moveTo(85, 150);
  ctx.lineTo(165, 75);
  ctx.lineTo(224, 150);
  ctx.fill();

  // Foreground lake / grass
  const lake = ctx.createLinearGradient(0, 150, 0, 224);
  lake.addColorStop(0, "#0f172a");
  lake.addColorStop(1, "#064e3b");
  ctx.fillStyle = lake;
  ctx.fillRect(0, 150, INPUT_SIZE, 74);

  previewImage.src = inputCanvas.toDataURL("image/png");
}

/**
 * Center-crops and resizes an HTMLImageElement onto the 224x224 canvas,
 * then extracts a Float32Array in NCHW RGB [0.0, 1.0] format.
 */
function getTensorFromCanvas() {
  const ctx = inputCanvas.getContext("2d", { willReadFrequently: true });
  const imgData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const numPixels = INPUT_SIZE * INPUT_SIZE;
  const floatData = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; i++) {
    const rgbaIdx = i * 4;
    floatData[i] = imgData[rgbaIdx] / 255.0;                 // R plane
    floatData[numPixels + i] = imgData[rgbaIdx + 1] / 255.0; // G plane
    floatData[2 * numPixels + i] = imgData[rgbaIdx + 2] / 255.0; // B plane
  }

  return new ort.Tensor("float32", floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function drawImageCenterCropped(img) {
  const ctx = inputCanvas.getContext("2d");
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;

  // Preserve full context for moderately wide/tall images while avoiding extreme distortion
  const ratio = Math.max(w, h) / Math.max(1, Math.min(w, h));
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  if (ratio <= 1.45) {
    ctx.drawImage(img, 0, 0, w, h, 0, 0, INPUT_SIZE, INPUT_SIZE);
  } else {
    const cropW = Math.min(w, Math.round(h * 1.35));
    const cropH = Math.min(h, Math.round(w * 1.35));
    const sx = Math.max(0, Math.floor((w - cropW) / 2));
    const sy = Math.max(0, Math.floor((h - cropH) / 2));
    ctx.drawImage(img, sx, sy, cropW, cropH, 0, 0, INPUT_SIZE, INPUT_SIZE);
  }
}

/**
 * Executes forward pass on CPU WASM and updates tag list UI.
 */
async function runInference() {
  if (!ortSession) return;
  runInferenceBtn.disabled = true;
  runInferenceBtn.textContent = "Running CPU Inference...";

  try {
    const inputTensor = getTensorFromCanvas();
    const t0 = performance.now();
    const results = await ortSession.run({ input: inputTensor });
    const elapsedMs = (performance.now() - t0).toFixed(1);

    lastScores = results.scores.data;
    latencyPill.textContent = `Latency: ${elapsedMs} ms`;
    renderPredictions();
  } catch (err) {
    console.error("[SAN-Web] Inference error:", err);
    tagsContainer.innerHTML = `<div class="empty-state">Error running inference: ${err.message}</div>`;
  } finally {
    runInferenceBtn.disabled = false;
    runInferenceBtn.textContent = "Run ONNX CPU Inference";
  }
}

function renderPredictions() {
  if (!lastScores) return;

  const threshold = parseFloat(thresholdSlider.value) / 100.0;
  const topK = parseInt(topKSlider.value, 10);

  const candidates = [];
  for (let i = 0; i < lastScores.length; i++) {
    const s = lastScores[i];
    if (s >= threshold) {
      candidates.push({ index: i, label: vocabulary[i] || `tag_${i}`, score: s });
    }
  }

  // Fallback if threshold is set higher than top tags
  if (candidates.length === 0) {
    for (let i = 0; i < lastScores.length; i++) {
      if (lastScores[i] >= 0.20) {
        candidates.push({ index: i, label: vocabulary[i] || `tag_${i}`, score: lastScores[i] });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  const selected = candidates.slice(0, topK);

  tagCountPill.textContent = `Detected: ${selected.length} tags`;

  if (selected.length === 0) {
    tagsContainer.innerHTML = `<div class="empty-state">No tags above threshold. Try lowering the confidence slider.</div>`;
    return;
  }

  tagsContainer.innerHTML = selected
    .map((item) => {
      const pct = (item.score * 100).toFixed(1);
      return `
        <div class="tag-row">
          <div class="tag-row-top">
            <div>
              <span class="tag-name">${item.label}</span>
              <span class="tag-id">#${item.index}</span>
            </div>
            <span class="tag-score">${pct}%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" style="width: ${Math.min(100, Math.max(2, pct))}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

// Event Listeners
function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      previewImage.src = img.src;
      drawImageCenterCropped(img);
      runInference();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

thresholdSlider.addEventListener("input", () => {
  thresholdVal.textContent = `${thresholdSlider.value}%`;
  renderPredictions();
});

topKSlider.addEventListener("input", () => {
  topKVal.textContent = topKSlider.value;
  renderPredictions();
});

runInferenceBtn.addEventListener("click", () => runInference());

window.addEventListener("DOMContentLoaded", initEngine);

