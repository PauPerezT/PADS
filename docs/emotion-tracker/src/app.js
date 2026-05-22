const STEP_SECONDS = 0.5;
const DEFAULT_SAMPLE_RATE = 8000;

const PRESETS = {
  baseline: {
    name: "Interview baseline",
    badge: "Baseline",
    duration: 24,
    seed: 13,
    summary: "Balanced energy and stable delivery."
  },
  upbeat: {
    name: "Energetic call",
    badge: "Active",
    duration: 26,
    seed: 29,
    summary: "Bright tone with high activation."
  },
  calm: {
    name: "Calm reflection",
    badge: "Calm",
    duration: 28,
    seed: 41,
    summary: "Warm delivery with low activation."
  },
  stress: {
    name: "Escalating stress",
    badge: "Rising",
    duration: 25,
    seed: 67,
    summary: "Tension and activity increase over time."
  }
};

const state = {
  sourceName: "",
  sourceSummary: "",
  samples: new Float32Array(),
  sampleRate: DEFAULT_SAMPLE_RATE,
  rawFrames: [],
  frames: [],
  cursorIndex: 0,
  windowSeconds: 4,
  playing: false,
  playTimer: 0,
  baseline: null,
  recorder: null,
  recorderChunks: [],
  recordingStream: null
};

const el = {
  file: document.querySelector("#audio-file"),
  record: document.querySelector("#record-button"),
  play: document.querySelector("#play-button"),
  presets: [...document.querySelectorAll(".preset")],
  sourceBadge: document.querySelector("#source-badge"),
  timeSlider: document.querySelector("#time-slider"),
  timeReadout: document.querySelector("#time-readout"),
  durationReadout: document.querySelector("#duration-readout"),
  windowLength: document.querySelector("#window-length"),
  smoothing: document.querySelector("#smoothing"),
  baselineName: document.querySelector("#baseline-name"),
  baselineValence: document.querySelector("#baseline-valence"),
  baselineArousal: document.querySelector("#baseline-arousal"),
  baselineDominance: document.querySelector("#baseline-dominance"),
  setBaseline: document.querySelector("#set-baseline"),
  face: document.querySelector("#emotion-face"),
  emotionLabel: document.querySelector("#emotion-label"),
  emotionSummary: document.querySelector("#emotion-summary"),
  valenceValue: document.querySelector("#valence-value"),
  valenceBar: document.querySelector("#valence-bar"),
  valenceSub: document.querySelector("#valence-sub"),
  arousalValue: document.querySelector("#arousal-value"),
  arousalBar: document.querySelector("#arousal-bar"),
  arousalSub: document.querySelector("#arousal-sub"),
  dominanceValue: document.querySelector("#dominance-value"),
  dominanceBar: document.querySelector("#dominance-bar"),
  dominanceSub: document.querySelector("#dominance-sub"),
  dominanceReadout: document.querySelector("#dominance-readout"),
  deltaReadout: document.querySelector("#delta-readout"),
  windowReadout: document.querySelector("#window-readout"),
  table: document.querySelector("#window-table"),
  waveform: document.querySelector("#waveform-canvas"),
  pad: document.querySelector("#pad-canvas"),
  radar: document.querySelector("#radar-canvas"),
  timeline: document.querySelector("#timeline-canvas"),
  bars: document.querySelector("#bars-canvas")
};

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function percent(value) {
  return `${Math.round(clamp(value) * 100)}%`;
}

function seconds(value) {
  return `${value.toFixed(1)}s`;
}

function makeRandom(seed) {
  let current = seed;
  return () => {
    current = (current * 1664525 + 1013904223) % 4294967296;
    return current / 4294967296;
  };
}

function curveForPreset(key, time, duration) {
  const p = clamp(time / duration);
  const wave = Math.sin(Math.PI * 2 * p);
  const faster = Math.sin(Math.PI * 8 * p + 0.4);

  if (key === "upbeat") {
    return {
      valence: clamp(0.72 + 0.1 * wave + 0.04 * faster),
      arousal: clamp(0.66 + 0.17 * Math.sin(Math.PI * 5 * p)),
      dominance: clamp(0.66 + 0.08 * Math.cos(Math.PI * 3 * p)),
      amp: 0.22 + 0.24 * (0.6 + 0.4 * Math.sin(Math.PI * 5 * p))
    };
  }

  if (key === "calm") {
    return {
      valence: clamp(0.64 + 0.06 * Math.cos(Math.PI * 2 * p)),
      arousal: clamp(0.28 + 0.05 * Math.sin(Math.PI * 4 * p)),
      dominance: clamp(0.56 + 0.04 * Math.sin(Math.PI * 3 * p)),
      amp: 0.11 + 0.08 * (0.5 + 0.5 * wave)
    };
  }

  if (key === "stress") {
    return {
      valence: clamp(0.44 - 0.16 * p + 0.06 * faster),
      arousal: clamp(0.48 + 0.34 * p + 0.07 * Math.sin(Math.PI * 7 * p)),
      dominance: clamp(0.47 - 0.16 * p + 0.05 * Math.cos(Math.PI * 4 * p)),
      amp: 0.14 + 0.32 * p + 0.07 * Math.max(0, faster)
    };
  }

  return {
    valence: clamp(0.55 + 0.06 * wave + 0.03 * faster),
    arousal: clamp(0.39 + 0.08 * Math.sin(Math.PI * 4 * p + 0.3)),
    dominance: clamp(0.52 + 0.05 * Math.cos(Math.PI * 3 * p)),
    amp: 0.13 + 0.1 * (0.5 + 0.5 * Math.sin(Math.PI * 4 * p))
  };
}

function generatePreset(key) {
  const preset = PRESETS[key];
  const sampleRate = DEFAULT_SAMPLE_RATE;
  const sampleCount = Math.floor(preset.duration * sampleRate);
  const samples = new Float32Array(sampleCount);
  const random = makeRandom(preset.seed);

  for (let i = 0; i < sampleCount; i += 1) {
    const time = i / sampleRate;
    const curve = curveForPreset(key, time, preset.duration);
    const carrier = Math.sin(Math.PI * 2 * (122 + curve.arousal * 130) * time);
    const harmonic = Math.sin(Math.PI * 2 * (244 + curve.valence * 75) * time + 0.9);
    const texture = (random() - 0.5) * (0.035 + curve.arousal * 0.025);
    samples[i] = clamp((carrier * 0.66 + harmonic * 0.22 + texture) * curve.amp, -1, 1);
  }

  const frames = [];
  for (let time = 0; time < preset.duration; time += STEP_SECONDS) {
    const curve = curveForPreset(key, time, preset.duration);
    frames.push({
      time,
      valence: curve.valence,
      arousal: curve.arousal,
      dominance: curve.dominance,
      energy: curve.amp
    });
  }

  return {
    samples,
    sampleRate,
    rawFrames: frames,
    sourceName: preset.name,
    sourceSummary: preset.summary,
    badge: preset.badge
  };
}

function smoothFrames(frames) {
  const amount = Number(el.smoothing.value) / 100;
  if (!frames.length || amount <= 0) {
    return frames.map((frame) => ({ ...frame }));
  }

  const smoothed = [];
  let previous = { ...frames[0] };

  for (const frame of frames) {
    const next = {
      ...frame,
      valence: previous.valence * amount + frame.valence * (1 - amount),
      arousal: previous.arousal * amount + frame.arousal * (1 - amount),
      dominance: previous.dominance * amount + frame.dominance * (1 - amount)
    };
    smoothed.push(next);
    previous = next;
  }

  return smoothed;
}

function setData(payload, options = {}) {
  state.samples = payload.samples;
  state.sampleRate = payload.sampleRate;
  state.rawFrames = payload.rawFrames;
  state.frames = smoothFrames(state.rawFrames);
  state.cursorIndex = 0;
  state.sourceName = payload.sourceName;
  state.sourceSummary = payload.sourceSummary;
  el.sourceBadge.textContent = payload.badge || "Audio";
  el.timeSlider.max = Math.max(0, state.frames.length - 1);
  el.timeSlider.value = "0";

  if (options.setBaseline || !state.baseline) {
    setBaselineFromSummary(summarizeFrames(state.frames), state.sourceName);
  }

  render();
}

function setBaselineFromSummary(summary, name) {
  state.baseline = { ...summary, name };
  el.baselineName.textContent = name;
  el.baselineValence.textContent = percent(summary.valence);
  el.baselineArousal.textContent = percent(summary.arousal);
  el.baselineDominance.textContent = percent(summary.dominance);
}

function summarizeFrames(frames) {
  const initial = { valence: 0, arousal: 0, dominance: 0, energy: 0 };
  if (!frames.length) return initial;

  const totals = frames.reduce(
    (acc, frame) => {
      acc.valence += frame.valence;
      acc.arousal += frame.arousal;
      acc.dominance += frame.dominance;
      acc.energy += frame.energy || 0;
      return acc;
    },
    { ...initial }
  );

  return {
    valence: totals.valence / frames.length,
    arousal: totals.arousal / frames.length,
    dominance: totals.dominance / frames.length,
    energy: totals.energy / frames.length
  };
}

function getWindowRange() {
  const half = Math.max(1, Math.round(state.windowSeconds / STEP_SECONDS / 2));
  const start = Math.max(0, state.cursorIndex - half);
  const end = Math.min(state.frames.length - 1, state.cursorIndex + half);
  return [start, end];
}

function getWindowFrames() {
  const [start, end] = getWindowRange();
  return state.frames.slice(start, end + 1);
}

function currentSummary() {
  const frames = getWindowFrames();
  return summarizeFrames(frames.length ? frames : state.frames);
}

function classifyEmotion(summary) {
  const { valence, arousal, dominance } = summary;

  if (valence >= 0.62 && arousal >= 0.56) {
    return {
      label: dominance >= 0.58 ? "Confident Positive" : "Activated Positive",
      summary: "Positive tone with clear activation.",
      image: "./assets/happy.jpg",
      alt: "Happy expression"
    };
  }

  if (valence >= 0.58 && arousal < 0.48) {
    return {
      label: "Calm Positive",
      summary: "Warm tone with low activation.",
      image: "./assets/happy.jpg",
      alt: "Happy expression"
    };
  }

  if (valence <= 0.42 && arousal >= 0.58) {
    return {
      label: "Tense Active",
      summary: "Low pleasure with elevated activation.",
      image: "./assets/sad.jpg",
      alt: "Sad expression"
    };
  }

  if (valence <= 0.42 && arousal < 0.5) {
    return {
      label: "Low Withdrawn",
      summary: "Low pleasure with restrained energy.",
      image: "./assets/sad.jpg",
      alt: "Sad expression"
    };
  }

  return {
    label: "Neutral",
    summary: "Balanced energy and stable delivery.",
    image: "./assets/neutral.jpg",
    alt: "Neutral expression"
  };
}

function canvasContext(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(10, rect.width);
  const height = Number(canvas.getAttribute("height")) || rect.height || 220;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function drawGrid(ctx, width, height) {
  ctx.strokeStyle = "#e2e8e5";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const y = (height / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawWaveform() {
  const { ctx, width, height } = canvasContext(el.waveform);
  const samples = state.samples;
  const duration = samples.length / state.sampleRate || 0;
  const mid = height / 2;
  const amp = height * 0.38;
  drawGrid(ctx, width, height);

  ctx.strokeStyle = "#2d8a69";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  const stride = Math.max(1, Math.floor(samples.length / width));
  for (let x = 0; x < width; x += 1) {
    const start = x * stride;
    let min = 1;
    let max = -1;
    for (let j = 0; j < stride && start + j < samples.length; j += 1) {
      const sample = samples[start + j];
      min = Math.min(min, sample);
      max = Math.max(max, sample);
    }
    ctx.moveTo(x, mid + min * amp);
    ctx.lineTo(x, mid + max * amp);
  }
  ctx.stroke();

  if (state.frames.length) {
    const [start, end] = getWindowRange();
    const startX = (state.frames[start].time / duration) * width;
    const endX = (state.frames[end].time / duration) * width;
    const cursorX = (state.frames[state.cursorIndex].time / duration) * width;
    ctx.fillStyle = "rgba(63, 104, 212, 0.12)";
    ctx.fillRect(startX, 0, Math.max(2, endX - startX), height);
    ctx.strokeStyle = "#3f68d4";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cursorX, 0);
    ctx.lineTo(cursorX, height);
    ctx.stroke();
  }
}

function drawTimeline() {
  const { ctx, width, height } = canvasContext(el.timeline);
  const frames = state.frames;
  const pad = 28;
  drawGrid(ctx, width, height);

  if (frames.length < 2) return;

  const [start, end] = getWindowRange();
  const shadeX = pad + (start / (frames.length - 1)) * (width - pad * 2);
  const shadeW = ((end - start) / (frames.length - 1)) * (width - pad * 2);
  ctx.fillStyle = "rgba(45, 138, 105, 0.12)";
  ctx.fillRect(shadeX, 0, Math.max(2, shadeW), height);

  const series = [
    ["valence", "#2d8a69", "Valence"],
    ["arousal", "#cf5f4c", "Arousal"],
    ["dominance", "#3f68d4", "Dominance"]
  ];

  for (const [key, color] of series) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    frames.forEach((frame, index) => {
      const x = pad + (index / (frames.length - 1)) * (width - pad * 2);
      const y = height - pad - frame[key] * (height - pad * 2);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  series.forEach(([, color, label], index) => {
    const x = pad + index * 106;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, 18);
    ctx.lineTo(x + 18, 18);
    ctx.stroke();
    ctx.fillStyle = "#64716d";
    ctx.font = "700 12px Inter, sans-serif";
    ctx.fillText(label, x + 24, 22);
  });

  const cursorX = pad + (state.cursorIndex / (frames.length - 1)) * (width - pad * 2);
  ctx.strokeStyle = "#17201d";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, height);
  ctx.stroke();
}

function drawPadMap(summary) {
  const { ctx, width, height } = canvasContext(el.pad);
  const pad = 34;
  const plotW = width - pad * 2;
  const plotH = height - pad * 2;
  ctx.strokeStyle = "#d9e0dd";
  ctx.strokeRect(pad, pad, plotW, plotH);
  ctx.strokeStyle = "#b7c3be";
  ctx.beginPath();
  ctx.moveTo(width / 2, pad);
  ctx.lineTo(width / 2, height - pad);
  ctx.moveTo(pad, height / 2);
  ctx.lineTo(width - pad, height / 2);
  ctx.stroke();

  ctx.fillStyle = "#64716d";
  ctx.font = "700 12px Inter, sans-serif";
  ctx.fillText("Negative", pad, height - 10);
  ctx.fillText("Positive", width - pad - 50, height - 10);
  ctx.save();
  ctx.translate(14, height / 2 + 34);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Arousal", 0, 0);
  ctx.restore();

  const x = pad + summary.valence * plotW;
  const y = height - pad - summary.arousal * plotH;
  const radius = 11 + summary.dominance * 22;
  ctx.fillStyle = "rgba(63, 104, 212, 0.18)";
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#3f68d4";
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fill();
}

function drawRadar(summary) {
  const { ctx, width, height } = canvasContext(el.radar);
  const cx = width / 2;
  const cy = height / 2 + 4;
  const radius = Math.min(width, height) * 0.34;
  const values = [
    summary.arousal,
    1 - summary.arousal,
    summary.dominance,
    1 - summary.dominance,
    summary.valence,
    1 - summary.valence
  ];
  const labels = ["Active", "Passive", "Strong", "Weak", "Positive", "Negative"];

  for (let ring = 1; ring <= 4; ring += 1) {
    ctx.strokeStyle = ring === 4 ? "#b7c3be" : "#d9e0dd";
    ctx.beginPath();
    for (let i = 0; i < labels.length; i += 1) {
      const angle = -Math.PI / 2 + (Math.PI * 2 * i) / labels.length;
      const r = (radius * ring) / 4;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }

  labels.forEach((label, i) => {
    const angle = -Math.PI / 2 + (Math.PI * 2 * i) / labels.length;
    ctx.strokeStyle = "#e2e8e5";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    ctx.stroke();
    ctx.fillStyle = "#64716d";
    ctx.font = "700 11px Inter, sans-serif";
    ctx.textAlign = Math.cos(angle) > 0.25 ? "left" : Math.cos(angle) < -0.25 ? "right" : "center";
    ctx.fillText(label, cx + Math.cos(angle) * (radius + 18), cy + Math.sin(angle) * (radius + 18));
  });
  ctx.textAlign = "left";

  ctx.beginPath();
  values.forEach((value, i) => {
    const angle = -Math.PI / 2 + (Math.PI * 2 * i) / values.length;
    const x = cx + Math.cos(angle) * radius * value;
    const y = cy + Math.sin(angle) * radius * value;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "rgba(45, 138, 105, 0.22)";
  ctx.strokeStyle = "#2d8a69";
  ctx.lineWidth = 2;
  ctx.fill();
  ctx.stroke();
}

function drawBars(summary) {
  const { ctx, width, height } = canvasContext(el.bars);
  const baseline = state.baseline || summary;
  const labels = ["Valence", "Arousal", "Dominance"];
  const current = [summary.valence, summary.arousal, summary.dominance];
  const base = [baseline.valence, baseline.arousal, baseline.dominance];
  const colors = ["#2d8a69", "#cf5f4c", "#3f68d4"];
  const pad = 34;
  const plotH = height - pad * 2;
  const groupW = (width - pad * 2) / labels.length;

  drawGrid(ctx, width, height);
  ctx.font = "700 12px Inter, sans-serif";
  ctx.fillStyle = "#9aa6a2";
  ctx.fillRect(pad, 16, 14, 9);
  ctx.fillText("Baseline", pad + 20, 25);
  ctx.fillStyle = "#2d8a69";
  ctx.fillRect(pad + 100, 16, 14, 9);
  ctx.fillStyle = "#64716d";
  ctx.fillText("Current", pad + 120, 25);

  labels.forEach((label, i) => {
    const x = pad + i * groupW + groupW * 0.2;
    const barW = Math.max(18, groupW * 0.2);
    const baseH = base[i] * plotH;
    const currentH = current[i] * plotH;
    ctx.fillStyle = "#c9d1ce";
    ctx.fillRect(x, height - pad - baseH, barW, baseH);
    ctx.fillStyle = colors[i];
    ctx.fillRect(x + barW + 8, height - pad - currentH, barW, currentH);
    ctx.fillStyle = "#64716d";
    ctx.font = "700 12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, x + barW, height - 10);
  });
  ctx.textAlign = "left";
}

function renderTable() {
  const [start, end] = getWindowRange();
  const rows = [];
  for (let i = start; i <= end; i += 1) {
    const frame = state.frames[i];
    rows.push(`
      <tr class="${i === state.cursorIndex ? "current-row" : ""}">
        <td>${seconds(frame.time)}</td>
        <td>${percent(frame.valence)}</td>
        <td>${percent(frame.arousal)}</td>
        <td>${percent(frame.dominance)}</td>
      </tr>
    `);
  }
  el.table.innerHTML = rows.join("");
}

function render() {
  state.frames = smoothFrames(state.rawFrames);
  state.cursorIndex = Math.min(state.cursorIndex, Math.max(0, state.frames.length - 1));
  const summary = currentSummary();
  const emotion = classifyEmotion(summary);
  const duration = state.samples.length / state.sampleRate || 0;
  const activeFrame = state.frames[state.cursorIndex] || { time: 0 };
  const [start, end] = getWindowRange();
  const baseline = state.baseline || summary;
  const delta =
    (Math.abs(summary.valence - baseline.valence) +
      Math.abs(summary.arousal - baseline.arousal) +
      Math.abs(summary.dominance - baseline.dominance)) /
    3;

  el.timeReadout.textContent = seconds(activeFrame.time);
  el.durationReadout.textContent = seconds(duration);
  el.windowReadout.textContent = `${seconds(state.frames[start]?.time || 0)} - ${seconds(state.frames[end]?.time || 0)}`;
  el.dominanceReadout.textContent = `Dominance ${percent(summary.dominance)}`;
  el.deltaReadout.textContent = `Delta ${percent(delta)}`;
  el.emotionLabel.textContent = emotion.label;
  el.emotionSummary.textContent = state.sourceSummary || emotion.summary;
  el.face.src = emotion.image;
  el.face.alt = emotion.alt;

  el.valenceValue.textContent = percent(summary.valence);
  el.valenceSub.textContent = `Negative ${percent(1 - summary.valence)}`;
  el.valenceBar.style.width = percent(summary.valence);
  el.arousalValue.textContent = percent(summary.arousal);
  el.arousalSub.textContent = `Passive ${percent(1 - summary.arousal)}`;
  el.arousalBar.style.width = percent(summary.arousal);
  el.dominanceValue.textContent = percent(summary.dominance);
  el.dominanceSub.textContent = `Weak ${percent(1 - summary.dominance)}`;
  el.dominanceBar.style.width = percent(summary.dominance);

  drawWaveform();
  drawTimeline();
  drawPadMap(summary);
  drawRadar(summary);
  drawBars(summary);
  renderTable();
}

function analyzeAudioBuffer(buffer, label) {
  const sampleRate = buffer.sampleRate;
  const length = buffer.length;
  const samples = new Float32Array(length);
  const channelCount = Math.min(2, buffer.numberOfChannels);
  for (let channel = 0; channel < channelCount; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < length; i += 1) {
      samples[i] += data[i] / channelCount;
    }
  }

  const windowSize = Math.max(512, Math.floor(sampleRate * STEP_SECONDS));
  const frames = [];
  let previousRms = 0;

  for (let start = 0; start < length; start += windowSize) {
    const end = Math.min(length, start + windowSize);
    let sumSquares = 0;
    let peak = 0;
    let zeroCrossings = 0;
    let roughness = 0;

    for (let i = start; i < end; i += 1) {
      const sample = samples[i];
      sumSquares += sample * sample;
      peak = Math.max(peak, Math.abs(sample));
      if (i > start && Math.sign(sample) !== Math.sign(samples[i - 1])) zeroCrossings += 1;
      if (i > start) roughness += Math.abs(sample - samples[i - 1]);
    }

    const count = Math.max(1, end - start);
    const rms = Math.sqrt(sumSquares / count);
    const zcr = zeroCrossings / count;
    const rough = roughness / count;
    const energy = clamp(rms * 8.5);
    const zNorm = clamp(zcr * 35);
    const roughNorm = clamp(rough * 28);
    const crest = peak / (rms + 0.0001);
    const crestNorm = clamp((crest - 1.8) / 7);
    const stability = clamp(1 - Math.abs(roughNorm - energy) * 0.65 - crestNorm * 0.18);
    const change = clamp(Math.abs(rms - previousRms) * 12);

    frames.push({
      time: start / sampleRate,
      valence: clamp(0.5 + stability * 0.22 - roughNorm * 0.18 - crestNorm * 0.1 + (energy - 0.5) * 0.04),
      arousal: clamp(0.17 + energy * 0.58 + roughNorm * 0.22 + zNorm * 0.15 + change * 0.12),
      dominance: clamp(0.33 + energy * 0.28 + stability * 0.28 - zNorm * 0.1 - crestNorm * 0.05),
      energy
    });
    previousRms = rms;
  }

  return {
    samples,
    sampleRate,
    rawFrames: frames,
    sourceName: label,
    sourceSummary: "Client-side audio features mapped into the PADS space.",
    badge: "Audio"
  };
}

async function loadAudioFromFile(file) {
  if (!file) return;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContextClass();
  const buffer = await file.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(buffer);
  await audioContext.close();
  setData(analyzeAudioBuffer(audioBuffer, file.name));
  el.presets.forEach((button) => button.classList.remove("active"));
}

async function loadAudioFromBlob(blob) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContextClass();
  const buffer = await blob.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(buffer);
  await audioContext.close();
  setData(analyzeAudioBuffer(audioBuffer, "Recorded clip"));
}

async function toggleRecording() {
  if (state.recorder && state.recorder.state === "recording") {
    state.recorder.stop();
    el.record.textContent = "Record";
    return;
  }

  try {
    state.recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.recorderChunks = [];
    state.recorder = new MediaRecorder(state.recordingStream);
    state.recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) state.recorderChunks.push(event.data);
    });
    state.recorder.addEventListener("stop", async () => {
      state.recordingStream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(state.recorderChunks, { type: state.recorder.mimeType });
      await loadAudioFromBlob(blob);
    });
    state.recorder.start();
    el.record.textContent = "Stop";
  } catch (error) {
    el.sourceBadge.textContent = "Mic blocked";
    console.warn(error);
  }
}

function togglePlayback() {
  state.playing = !state.playing;
  el.play.textContent = state.playing ? "II" : ">";

  if (state.playing) {
    state.playTimer = window.setInterval(() => {
      state.cursorIndex += 1;
      if (state.cursorIndex >= state.frames.length) {
        state.cursorIndex = 0;
      }
      el.timeSlider.value = String(state.cursorIndex);
      render();
    }, STEP_SECONDS * 1000);
  } else {
    window.clearInterval(state.playTimer);
  }
}

function bindEvents() {
  el.presets.forEach((button) => {
    button.addEventListener("click", () => {
      el.presets.forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      setData(generatePreset(button.dataset.preset));
    });
  });

  el.file.addEventListener("change", (event) => {
    loadAudioFromFile(event.target.files[0]).catch((error) => {
      console.warn(error);
      el.sourceBadge.textContent = "Decode error";
    });
  });

  el.record.addEventListener("click", toggleRecording);
  el.play.addEventListener("click", togglePlayback);
  el.timeSlider.addEventListener("input", () => {
    state.cursorIndex = Number(el.timeSlider.value);
    render();
  });
  el.windowLength.addEventListener("change", () => {
    state.windowSeconds = Number(el.windowLength.value);
    render();
  });
  el.smoothing.addEventListener("input", render);
  el.setBaseline.addEventListener("click", () => {
    setBaselineFromSummary(currentSummary(), state.sourceName);
    render();
  });
  window.addEventListener("resize", render);
}

bindEvents();
setData(generatePreset("baseline"), { setBaseline: true });
