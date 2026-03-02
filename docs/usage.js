
const fileInput    = document.getElementById('fileInput');
const fileList     = document.getElementById('fileList');
const trainBtn     = document.getElementById('trainBtn');
const stopBtn      = document.getElementById('stopBtn');
const status       = document.getElementById('status');
const canvas       = document.getElementById('lossCanvas');
const samplesPanel = document.getElementById('samplesPanel');

let docs = [];
let worker = null;

// --- File upload ---

fileInput.addEventListener('change', async () => {
  const files = [...fileInput.files];
  if (files.length === 0) return;

  docs = [];
  const names = [];

  for (const file of files) {
    const text = await file.text();
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    docs.push(...lines);
    names.push(file.name);
  }

  fileList.textContent = `${names.join(', ')} — ${docs.length} documents`;
  trainBtn.disabled = false;
});

// --- Loss chart ---

const lossPoints = [];

function drawChart() {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const pad = { top: 20, right: 20, bottom: 40, left: 55 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  if (lossPoints.length === 0) return;

  const xs = lossPoints.map(p => p.step);
  const ys = lossPoints.map(p => p.loss);
  const xMin = Math.min(...xs), xMax = Math.max(...xs) || 1;
  const yMin = 0;
  const yMax = Math.max(...ys) * 1.05 || 1;

  function toX(step) { return pad.left + (step - xMin) / (xMax - xMin || 1) * plotW; }
  function toY(loss) { return pad.top  + (1 - (loss - yMin) / (yMax - yMin || 1)) * plotH; }

  // axes
  ctx.strokeStyle = '#999';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // axis labels
  ctx.fillStyle = '#666';
  ctx.font = '12px Fira Sans, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Step', pad.left + plotW / 2, H - 5);

  ctx.save();
  ctx.translate(14, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Loss', 0, 0);
  ctx.restore();

  // tick labels
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const xTicks = 5;
  for (let i = 0; i <= xTicks; i++) {
    const v = xMin + (xMax - xMin) * i / xTicks;
    ctx.fillText(Math.round(v).toString(), toX(v), pad.top + plotH + 6);
  }

  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + (yMax - yMin) * i / yTicks;
    ctx.fillText(v.toFixed(2), pad.left - 6, toY(v));
  }

  // loss line
  ctx.strokeStyle = '#3a6ea5';
  ctx.lineWidth = 2;
  ctx.beginPath();
  lossPoints.forEach((p, i) => {
    const x = toX(p.step), y = toY(p.loss);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // dots (hide when too dense)
  if (lossPoints.length < 900) {
    ctx.fillStyle = '#3a6ea5';
    lossPoints.forEach(p => {
      ctx.beginPath();
      ctx.arc(toX(p.step), toY(p.loss), 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // rolling average line (last 10 steps)
  if (lossPoints.length > 1) {
    ctx.strokeStyle = '#c03030';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 1; i < lossPoints.length; i++) {
      const window = lossPoints.slice(Math.max(0, i - 9), i + 1);
      const avg = window.reduce((s, p) => s + p.loss, 0) / window.length;
      const x = toX(lossPoints[i].step), y = toY(avg);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

// --- Training ---

function setTraining(active) {
  trainBtn.disabled = active;
  stopBtn.disabled = !active;
}

trainBtn.addEventListener('click', () => {
  setTraining(true);
  lossPoints.length = 0;
  samplesPanel.innerHTML = '<em>Training...</em>';
  status.textContent = 'Starting training...';

  const config = {
    docs,
    numSteps:       parseInt(document.getElementById('cfgSteps').value, 10),
    commentNthStep: parseInt(document.getElementById('cfgCommentNth').value, 10),
    sampleCount:    parseInt(document.getElementById('cfgSampleCount').value, 10),
    nLayer:         parseInt(document.getElementById('cfgNLayer').value, 10),
    nEmbd:          parseInt(document.getElementById('cfgNEmbd').value, 10),
    blockSize:      parseInt(document.getElementById('cfgBlockSize').value, 10),
    nHead:          parseInt(document.getElementById('cfgNHead').value, 10),
    learningRate:   parseFloat(document.getElementById('cfgLR').value),
    temperature:    parseFloat(document.getElementById('cfgTemp').value),
  };

  worker = new Worker('usage-worker.js', { type: 'module' });

  worker.onmessage = (e) => {
    const msg = e.data;

    if (msg.type === 'step') {
      status.textContent = `Step ${msg.step} / ${msg.numSteps} | Loss: ${msg.loss.toFixed(4)}`;
      lossPoints.push({ step: msg.step, loss: msg.loss });
      drawChart();

      if (msg.samples.length > 0) {
        samplesPanel.innerHTML = '';
        const heading = document.createElement('strong');
        heading.textContent = `Samples at step ${msg.step}:`;
        samplesPanel.appendChild(heading);
        for (const s of msg.samples) {
          const div = document.createElement('div');
          div.className = 'sample';
          div.textContent = s;
          samplesPanel.appendChild(div);
        }
      }
    }

    if (msg.type === 'done') {
      status.textContent = `Training complete — ${config.numSteps} steps.`;
      setTraining(false);
      worker = null;
    }

    if (msg.type === 'error') {
      status.textContent = `Error: ${msg.message}`;
      setTraining(false);
      worker = null;
    }
  };

  worker.postMessage({ type: 'train', config });
});

stopBtn.addEventListener('click', () => {
  if (worker) {
    worker.postMessage({ type: 'stop' });
    status.textContent += ' — Stopping...';
    // If the worker doesn't stop promptly (stuck in a long step), terminate it
    setTimeout(() => {
      if (worker) {
        worker.terminate();
        worker = null;
        status.textContent = status.textContent.replace(' — Stopping...', '') + ' — Stopped.';
        setTraining(false);
      }
    }, 3000);
  }
});
