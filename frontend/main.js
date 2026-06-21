const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

// DOM Elements
const videoStream = document.getElementById('video-stream');
const connIndicator = document.getElementById('conn-status');
const connText = document.getElementById('conn-text');
const valFps = document.getElementById('val-fps');
const valDetections = document.getElementById('val-detections');
const valLatency = document.getElementById('val-latency');

const btnStart = document.getElementById('btn-start');
const inputSource = document.getElementById('input-source');
const checkRtsp = document.getElementById('check-rtsp');
const selectModel = document.getElementById('select-model');
const sliderConf = document.getElementById('slider-conf');
const valConf = document.getElementById('val-conf');
const sliderFps = document.getElementById('slider-fps');
const valTargetFps = document.getElementById('val-target-fps');

let ws = null;

// Connect WebSocket
function connectWebSocket() {
  if (ws) ws.close();
  
  ws = new WebSocket(`${WS_BASE}/ws/video`);
  
  ws.onopen = () => {
    connIndicator.classList.add('connected');
    connText.textContent = 'Connected';
  };
  
  ws.onclose = () => {
    connIndicator.classList.remove('connected');
    connText.textContent = 'Disconnected';
    setTimeout(connectWebSocket, 2000); // Reconnect loop
  };
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Update image
    videoStream.src = `data:image/jpeg;base64,${data.image}`;
    
    // Update metrics
    valFps.textContent = data.metrics.fps;
    valDetections.textContent = data.metrics.detections;
    valLatency.textContent = `${Math.round(data.metrics.latency_ms)}ms`;
  };
}

// Send config to backend
async function updateConfig(config) {
  try {
    await fetch(`${API_BASE}/api/config`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(config)
    });
  } catch (error) {
    console.error("Failed to update config:", error);
  }
}

// Event Listeners
btnStart.addEventListener('click', () => {
  const source = inputSource.value;
  if (!source) return alert("Please enter a video source");
  
  updateConfig({
    source: source,
    is_rtsp: checkRtsp.checked,
    model_size: selectModel.value,
    conf_threshold: parseFloat(sliderConf.value),
    target_fps: parseInt(sliderFps.value)
  });
});

selectModel.addEventListener('change', (e) => {
  updateConfig({ model_size: e.target.value });
});

sliderConf.addEventListener('input', (e) => {
  valConf.textContent = e.target.value;
});
sliderConf.addEventListener('change', (e) => {
  updateConfig({ conf_threshold: parseFloat(e.target.value) });
});

sliderFps.addEventListener('input', (e) => {
  valTargetFps.textContent = e.target.value;
});
sliderFps.addEventListener('change', (e) => {
  updateConfig({ target_fps: parseInt(e.target.value) });
});

// Initialize
connectWebSocket();

// Auto-start stream on load
setTimeout(() => {
  if (btnStart) btnStart.click();
}, 1000);
