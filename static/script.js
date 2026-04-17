const videoElement = document.getElementById('hiddenVideo');
const canvasElement = document.getElementById('hiddenCanvas');
const processedImage = document.getElementById('videoFrame');
const statusText = document.getElementById('status');

// Define connection objects
let ws;
let isStreaming = false;

// --- Voice Assistant Implementation ---
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.continuous = true;
recognition.interimResults = false;
recognition.lang = 'de-DE';

let voiceState = 'IDLE'; // IDLE -> ASKING_IS_NEW -> ASKING_NAME -> CONFIRMING_START -> COUNTDOWN -> RECORDING
let patientName = '';
let isNewPatient = false;

const fmsContainer = document.getElementById('fms-total-container');
const voiceContainer = document.getElementById('voice-assistant-container');
const voiceStatus = document.getElementById('voice-status');
const voiceCountdown = document.getElementById('voice-countdown');

function updateVoiceUI(statusText, showCountdown = false) {
  // Hide the FMS score dynamically and show voice assistant
  if (fmsContainer) fmsContainer.style.display = 'none';
  if (voiceContainer) voiceContainer.style.display = 'block';
  if (statusText) voiceStatus.textContent = statusText;
  if (voiceCountdown) voiceCountdown.style.display = showCountdown ? 'block' : 'none';
}

// Timer helper for countdowns since we removed audio TTS
function waitAndCount(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

recognition.onstart = function () {
  console.log("Voice Assistant listening...");
  updateVoiceUI("Warte auf 'Sitzung starten'...", false);
};

recognition.onresult = async function (event) {
  const transcript = event.results[event.results.length - 1][0].transcript.trim().toLowerCase();
  console.log("Voice Input heard:", transcript);

  if (transcript.includes("abbrechen")) {
    voiceState = 'IDLE';
    updateVoiceUI("Abgebrochen. Warte auf 'Sitzung starten'...");
    return;
  }

  if (voiceState === 'IDLE' && transcript.includes("sitzung starten")) {
    voiceState = 'ASKING_IS_NEW';
    updateVoiceUI("Bist du neu hier? (Antworte 'Ja' oder 'Nein')");

  } else if (voiceState === 'ASKING_IS_NEW') {
    if (transcript.includes("ja")) {
      isNewPatient = true;
      voiceState = 'ASKING_NAME';
      updateVoiceUI("Wie ist dein Vor- und Nachname?");
    } else if (transcript.includes("nein")) {
      isNewPatient = false;
      voiceState = 'ASKING_NAME';
      updateVoiceUI("Wie ist dein Vor- und Nachname?");
    }
  } else if (voiceState === 'ASKING_NAME') {
    patientName = transcript;
    voiceState = 'CONFIRMING_START';
    updateVoiceUI(`Name: ${patientName}. Soll ich starten?`);

  } else if (voiceState === 'CONFIRMING_START' && transcript.includes("starten")) {
    voiceState = 'COUNTDOWN';
    updateVoiceUI("Mach dich bereit!", true);

    // Perform manual UI countdown
    voiceCountdown.textContent = "3";
    await waitAndCount(1000);
    voiceCountdown.textContent = "2";
    await waitAndCount(1000);
    voiceCountdown.textContent = "1";
    await waitAndCount(1000);

    // Start Recording
    voiceState = 'RECORDING';
    updateVoiceUI(`Aufnahme läuft für ${patientName}... Sag 'Stopp' zum Beenden.`, false);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        command: 'start_recording',
        name: patientName,
        is_new_patient: isNewPatient
      }));
    }
  } else if (voiceState === 'RECORDING' && transcript.includes("stopp")) {
    voiceState = 'IDLE';
    updateVoiceUI("Aufnahme beendet. Warte auf 'Sitzung starten'...");
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ command: 'stop_recording' }));
    }
  }
};

recognition.onend = function () {
  console.log("Recognition ended, restarting...");
  try { recognition.start(); } catch (e) { }
};
recognition.onerror = function (event) {
  console.error("Speech recognition error:", event.error);
};

// Attempt to start speech recognition when user interacts, or initially
try { recognition.start(); } catch (e) { console.warn(e); }

// --- WebSocket & Camera Implementation ---
async function initCamera() {
  try {
    // Request access to the webcam
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });

    videoElement.srcObject = stream;
    if (statusText) {
      statusText.textContent = "Camera accessed, connecting to WebSocket...";
      statusText.style.color = "blue";
    }

    // Once the video starts playing, we initialize the websocket
    videoElement.addEventListener('play', () => {
      initWebSocket();
    });
  } catch (err) {
    console.error("Error accessing the camera:", err);
    if (statusText) {
      statusText.textContent = "Error: Could not access camera. Please allow permissions.";
      statusText.style.color = "red";
    }
  }
}

function initWebSocket() {
  // Connect to FastAPI WebSocket endpoint
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    if (statusText) {
      statusText.textContent = "Connected and streaming!";
      statusText.style.color = "green";
    }
    isStreaming = true;

    // Match canvas to video size
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    // Start sending frames by sending the first one
    sendFrame();
  };

  ws.onmessage = (event) => {
    // Display the processed image and UI metric updates sent from backend
    const data = event.data;
    if (data.startsWith("{")) {
      try {
        const parsed = JSON.parse(data);
        if (parsed.image) processedImage.src = parsed.image;

        // Update left knee angle UI
        if (parsed.kf_l !== null) {
          const el = document.getElementById('kf-l');
          if (el) el.innerHTML = `${parsed.kf_l}<span class="val-unit">°</span>`;
        } else {
          const el = document.getElementById('kf-l');
          if (el) el.innerHTML = `--<span class="val-unit">°</span>`;
        }

        // Update right knee angle UI
        if (parsed.kf_r !== null) {
          const el = document.getElementById('kf-r');
          if (el) el.innerHTML = `${parsed.kf_r}<span class="val-unit">°</span>`;
        } else {
          const el = document.getElementById('kf-r');
          if (el) el.innerHTML = `--<span class="val-unit">°</span>`;
        }

      } catch (e) {
        console.error("Error parsing websocket message", e);
      }
    } else if (data.startsWith("data:image")) {
      processedImage.src = data;
    } else if (!data.startsWith("error")) {
      if (statusText) statusText.textContent = event.data;
    }

    // CRITICAL FIX: Backpressure implementation
    if (isStreaming) {
      requestAnimationFrame(sendFrame);
    }
  };

  ws.onerror = (err) => {
    console.error("WebSocket Error:", err);
    if (statusText) {
      statusText.textContent = "WebSocket connection error.";
      statusText.style.color = "red";
    }
    isStreaming = false;
  };

  ws.onclose = () => {
    if (statusText) {
      statusText.textContent = "Server disconnected. Please restart.";
      statusText.style.color = "red";
    }
    isStreaming = false;
  };
}

function sendFrame() {
  if (!isStreaming || ws.readyState !== WebSocket.OPEN) {
    return;
  }

  // Draw current video frame onto canvas
  const ctx = canvasElement.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

  // Convert canvas to base64 jpeg string
  const imageData = canvasElement.toDataURL('image/jpeg', 0.5);

  // Send base64 image data to the server
  ws.send(imageData);
}

// Start initialization when the page loads
window.addEventListener('load', initCamera);
