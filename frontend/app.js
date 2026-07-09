/*
  AURA AI - Emotion-Based Music Recommendation System
  Frontend UI Shell & Integration Interface
*/

// ==========================================
// 1. INTEGRATION CONFIGURATION & API HOOKS
// ==========================================
// The backend and ML developers can change these values to connect their APIs.
const API_CONFIG = {
  serverBaseUrl: "http://127.0.0.1:8000/api",
  
  // Endpoint definitions
  endpoints: {
    analyzeFace: "/emotion/face",       // POST: webcam snapshot -> returns detected emotion
    analyzeVoice: "/emotion/voice",     // POST: mic audio blob -> returns detected emotion
    analyzeText: "/emotion/text",       // POST: journal text string -> returns detected emotion
    recommendations: "/music/recommend" // GET: fetch songs list filtered by emotion/acoustic metrics
  }
};
// ==========================================
// LOCAL PERSISTENCE (Liked / History / Top Played)
// ==========================================
function getLikedTracks() {
  return JSON.parse(localStorage.getItem("auraai_liked") || "[]");
}

function toggleLikedTrack(song) {
  let liked = getLikedTracks();
  const exists = liked.some(t => t.spotify_track_link === song.spotify_track_link);

  if (exists) {
    liked = liked.filter(t => t.spotify_track_link !== song.spotify_track_link);
  } else {
    liked.push(song);
  }

  localStorage.setItem("auraai_liked", JSON.stringify(liked));
}

function recordPlayHistory(song) {
  let history = JSON.parse(localStorage.getItem("auraai_history") || "[]");
  history.unshift({ ...song, playedAt: new Date().toISOString() });
  history = history.slice(0, 50);
  localStorage.setItem("auraai_history", JSON.stringify(history));

  let counts = JSON.parse(localStorage.getItem("auraai_playcounts") || "{}");
  counts[song.spotify_track_link] = (counts[song.spotify_track_link] || 0) + 1;
  localStorage.setItem("auraai_playcounts", JSON.stringify(counts));
}

function getTopPlayed() {
  const history = JSON.parse(localStorage.getItem("auraai_history") || "[]");
  const counts = JSON.parse(localStorage.getItem("auraai_playcounts") || "{}");

  const uniqueSongs = [];
  const seen = new Set();
  history.forEach(song => {
    if (!seen.has(song.spotify_track_link)) {
      seen.add(song.spotify_track_link);
      uniqueSongs.push(song);
    }
  });

  return uniqueSongs
    .sort((a, b) => (counts[b.spotify_track_link] || 0) - (counts[a.spotify_track_link] || 0))
    .slice(0, 10);
}

function renderLibrary() {
  renderTrackList("library-liked-container", getLikedTracks());
  renderTrackList("library-history-container", JSON.parse(localStorage.getItem("auraai_history") || "[]"));
  renderTrackList("library-top-played-container", getTopPlayed());
}

// Global App State
const state = {
  currentMood: "neutral",
  currentTrackList: [],
  currentTrackIndex: -1,
  isPlaying: false,
  isMuted: false,
  volume: 0.8,
  currentTime: 0,
  playbackTimer: null,
  
  // History, Liked & Top Played tracks (for integration)
  playHistory: [],
  topPlayed: [],
  likedTracks: [],
  
  // Devices state
  cameraStream: null,
  isScanning: false,
  cameraHudAnimationId: null,
  
  // Voice recording state
  audioContext: null,
  audioAnalyser: null,
  isRecordingVoice: false,
  
  // Dummy weekly history (for rendering the dashboard graph grid)
  weeklyHistory: [
    { day: "Mon", valence: 0.6, energy: 0.4 },
    { day: "Tue", valence: 0.7, energy: 0.2 },
    { day: "Wed", valence: 0.3, energy: 0.3 },
    { day: "Thu", valence: 0.8, energy: 0.7 },
    { day: "Fri", valence: 0.9, energy: 0.95 },
    { day: "Sat", valence: 0.75, energy: 0.3 },
    { day: "Sun", valence: 0.85, energy: 0.68 }
  ]
};

// ==========================================
// 2. CONNECTORS & INTEGRATION FUNCTIONS
// ==========================================

/*
  HOW TO INTEGRATE:
  When your backend API is ready:
  1. Uncomment the fetch requests inside the functions below.
  2. Map the response data arrays directly into 'state.currentTrackList' and render them.
  3. Fetch user history and top played data from your endpoints and append them to 'library-history-container' and 'library-top-played-container'.
*/



// Triggered when mic recording is complete
async function sendVoiceBlobToBackend(audioBlob) {
  try {
    const formData = new FormData();
    formData.append("file", audioBlob, "voice.wav");

    const response = await fetch(`${API_CONFIG.serverBaseUrl}${API_CONFIG.endpoints.analyzeVoice}`, {
      method: "POST",
      body: formData
    });
    const result = await response.json();
    return { success: result.success, detectedEmotion: result.detectedEmotion };
  } catch (e) {
    console.error("Voice integration error:", e);
    return { success: false };
  }
}

// Triggered when user finishes typing in journal
async function sendTextToBackend(textString) {
  try {
    const response = await fetch(`${API_CONFIG.serverBaseUrl}${API_CONFIG.endpoints.analyzeText}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: textString })
    });
    const result = await response.json();
    return { success: result.success, detectedEmotion: result.detectedEmotion };
  } catch (e) {
    console.error("Text integration error:", e);
    return { success: false };
  }
}

// Triggered to update recommendation lists based on mood state
async function fetchRecommendationsFromBackend(mood, valence, energy) {
  try {
    const response = await fetch(`${API_CONFIG.serverBaseUrl}${API_CONFIG.endpoints.recommendations}?mood=${mood}`);
    const songData = await response.json();

    state.currentTrackList = songData;
    renderTrackList("recommended-tracks-container", songData);
  } catch (e) {
    console.error("Recommendations integration error:", e);
  }
}
function renderTrackList(containerId, songs) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  if (!songs || songs.length === 0) {
    container.innerHTML = `<p style="color: var(--text-muted); padding: 16px;">No recommendations found.</p>`;
    return;
  }

  songs.forEach((song) => {
    const row = document.createElement("div");
    row.className = "track-row";
    row.style.cursor = "pointer";

    const initials = song.track_name ? song.track_name.substring(0, 2).toUpperCase() : "??";
    const isLiked = getLikedTracks().some(t => t.spotify_track_link === song.spotify_track_link);

    row.innerHTML = `
      <div class="track-cover" style="background: linear-gradient(135deg, var(--accent-color), var(--accent-pink));">${initials}</div>
      <div class="track-details">
        <div class="track-title">${song.track_name}</div>
        <div class="track-artist">${song.artists}</div>
      </div>
      <div class="track-mood-badge" style="background: rgba(99,102,241,0.15); color: var(--accent-color);">${song.mood}</div>
      <button class="control-btn btn-like-track" style="margin-left:8px;" data-liked="${isLiked}">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="${isLiked ? 'currentColor' : 'none'}">
          <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
        </svg>
      </button>
    `;

    row.addEventListener("click", (e) => {
      if (e.target.closest(".btn-like-track")) return;
      playSpotifyTrack(song);
    });

    row.querySelector(".btn-like-track").addEventListener("click", (e) => {
      e.stopPropagation();
      toggleLikedTrack(song);
      renderTrackList(containerId, songs);
    });

    container.appendChild(row);
  });
}
// ==========================================
// SEARCH (mood-based)
// ==========================================
const globalSearchInput = document.getElementById("global-search-input");
const VALID_MOODS = ["happy", "sad", "calm", "energetic", "motivated", "romantic"];

globalSearchInput.addEventListener("keypress", async (e) => {
  if (e.key !== "Enter") return;

  const query = globalSearchInput.value.trim().toLowerCase();
  if (!query) return;

  const matchedMood = VALID_MOODS.find(m => query.includes(m)) || "Happy";

  document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
  document.querySelectorAll(".content-section").forEach(s => s.classList.remove("active"));
  document.getElementById("section-search").classList.add("active");

  try {
    const response = await fetch(`${API_CONFIG.serverBaseUrl}${API_CONFIG.endpoints.recommendations}?mood=${matchedMood.charAt(0).toUpperCase() + matchedMood.slice(1)}`);
    const songData = await response.json();
    renderTrackList("search-results-container", songData);
  } catch (e) {
    console.error("Search error:", e);
  }
});

function playSpotifyTrack(song) {
  const spotifyContainer = document.getElementById("spotify-player-container");
  const spotifyIframe = document.getElementById("spotify-iframe");

  const trackId = song.spotify_track_link.split("/track/")[1];
  spotifyIframe.src = `https://open.spotify.com/embed/track/${trackId}`;

  spotifyContainer.style.display = "block";

  document.getElementById("current-player-title").textContent = song.track_name;
  document.getElementById("current-player-artist").textContent = song.artists;

  recordPlayHistory(song);
}

// ==========================================
// 3. UI TAB AND NAVIGATION ROUTING
// ==========================================
function initNavigation() {
  const links = document.querySelectorAll(".nav-link");
  const sections = document.querySelectorAll(".content-section");
  
  links.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const target = link.getAttribute("data-target");
      
      links.forEach(l => l.classList.remove("active"));
      link.classList.add("active");
      
      sections.forEach(sec => {
        sec.classList.remove("active");
        if (sec.id === `section-${target}`) {
          sec.classList.add("active");
          if (target === "library") {
          renderLibrary();
        }
        }
      });
    });
  });
}

// ==========================================
// 4. WEBCAM SCANNER & CANON BIOMETRIC HUD
// ==========================================
const webcamVideo = document.getElementById("webcam-feed");
const webcamPlaceholder = document.getElementById("webcam-placeholder");
const webcamHud = document.getElementById("webcam-hud");
const hudScanLine = document.getElementById("hud-scan-line");
const hudTargetBox = document.getElementById("hud-target-box");
const hudStats = document.getElementById("hud-stats-overlay");
const btnToggleCamera = document.getElementById("btn-toggle-camera");
const btnCaptureMood = document.getElementById("btn-capture-mood");
const webcamOverlayCanvas = document.getElementById("webcam-canvas-overlay");
const webcamOverlayCtx = webcamOverlayCanvas.getContext("2d");

async function toggleCamera() {
  if (state.cameraStream) {
    // Stop stream
    state.cameraStream.getTracks().forEach(track => track.stop());
    state.cameraStream = null;
    webcamVideo.srcObject = null;
    
    // Reset HUD
    webcamPlaceholder.style.display = "flex";
    webcamHud.classList.remove("active");
    hudScanLine.classList.remove("active");
    hudTargetBox.classList.remove("active");
    hudStats.textContent = "STATUS: STANDBY";
    
    btnToggleCamera.textContent = "Start Scanner";
    btnToggleCamera.className = "btn btn-secondary";
    btnCaptureMood.style.display = "none";
    
    // Stop canvas animation
    cancelAnimationFrame(state.cameraHudAnimationId);
    webcamOverlayCtx.clearRect(0, 0, webcamOverlayCanvas.width, webcamOverlayCanvas.height);
  } else {
    try {
      hudStats.textContent = "STATUS: INITIALIZING...";
      state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      webcamVideo.srcObject = state.cameraStream;
      
      webcamPlaceholder.style.display = "none";
      webcamHud.classList.add("active");
      hudStats.textContent = "STATUS: SCANNER LIVE";
      
      btnToggleCamera.textContent = "Stop Scanner";
      btnToggleCamera.className = "btn btn-danger";
      btnCaptureMood.style.display = "inline-flex";
      
      // Start real-time biometric HUD drawings
      webcamOverlayCanvas.width = webcamVideo.clientWidth || 320;
      webcamOverlayCanvas.height = webcamVideo.clientHeight || 240;
      animateBiometricHUD();
      
    } catch (err) {
      alert("Unable to access camera. Please check browser permissions.");
      hudStats.textContent = "STATUS: CAMERA ACCESS DENIED";
    }
  }
}

function animateBiometricHUD() {
  if (!state.cameraStream) return;
  state.cameraHudAnimationId = requestAnimationFrame(animateBiometricHUD);
  
  const ctx = webcamOverlayCtx;
  const w = webcamOverlayCanvas.width;
  const h = webcamOverlayCanvas.height;
  ctx.clearRect(0, 0, w, h);
  
  // Set glow shadow styles
  ctx.shadowBlur = 10;
  ctx.shadowColor = "rgba(99, 102, 241, 0.4)";
  ctx.strokeStyle = "rgba(99, 102, 241, 0.5)";
  ctx.lineWidth = 1.5;
  
  const rectSize = Math.min(w, h) * 0.5;
  const rx = (w - rectSize) / 2;
  const ry = (h - rectSize) / 2;
  
  // Draw HUD brackets corner paths
  const len = 15;
  // Top Left
  ctx.beginPath(); ctx.moveTo(rx, ry + len); ctx.lineTo(rx, ry); ctx.lineTo(rx + len, ry); ctx.stroke();
  // Top Right
  ctx.beginPath(); ctx.moveTo(rx + rectSize - len, ry); ctx.lineTo(rx + rectSize, ry); ctx.lineTo(rx + rectSize, ry + len); ctx.stroke();
  // Bottom Left
  ctx.beginPath(); ctx.moveTo(rx, ry + rectSize - len); ctx.lineTo(rx, ry + rectSize); ctx.lineTo(rx + len, ry + rectSize); ctx.stroke();
  // Bottom Right
  ctx.beginPath(); ctx.moveTo(rx + rectSize - len, ry + rectSize); ctx.lineTo(rx + rectSize, ry + rectSize); ctx.lineTo(rx + rectSize, ry + rectSize - len); ctx.stroke();
  
  // Draw rotating circular tracker in the center
  const cx = w / 2;
  const cy = h / 2;
  const time = Date.now() * 0.001;
  
  ctx.strokeStyle = "rgba(99, 102, 241, 0.25)";
  ctx.beginPath();
  ctx.arc(cx, cy, 35, time, time + Math.PI * 0.5);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, 35, time + Math.PI, time + Math.PI * 1.5);
  ctx.stroke();
  
  // Draw animated dots lock points inside brackets
  const pulse = Math.abs(Math.sin(Date.now() * 0.003));
  ctx.fillStyle = `rgba(16, 185, 129, ${0.3 + pulse * 0.5})`;
  ctx.beginPath();
  ctx.arc(rx + rectSize * 0.3, ry + rectSize * 0.45, 2.5, 0, Math.PI * 2); // Left Eye Lock
  ctx.arc(rx + rectSize * 0.7, ry + rectSize * 0.45, 2.5, 0, Math.PI * 2); // Right Eye Lock
  ctx.arc(cx, cy + rectSize * 0.15, 2.5, 0, Math.PI * 2); // Mouth Lock
  ctx.fill();
  
  // Draw fake biometric stats text
  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(99, 102, 241, 0.7)";
  ctx.font = "8px monospace";
  ctx.fillText("FACIAL_LOCK: ON", rx + 6, ry + 12);
  ctx.fillText("RESOLUTION: 640x480", rx + 6, ry + 22);
  ctx.fillText(`FPS: ${28 + Math.floor(Math.random()*3)}`, rx + rectSize - 54, ry + 12);
  ctx.fillText("POINTS: 68_LM", rx + rectSize - 54, ry + 22);
}

async function captureAndScanFace() {
  if (!state.cameraStream) return;
  
  hudScanLine.classList.add("active");
  hudTargetBox.classList.add("active");
  hudStats.textContent = "STATUS: ANALYZING BIOMETRICS...";
  btnCaptureMood.disabled = true;
  
  // Capture a snapshot frame
  const canvas = document.createElement("canvas");
  canvas.width = webcamVideo.videoWidth;
  canvas.height = webcamVideo.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
  const frameData = canvas.toDataURL("image/png");
  
  const result = await sendFaceFrameToBackend(frameData);
  
  setTimeout(() => {
    hudScanLine.classList.remove("active");
    hudTargetBox.classList.remove("active");
    btnCaptureMood.disabled = false;
    
    if (result && result.success) {
      hudStats.textContent = `STATUS: EMOTION DETECTED (${result.detectedEmotion.toUpperCase()})`;
      updateSystemMood(result.detectedEmotion);
    } else {
      hudStats.textContent = "STATUS: SCAN ERROR";
    }
  }, 1200);
}

async function sendFaceFrameToBackend(base64Frame) {
  try {
    const response = await fetch(`${API_CONFIG.serverBaseUrl}${API_CONFIG.endpoints.analyzeFace}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64Frame })
    });
    const result = await response.json();
    return { success: result.success, detectedEmotion: result.detectedEmotion };
  } catch (e) {
    console.error("Face scan integration error:", e);
    return { success: false };
  }
}

// ==========================================
// 5. VOICE RECORDER & GLOWING EQUALIZER
// ==========================================
const btnRecordVoice = document.getElementById("btn-record-voice");
const voiceStatusText = document.getElementById("voice-status-text");
const voiceCanvas = document.getElementById("voice-visualizer");
const voiceCtx = voiceCanvas.getContext("2d");

function drawWavePlaceholder() {
  const w = voiceCanvas.width;
  const h = voiceCanvas.height;
  voiceCtx.clearRect(0, 0, w, h);
  
  // Draw flat horizontal lines (equalizer idle state)
  voiceCtx.fillStyle = "rgba(255, 255, 255, 0.08)";
  const barWidth = 3;
  const gap = 3;
  const totalBars = Math.floor(w / (barWidth + gap));
  
  for(let i=0; i<totalBars; i++) {
    const x = i * (barWidth + gap);
    voiceCtx.fillRect(x, h/2 - 1.5, barWidth, 3);
  }
}

async function recordVoiceClip() {
  if (state.isRecordingVoice) return;
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.isRecordingVoice = true;
    const recordedChunks = [];
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => recordedChunks.push(e.data);
    mediaRecorder.start();
    btnRecordVoice.classList.add("recording");
    voiceStatusText.textContent = "Listening... (3 seconds)";
    
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = state.audioContext.createMediaStreamSource(stream);
    state.audioAnalyser = state.audioContext.createAnalyser();
    state.audioAnalyser.fftSize = 64; // Small fft for 32 clean bars
    source.connect(state.audioAnalyser);
    
    const bufferLength = state.audioAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
      if (!state.isRecordingVoice) return;
      requestAnimationFrame(draw);
      
      state.audioAnalyser.getByteFrequencyData(dataArray);
      
      const w = voiceCanvas.width;
      const h = voiceCanvas.height;
      voiceCtx.fillStyle = "#161b22";
      voiceCtx.fillRect(0, 0, w, h);
      
      const barWidth = 4;
      const gap = 3;
      const totalBars = Math.min(bufferLength, Math.floor(w / (barWidth + gap)));
      
      for(let i = 0; i < totalBars; i++) {
        // Normalize value
        const val = dataArray[i] / 255.0;
        // Calculate height
        const barHeight = Math.max(3, val * h * 0.95);
        const x = i * (barWidth + gap);
        const y = h - barHeight;
        
        // Draw rounded gradient bar
        const grad = voiceCtx.createLinearGradient(x, h, x, y);
        grad.addColorStop(0, "#6366f1");
        grad.addColorStop(1, "#06b6d4");
        
        // Glow effect
        voiceCtx.shadowBlur = 8;
        voiceCtx.shadowColor = "rgba(99, 102, 241, 0.4)";
        
        voiceCtx.fillStyle = grad;
        voiceCtx.fillRect(x, y, barWidth, barHeight);
      }
      voiceCtx.shadowBlur = 0;
    }
    
    draw();
    
    setTimeout(async () => {
      state.isRecordingVoice = false;
      btnRecordVoice.classList.remove("recording");
      voiceStatusText.textContent = "Analyzing tone...";

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop());
        if (state.audioContext) state.audioContext.close();

        const audioBlob = new Blob(recordedChunks, { type: "audio/webm" });
        const result = await sendVoiceBlobToBackend(audioBlob);
        if (result && result.success) {
          voiceStatusText.textContent = `Tone: ${result.detectedEmotion.toUpperCase()}`;
          updateSystemMood(result.detectedEmotion);
        } else {
          voiceStatusText.textContent = "Voice scan failed";
        }
        drawWavePlaceholder();
      };

      mediaRecorder.stop();
    }, 3000);
    
  } catch (err) {
     console.error("Mic error details:", err);
     alert("Unable to access microphone.");
     voiceStatusText.textContent = "Microphone error";
   }
}

// ==========================================
// 6. TEXT SENTIMENT JOURNAL WIDGET
// ==========================================
const journalInput = document.getElementById("journal-input");
const sentimentDot = document.getElementById("sentiment-dot");
const sentimentText = document.getElementById("sentiment-text");
let textTimer = null;

journalInput.addEventListener("input", () => {
  clearTimeout(textTimer);
  sentimentText.textContent = "Typing...";
  sentimentDot.style.backgroundColor = "var(--text-muted)";
  sentimentDot.style.boxShadow = "none";
  
  textTimer = setTimeout(async () => {
    const textVal = journalInput.value.trim();
    if (!textVal) {
      sentimentText.textContent = "Idle";
      return;
    }
    
    const result = await sendTextToBackend(textVal);
    if (result && result.success) {
      sentimentText.textContent = result.detectedEmotion.toUpperCase();
      sentimentDot.style.backgroundColor = "var(--accent-green)";
      sentimentDot.style.boxShadow = "0 0 8px var(--accent-green)";
      
      updateSystemMood(result.detectedEmotion);
    }
  }, 1000);
});

// ==========================================
// 7. 2D MANUAL MOOD GRID & COORDINATE HUD
// ==========================================
const manualGrid = document.getElementById("manual-mood-grid");
const manualPointer = document.getElementById("manual-mood-pointer");
const gridTooltip = document.getElementById("manual-grid-tooltip");

manualGrid.addEventListener("mousemove", (e) => {
  const rect = manualGrid.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  
  // Calculate Valence (-1 to 1) and Arousal (-1 to 1)
  const valence = (x / rect.width) * 2 - 1;
  const arousal = 1 - (y / rect.height) * 2;
  
  // Show tooltip following cursor position
  gridTooltip.style.display = "block";
  gridTooltip.style.left = `${x}px`;
  gridTooltip.style.top = `${y}px`;
  gridTooltip.textContent = `Val: ${valence >= 0 ? '+' : ''}${valence.toFixed(2)} | Aro: ${arousal >= 0 ? '+' : ''}${arousal.toFixed(2)}`;
});

manualGrid.addEventListener("mouseleave", () => {
  gridTooltip.style.display = "none";
});

manualGrid.addEventListener("click", (e) => {
  const rect = manualGrid.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  
  const px = (x / rect.width) * 100;
  const py = (y / rect.height) * 100;
  manualPointer.style.left = `${px}%`;
  manualPointer.style.top = `${py}%`;
  
  const valence = (x / rect.width) * 2 - 1;
  const arousal = 1 - (y / rect.height) * 2;
  
  let mood = "calm";
if (valence >= 0 && arousal >= 0) mood = "happy";
else if (valence >= 0 && arousal < 0) mood = "romantic";
else if (valence < 0 && arousal >= 0) mood = "energetic";
else if (valence < 0 && arousal < 0) mood = "sad";
  
  updateSystemMood(mood, valence, arousal);
});

// ==========================================
// 8. MOOD MIXOLOGY SLIDERS (Alchemy)
// ==========================================
const mixologyContainer = document.getElementById("mixology-sliders-container");
const mixArtGradient = document.getElementById("mix-art-gradient");
const mixPlaylistTitle = document.getElementById("mix-playlist-title");
const mixPlaylistSubtitle = document.getElementById("mix-playlist-subtitle");
const btnResetBrew = document.getElementById("btn-reset-brew");
const btnBrewPlaylist = document.getElementById("btn-brew-playlist");

const INTEGRATOR_EMOTIONS = ["happy", "sad", "calm", "energetic", "motivated", "romantic"];

function renderMixologySliders() {
  if (!mixologyContainer) return;
  mixologyContainer.innerHTML = "";
  
  INTEGRATOR_EMOTIONS.forEach(emotion => {
    const card = document.createElement("div");
    card.className = "mix-slider-card";
    
    let dotColor = "var(--accent-color)";
    if (emotion === "happy") dotColor = "var(--accent-orange)";
    if (emotion === "energetic") dotColor = "var(--accent-red)";
    if (emotion === "calm") dotColor = "var(--accent-cyan)";
    if (emotion === "focused") dotColor = "var(--accent-green)";
    if (emotion === "sad") dotColor = "var(--text-muted)";
    if (emotion === "romantic") dotColor = "var(--accent-pink)";
    
    card.innerHTML = `
      <div class="mix-slider-header">
        <span class="mix-slider-title">
          <span class="mix-slider-dot" style="background-color: ${dotColor};"></span>
          ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}
        </span>
        <span class="mix-slider-value" id="val-mix-${emotion}">0%</span>
      </div>
      <input type="range" class="input-slider mix-slider" id="slider-mix-${emotion}" min="0" max="100" value="0">
    `;
    
    mixologyContainer.appendChild(card);
    
    const slider = card.querySelector(".mix-slider");
    slider.addEventListener("input", () => {
      document.getElementById(`val-mix-${emotion}`).textContent = `${slider.value}%`;
      updateMixologyCard();
    });
  });
}

function updateMixologyCard() {
  const sliders = document.querySelectorAll(".mix-slider");
  let total = 0;
  let highestVal = -1;
  let highestMood = "";
  
  sliders.forEach(slider => {
    const val = parseInt(slider.value);
    total += val;
    const id = slider.id.replace("slider-mix-", "");
    if (val > highestVal) {
      highestVal = val;
      highestMood = id;
    }
  });
  
  if (total === 0) {
    mixArtGradient.style.background = "linear-gradient(135deg, var(--bg-card), var(--border-color))";
    mixPlaylistTitle.textContent = "Custom Formula";
    mixPlaylistSubtitle.textContent = "Blend ingredients";
    return;
  }
  
  let c1 = "var(--accent-color)";
  if (highestMood === "happy") c1 = "var(--accent-orange)";
  if (highestMood === "energetic") c1 = "var(--accent-red)";
  if (highestMood === "calm") c1 = "var(--accent-cyan)";
  if (highestMood === "focused") c1 = "var(--accent-green)";
  if (highestMood === "sad") c1 = "var(--text-muted)";
  if (highestMood === "romantic") c1 = "var(--accent-pink)";
  
  mixArtGradient.style.background = `linear-gradient(135deg, ${c1}, var(--bg-main))`;
  mixPlaylistTitle.textContent = `${highestMood.charAt(0).toUpperCase() + highestMood.slice(1)} Essence`;
  mixPlaylistSubtitle.textContent = `Strength: ${Math.round((highestVal / total) * 100)}%`;
}

btnResetBrew.addEventListener("click", () => {
  document.querySelectorAll(".mix-slider").forEach(slider => {
    slider.value = 0;
    const id = slider.id.replace("slider-mix-", "");
    document.getElementById(`val-mix-${id}`).textContent = "0%";
  });
  updateMixologyCard();
});

btnBrewPlaylist.addEventListener("click", async () => {
  const sliders = document.querySelectorAll(".mix-slider");
  let highestVal = -1;
  let highestMood = "happy";

  sliders.forEach(slider => {
    const val = parseInt(slider.value);
    const id = slider.id.replace("slider-mix-", "");
    if (val > highestVal) {
      highestVal = val;
      highestMood = id;
    }
  });

  const mood = highestMood.charAt(0).toUpperCase() + highestMood.slice(1);

  try {
    const response = await fetch(`${API_CONFIG.serverBaseUrl}${API_CONFIG.endpoints.recommendations}?mood=${mood}`);
    const songData = await response.json();
    renderTrackList("brewed-tracks-container", songData);
  } catch (e) {
    console.error("Brew error:", e);
  }
});

// ==========================================
// 9. SYSTEM MOOD & AMBIENT GLOW BACKDROP
// ==========================================
function updateSystemMood(mood, valence = 0.5, energy = 0.5) {
  state.currentMood = mood;
  
  // Update top badge
  const moodBadge = document.getElementById("current-mood-text");
  const moodDot = document.querySelector(".active-mood-dot");
  moodBadge.textContent = `Mood: ${mood.toUpperCase()}`;
  
  let color = "#6366f1";
  let glowStyle = "rgba(99, 102, 241, 0.04)"; // default glow color
  
  if (mood === "happy") { color = "var(--accent-orange)"; glowStyle = "rgba(245, 158, 11, 0.05)"; }
  if (mood === "energetic") { color = "var(--accent-red)"; glowStyle = "rgba(239, 68, 68, 0.05)"; }
  if (mood === "calm") { color = "var(--accent-cyan)"; glowStyle = "rgba(6, 182, 212, 0.05)"; }
  if (mood === "focused") { color = "var(--accent-green)"; glowStyle = "rgba(16, 185, 129, 0.05)"; }
  if (mood === "sad") { color = "var(--text-muted)"; glowStyle = "rgba(107, 114, 128, 0.05)"; }
  if (mood === "romantic") { color = "var(--accent-pink)"; glowStyle = "rgba(236, 72, 153, 0.05)"; }
  
  moodDot.style.backgroundColor = color;
  moodDot.style.boxShadow = `0 0 10px ${color}`;
  
  document.getElementById("recommendation-header-title").textContent = `Recommended for feeling ${mood.toUpperCase()}`;
  
  // Update dynamic background ambient glow backdrop
  const dynamicBackdrop = document.getElementById("theme-ambient-glow");
  if (dynamicBackdrop) {
    dynamicBackdrop.style.background = `radial-gradient(circle at 85% 15%, ${glowStyle} 0%, transparent 55%)`;
  }
  
  // Call hook
  fetchRecommendationsFromBackend(mood, valence, energy);
}

// ==========================================
// 10. PERSISTENT PLAYER BAR CONTROLS
// ==========================================
const btnPlayerPlay = document.getElementById("btn-player-play");
const progressContainer = document.getElementById("player-progress-container");
const progressBar = document.getElementById("player-progress-bar");
const timeCurrent = document.getElementById("current-player-time");
const volumeSlider = document.getElementById("player-volume-slider");
const btnVolumeMute = document.getElementById("btn-volume-mute");
const volumeIcon = document.getElementById("volume-icon");
const playingEqIcon = document.getElementById("playing-eq-icon");

// Ambient Overlay elements
const ambientOverlay = document.getElementById("ambient-visualizer-overlay");
const btnOpenAmbient = document.getElementById("btn-open-ambient");
const btnCloseAmbient = document.getElementById("btn-close-ambient");
const ambientCanvas = document.getElementById("ambient-osc");
const ambientCtx = ambientCanvas.getContext("2d");

function togglePlayPause() {
  state.isPlaying = !state.isPlaying;
  if (state.isPlaying) {
    btnPlayerPlay.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
    
    // Show active player EQ bouncing icon
    if(playingEqIcon) playingEqIcon.style.display = "inline-flex";
    
    // Start fake progress playback slider animation
    clearInterval(state.playbackTimer);
    state.playbackTimer = setInterval(() => {
      state.currentTime += 1;
      timeCurrent.textContent = formatTime(state.currentTime);
      const pct = Math.min((state.currentTime / 180) * 100, 100); // Mock 3min song
      progressBar.style.width = `${pct}%`;
      
      if (state.currentTime >= 180) {
        state.currentTime = 0;
      }
      
      // Draw standard placeholder waves
      if (ambientOverlay.classList.contains("active")) {
        drawFakeOscilloscope();
      }
    }, 1000);
  } else {
    btnPlayerPlay.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
    
    // Hide active player EQ bouncing icon
    if(playingEqIcon) playingEqIcon.style.display = "none";
    
    clearInterval(state.playbackTimer);
  }
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
}

progressContainer.addEventListener("click", (e) => {
  const rect = progressContainer.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickPct = clickX / rect.width;
  state.currentTime = Math.floor(clickPct * 180);
  progressBar.style.width = `${clickPct * 100}%`;
  timeCurrent.textContent = formatTime(state.currentTime);
});

// volume slider logic
volumeSlider.addEventListener("input", () => {
  state.volume = volumeSlider.value / 100;
  state.isMuted = state.volume === 0;
  updateVolumeIcon();
});

btnVolumeMute.addEventListener("click", () => {
  state.isMuted = !state.isMuted;
  updateVolumeIcon();
});

function updateVolumeIcon() {
  if (state.isMuted) {
    volumeIcon.innerHTML = `<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>`;
    volumeSlider.value = 0;
  } else {
    volumeIcon.innerHTML = `<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>`;
    volumeSlider.value = state.volume * 100;
  }
}

// Ambient View controls
btnOpenAmbient.addEventListener("click", () => {
  ambientOverlay.style.display = "block";
  setTimeout(() => {
    ambientOverlay.classList.add("active");
  }, 50);
});

btnCloseAmbient.addEventListener("click", () => {
  ambientOverlay.classList.remove("active");
  setTimeout(() => {
    ambientOverlay.style.display = "none";
  }, 500);
});

function drawFakeOscilloscope() {
  ambientCtx.clearRect(0, 0, ambientCanvas.width, ambientCanvas.height);
  ambientCtx.strokeStyle = "rgba(255, 255, 255, 0.35)";
  ambientCtx.lineWidth = 2.5;
  ambientCtx.beginPath();
  
  const width = ambientCanvas.width;
  const height = ambientCanvas.height;
  
  ambientCtx.moveTo(0, height / 2);
  const time = Date.now() * 0.005;
  for (let x = 0; x < width; x++) {
    const angle = (x / width) * Math.PI * 6 + time;
    const y = height / 2 + Math.sin(angle) * 20 * Math.sin(x * 0.01);
    ambientCtx.lineTo(x, y);
  }
  ambientCtx.stroke();
}

// ==========================================
// 11. GRAPHICAL GRAPHICS (Mood trend canvas rendering)
// ==========================================
function drawHistoryChart() {
  const chartCanvas = document.getElementById("mood-trend-canvas");
  if (!chartCanvas) return;
  
  const rect = chartCanvas.getBoundingClientRect();
  chartCanvas.width = rect.width;
  chartCanvas.height = rect.height;
  
  const ctx = chartCanvas.getContext("2d");
  ctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
  
  const margin = 30;
  const graphWidth = chartCanvas.width - margin * 2;
  const graphHeight = chartCanvas.height - margin * 2;
  
  // Draw grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = margin + (graphHeight * i) / 4;
    ctx.beginPath();
    ctx.moveTo(margin, y);
    ctx.lineTo(chartCanvas.width - margin, y);
    ctx.stroke();
  }
  
  const points = state.weeklyHistory.map((d, index) => {
    const x = margin + (graphWidth * index) / (state.weeklyHistory.length - 1);
    const y = margin + graphHeight * (1 - d.valence);
    return { x, y, info: d };
  });
  
  // Draw gradient area
  const fillGrad = ctx.createLinearGradient(0, margin, 0, chartCanvas.height - margin);
  fillGrad.addColorStop(0, "rgba(99, 102, 241, 0.12)");
  fillGrad.addColorStop(1, "rgba(99, 102, 241, 0)");
  ctx.fillStyle = fillGrad;
  ctx.beginPath();
  ctx.moveTo(points[0].x, chartCanvas.height - margin);
  points.forEach(pt => ctx.lineTo(pt.x, pt.y));
  ctx.lineTo(points[points.length - 1].x, chartCanvas.height - margin);
  ctx.closePath();
  ctx.fill();
  
  // Draw curve line
  ctx.strokeStyle = "#6366f1";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 0; i < points.length - 1; i++) {
    const xc = (points[i].x + points[i+1].x) / 2;
    const yc = (points[i].y + points[i+1].y) / 2;
    ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
  }
  ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
  ctx.stroke();
  
  // Draw dot nodes and day labels
  points.forEach(pt => {
    ctx.fillStyle = "#6366f1";
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = "var(--text-muted)";
    ctx.font = "500 11px 'Outfit'";
    ctx.textAlign = "center";
    ctx.fillText(pt.info.day, pt.x, chartCanvas.height - 10);
  });
}

window.addEventListener("resize", drawHistoryChart);

// ==========================================
// 12. INITIALIZATION ENTRYPOINT
// ==========================================
function init() {
  initNavigation();
  renderMixologySliders();
  drawWavePlaceholder();
  drawHistoryChart();
  
  // Dynamic header buttons configurations
  const tagContainer = document.getElementById("search-mood-tags");
  if (tagContainer) {
    tagContainer.innerHTML = `<button class="mood-tag-btn active" data-mood="all">All Moods</button>`;
    INTEGRATOR_EMOTIONS.forEach(emotion => {
      const btn = document.createElement("button");
      btn.className = "mood-tag-btn";
      btn.setAttribute("data-mood", emotion);
      btn.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
      tagContainer.appendChild(btn);
    });
    
    // Attach filter clicks
    tagContainer.querySelectorAll(".mood-tag-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        tagContainer.querySelectorAll(".mood-tag-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        console.log(`Filter list click: filtered by mood tags '${btn.getAttribute("data-mood")}'`);
      });
    });
  }
  
  btnToggleCamera.addEventListener("click", toggleCamera);
  btnCaptureMood.addEventListener("click", captureAndScanFace);
  btnRecordVoice.addEventListener("click", recordVoiceClip);
  btnPlayerPlay.addEventListener("click", togglePlayPause);
}

window.addEventListener("DOMContentLoaded", init);
