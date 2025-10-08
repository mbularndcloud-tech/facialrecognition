// static/js/recognition.js (auto-start version)
const socket = io();

let video = document.getElementById('video');
let overlay = document.getElementById('overlay');
let overlayCtx = overlay.getContext('2d');
let stopBtn = document.getElementById('stopBtn');
let recName = document.getElementById('recName');
let recStatus = document.getElementById('recStatus');
let actionText = document.getElementById('actionText');
let clockDiv = document.getElementById('clock');

let streaming = false;
let stream;
let sendInterval;

function updateClock() {
    const now = new Date();
    clockDiv.textContent = now.toLocaleString();
}
setInterval(updateClock, 1000);

// Start camera automatically when page loads
async function startCameraAuto() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640 }, audio: false });
        video.srcObject = stream;
        video.play();
        streaming = true;

        video.onloadedmetadata = () => {
            overlay.width = video.videoWidth;
            overlay.height = video.videoHeight;
        };

        // start sending frames at 1 FPS (tweak as needed)
        sendInterval = setInterval(captureAndSend, 1000);
    } catch (e) {
        console.error('Unable to access camera:', e);
        recName.textContent = 'Camera error';
        recStatus.innerHTML = `<span style="color:#ff4d4f">${e.message}</span>`;
    }
}

// Stop camera function if you want to expose it later
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
    }
    clearInterval(sendInterval);
    streaming = false;
    recName.textContent = 'Stopped';
    recStatus.innerHTML = '';
    actionText.innerText = '';
}

// start immediately
startCameraAuto();

function captureAndSend() {
    if (!streaming) return;
    // draw video to canvas
    let tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = video.videoWidth;
    tmpCanvas.height = video.videoHeight;
    let ctx = tmpCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, tmpCanvas.width, tmpCanvas.height);
    // reduce size and quality
    let dataUrl = tmpCanvas.toDataURL('image/jpeg', 0.6);
    socket.emit('frame', { image: dataUrl });
}

socket.on('recognition', (payload) => {
    if (!payload) return;
    if (!payload.recognized) {
        recName.innerText = payload.label || 'Unknown';
        recStatus.innerHTML = '<span style="color:#ff4d4f">✖ Not recognized</span>';
        actionText.innerText = '';
    } else {
        recName.innerText = `${payload.name} ${payload.surname}`;
        recStatus.innerHTML = `<span style="color:#28a745">✔ Recognized</span> <small>score:${(payload.score||0).toFixed(2)}</small>`;

        // Show action only for actual DB writes
        if (payload.action === 'clock_in') {
            actionText.innerText = 'CLOCK IN';
        } else if (payload.action === 'clock_out') {
            actionText.innerText = 'CLOCK OUT';
        } else {
            // 'none' - acknowledged recognition but no DB action
            actionText.innerText = '';
        }

        // show action badge for short time when action occurred
        if (payload.action === 'clock_in' || payload.action === 'clock_out') {
            setTimeout(() => { actionText.innerText = ''; }, 3500);
        }
    }
});

