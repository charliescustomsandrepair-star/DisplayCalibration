"""
DisplayCalibration Capture Server
==================================
Flask + SocketIO server that:
1. Displays fullscreen fringe patterns on the computer monitor
2. Serves the mobile capture interface to the user's phone
3. Receives uploaded calibration images from the phone
4. Orchestrates the 15-capture sequence via WebSocket
5. Runs edge detection / cropping on received images
6. Feeds processed images into the calibration pipeline
"""

import os
import sys
import json
import time
import socket
import base64
import logging
import threading
from io import BytesIO
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import qrcode

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS

from calibration_pipeline import run_calibration_session, save_token

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAPTURE_COUNT = 15          # Total captures in a full calibration run
SETTLING_TIME_MS = 800      # ms to wait after pattern change before capture
PHASE_STEPS = 5             # Phase shifts per frequency
FREQUENCIES = [3, 9, 27]    # Fringe frequencies (cycles across screen)
# 5 phase steps × 3 frequencies = 15 captures

CAPTURE_DIR = Path(__file__).parent / "captures"
PROCESSED_DIR = Path(__file__).parent / "processed"
CALIBRATION_OUTPUT = Path(__file__).parent / "calibration_result.json"
PATTERN_ACK_LOCK = threading.Lock()
PATTERN_ACK: dict = {}

# Camera lock recommendations (sent to client)
CAMERA_DEFAULTS = {
    "iso": 100,
    "focusDistance": 0.3,       # meters — typical phone-to-screen distance
    "exposureTime": 33333,      # microseconds (1/30s)
    "whiteBalance": 6500,       # Kelvin (daylight)
    "fillPercent": 0.95,        # Target: 95% of camera view filled with display
    "settlingTimeMs": SETTLING_TIME_MS,
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")
app.config["SECRET_KEY"] = "calibration-secret"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    max_http_buffer_size=50 * 1024 * 1024)  # 50 MB for images

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("capture-server")

# State
capture_state = {
    "running": False,
    "current_index": 0,
    "total": CAPTURE_COUNT,
    "captured_files": [],
    "start_time": None,
    "session_id": None,
    "last_calibration_path": None,
}


def schedule_ready_fallback(pattern_index: int):
    """If the pattern display never acks, still allow capture after settling + margin."""

    def _fallback():
        delay = SETTLING_TIME_MS / 1000.0 + 1.0
        time.sleep(delay)
        with PATTERN_ACK_LOCK:
            if not capture_state.get("running"):
                return
            if capture_state.get("current_index") != pattern_index:
                return
            if PATTERN_ACK.get(pattern_index):
                return
            PATTERN_ACK[pattern_index] = True
        pat = PATTERN_SEQUENCE[pattern_index] if pattern_index < len(PATTERN_SEQUENCE) else {}
        log.warning("Pattern ack timeout — releasing capture %s", pattern_index)
        socketio.emit(
            "ready_for_capture",
            {
                "index": pattern_index,
                "pattern": pat,
                "settling_time": 0,
                "fallback": True,
            },
            room="capture",
        )

    threading.Thread(target=_fallback, daemon=True).start()


def broadcast_show_pattern(pattern_index: int):
    with PATTERN_ACK_LOCK:
        PATTERN_ACK[pattern_index] = False
    pattern = PATTERN_SEQUENCE[pattern_index]
    socketio.emit(
        "show_pattern",
        {
            "index": pattern_index,
            "frequency": pattern["frequency"],
            "phase": pattern["phase"],
            "label": pattern["label"],
            "settling_time": SETTLING_TIME_MS,
        },
    )
    schedule_ready_fallback(pattern_index)


def get_local_ip():
    """Get the machine's LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def generate_qr(url: str) -> str:
    """Generate a QR code PNG as a base64 data URI."""
    qr = qrcode.QRCode(version=1, box_size=6, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def build_pattern_sequence():
    """Build the ordered list of (frequency, phase_shift) for all 15 captures."""
    sequence = []
    for freq in FREQUENCIES:
        for step in range(PHASE_STEPS):
            phase = (2.0 * np.pi * step) / PHASE_STEPS
            sequence.append({"frequency": freq, "phase": round(phase, 6),
                             "label": f"f={freq} step={step+1}/{PHASE_STEPS}"})
    return sequence


PATTERN_SEQUENCE = build_pattern_sequence()


def detect_screen_edges(img_array):
    """
    Detect the display edges in the captured image using gradient analysis.
    Returns crop rectangle (x, y, w, h) and scale info.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to find bright screen region against dark surroundings
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find largest contour (the display)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: use entire image
        h, w = gray.shape[:2]
        return {"x": 0, "y": 0, "w": w, "h": h, "rotation": 0.0,
                "scale_x": 1.0, "scale_y": 1.0, "confidence": 0.0}

    largest = max(contours, key=cv2.contourArea)

    # Fit rotated rectangle for orientation
    rect = cv2.minAreaRect(largest)
    center, (rw, rh), angle = rect

    # Ensure width > height for consistent orientation
    if rw < rh:
        rw, rh = rh, rw
        angle += 90

    # Bounding rect for cropping
    x, y, w, h = cv2.boundingRect(largest)

    # Scale: ratio of detected screen size to image size
    img_h, img_w = gray.shape[:2]
    scale_x = w / img_w
    scale_y = h / img_h

    return {
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "rotation": round(float(angle), 3),
        "scale_x": round(scale_x, 4),
        "scale_y": round(scale_y, 4),
        "center_x": round(float(center[0]), 1),
        "center_y": round(float(center[1]), 1),
        "confidence": round(float(cv2.contourArea(largest) / (img_w * img_h)), 4),
    }


def correct_perspective(img_array, edge_info):
    """
    Apply perspective correction based on detected edges.
    Corrects rotation and crops to screen bounds.
    """
    x, y, w, h = edge_info["x"], edge_info["y"], edge_info["w"], edge_info["h"]
    rotation = edge_info.get("rotation", 0.0)

    # Rotate if needed (small angles only — large rotation means phone was tilted)
    if abs(rotation) > 0.5 and abs(rotation) < 10.0:
        center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        img_array = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))

    # Crop to screen region
    cropped = img_array[y:y+h, x:x+w]
    return cropped


def process_captured_image(image_bytes, capture_index, session_dir):
    """
    Process a single captured image:
    1. Decode
    2. Detect screen edges
    3. Correct perspective
    4. Crop to pattern region
    5. Save processed image
    Returns metadata dict.
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Failed to decode image", "index": capture_index}

    # Save raw image
    raw_path = session_dir / f"raw_{capture_index:02d}.png"
    cv2.imwrite(str(raw_path), img)

    # Detect screen edges
    edge_info = detect_screen_edges(img)

    # Perspective correction and crop
    corrected = correct_perspective(img, edge_info)

    # Save processed image
    proc_path = session_dir / f"processed_{capture_index:02d}.png"
    cv2.imwrite(str(proc_path), corrected)

    pattern = PATTERN_SEQUENCE[capture_index] if capture_index < len(PATTERN_SEQUENCE) else {}

    metadata = {
        "index": capture_index,
        "raw_path": str(raw_path),
        "processed_path": str(proc_path),
        "raw_size": {"w": img.shape[1], "h": img.shape[0]},
        "cropped_size": {"w": corrected.shape[1], "h": corrected.shape[0]},
        "edge_info": edge_info,
        "pattern": pattern,
        "timestamp": datetime.now().isoformat(),
    }
    return metadata


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

PORT = int(os.environ.get("PORT", "5000"))


@app.route("/")
def index():
    """Landing page — shows QR code and instructions."""
    ip = get_local_ip()
    port = PORT
    capture_url = f"http://{ip}:{port}/capture"
    qr_data = generate_qr(capture_url)
    return render_template("index.html",
                           capture_url=capture_url,
                           qr_data=qr_data,
                           ip=ip, port=port)


@app.route("/patterns")
def patterns():
    """Fullscreen pattern display page (shown on the computer monitor)."""
    return render_template("patterns.html",
                           sequence=json.dumps(PATTERN_SEQUENCE),
                           settling_time=SETTLING_TIME_MS)


@app.route("/capture")
def capture():
    """Mobile capture interface (opened on the phone via QR code)."""
    return render_template("capture.html",
                           total=CAPTURE_COUNT,
                           camera_defaults=json.dumps(CAMERA_DEFAULTS),
                           settling_time=SETTLING_TIME_MS)


@app.route("/api/status")
def api_status():
    """Return current capture session state."""
    return jsonify(capture_state)


@app.route("/api/pattern-sequence")
def api_pattern_sequence():
    """Return the full pattern sequence."""
    return jsonify({"sequence": PATTERN_SEQUENCE, "total": CAPTURE_COUNT})


@app.route("/api/calibration-token")
def api_calibration_token():
    """Return the latest calibration token JSON for the perceptual renderer."""
    if not CALIBRATION_OUTPUT.exists():
        return jsonify({"error": "No calibration token yet"}), 404
    with open(CALIBRATION_OUTPUT, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------

@socketio.on("connect")
def handle_connect():
    log.info(f"Client connected: {request.sid}")
    emit("server_state", capture_state)


@socketio.on("join_role")
def handle_join_role(data):
    role = (data or {}).get("role")
    if role == "capture":
        join_room("capture")
        log.info("Client joined capture room: %s", request.sid)
    elif role == "pattern":
        join_room("pattern")
        log.info("Client joined pattern room: %s", request.sid)


@socketio.on("pattern_settled")
def handle_pattern_settled(data):
    """Pattern display finished settling; allow the phone to capture."""
    idx = (data or {}).get("index")
    if idx is None or not capture_state.get("running"):
        return
    if int(idx) != int(capture_state["current_index"]):
        return
    with PATTERN_ACK_LOCK:
        if PATTERN_ACK.get(int(idx)):
            return
        PATTERN_ACK[int(idx)] = True
    pat = PATTERN_SEQUENCE[int(idx)] if int(idx) < len(PATTERN_SEQUENCE) else {}
    socketio.emit(
        "ready_for_capture",
        {
            "index": int(idx),
            "pattern": pat,
            "settling_time": 0,
            "fallback": False,
        },
        room="capture",
    )


@socketio.on("start_capture")
def handle_start_capture():
    """Phone signals to begin the capture sequence."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = CAPTURE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    capture_state.update({
        "running": True,
        "current_index": 0,
        "total": CAPTURE_COUNT,
        "captured_files": [],
        "start_time": time.time(),
        "session_id": session_id,
    })

    log.info(f"Capture session started: {session_id}")

    broadcast_show_pattern(0)

    # Tell the phone the sequence has started
    emit(
        "capture_started",
        {
            "session_id": session_id,
            "total": CAPTURE_COUNT,
            "settling_time": SETTLING_TIME_MS,
            "first_pattern": PATTERN_SEQUENCE[0],
        },
    )


@socketio.on("image_captured")
def handle_image_captured(data):
    """Phone sends a captured image."""
    if not capture_state["running"]:
        emit("error", {"message": "No active capture session"})
        return

    idx = capture_state["current_index"]
    session_id = capture_state["session_id"]
    session_dir = CAPTURE_DIR / session_id

    # Decode base64 image data
    image_data = data.get("image", "")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        emit("error", {"message": f"Failed to decode image: {str(e)}"})
        return

    # Process the image
    metadata = process_captured_image(image_bytes, idx, session_dir)

    if "error" in metadata:
        emit("capture_error", metadata)
        return

    capture_state["captured_files"].append(metadata)
    log.info(f"Captured image {idx+1}/{CAPTURE_COUNT}: "
             f"edges={metadata['edge_info']['confidence']:.2f} confidence")

    # Notify phone of successful capture
    emit("capture_confirmed", {
        "index": idx,
        "edge_info": metadata["edge_info"],
        "cropped_size": metadata["cropped_size"],
    })

    # Advance to next pattern or finish
    next_idx = idx + 1
    capture_state["current_index"] = next_idx

    if next_idx >= CAPTURE_COUNT:
        # All captures complete
        elapsed = time.time() - capture_state["start_time"]
        capture_state["running"] = False

        # Save session metadata
        session_meta = {
            "session_id": session_id,
            "total_captures": CAPTURE_COUNT,
            "elapsed_seconds": round(elapsed, 2),
            "captures": capture_state["captured_files"],
            "pattern_sequence": PATTERN_SEQUENCE,
            "camera_settings": CAMERA_DEFAULTS,
        }
        meta_path = session_dir / "session_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(session_meta, f, indent=2)

        log.info(f"Capture session complete: {CAPTURE_COUNT} images in {elapsed:.1f}s")

        token = run_calibration_session(session_dir, PATTERN_SEQUENCE, FREQUENCIES)
        save_token(token, CALIBRATION_OUTPUT)
        save_token(token, session_dir / "calibration_token.json")
        capture_state["last_calibration_path"] = str(CALIBRATION_OUTPUT.resolve())
        log.info("Calibration token written to %s", CALIBRATION_OUTPUT)

        socketio.emit(
            "capture_complete",
            {
                "session_id": session_id,
                "total": CAPTURE_COUNT,
                "elapsed": round(elapsed, 2),
                "output_dir": str(session_dir),
                "calibration_path": str(CALIBRATION_OUTPUT.resolve()),
                "layout": token.get("layout"),
                "gamma_estimate": token.get("gamma_estimate"),
            },
        )
        socketio.emit(
            "calibration_ready",
            {
                "path": str(CALIBRATION_OUTPUT.resolve()),
                "layout": token.get("layout"),
                "pitch_x": token.get("pitch_x"),
                "warnings": token.get("warnings", []),
            },
            room="capture",
        )
    else:
        broadcast_show_pattern(next_idx)


@socketio.on("abort_capture")
def handle_abort():
    """Phone or display requests capture abort."""
    capture_state["running"] = False
    log.info("Capture session aborted")
    socketio.emit("capture_aborted", {"reason": "User aborted"})


@socketio.on("disconnect")
def handle_disconnect():
    log.info(f"Client disconnected: {request.sid}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ip = get_local_ip()
    port = PORT

    print("\n" + "=" * 60)
    print("  DisplayCalibration Capture Server")
    print("=" * 60)
    print(f"\n  Desktop pattern display:  http://{ip}:{port}/patterns")
    print(f"  Mobile capture app:       http://{ip}:{port}/capture")
    print(f"  Dashboard:                http://{ip}:{port}/")
    print(f"  Calibration token API:    http://{ip}:{port}/api/calibration-token")
    print(f"\n  Open /patterns fullscreen on your monitor,")
    print(f"  then scan the QR code or open /capture on your phone.")
    print("=" * 60 + "\n")

    socketio.run(app, host="0.0.0.0", port=port, debug=False)
