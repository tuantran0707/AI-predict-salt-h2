"""
run_camera.py
-------------
Live salt / sulfate detection on a marine battery using the Raspberry Pi 4
CSI camera (or any USB webcam on a development machine).

On the Pi we use Picamera2 (libcamera) for the CSI port. On a development
machine without picamera2 the script automatically falls back to OpenCV
VideoCapture (a USB webcam).

HTTP MJPEG stream:
    http://<Pi-IP>:5800/stream     — MJPEG stream (use in Electron app)
    http://<Pi-IP>:5800/snapshot   — single JPEG frame
    http://<Pi-IP>:5800/status     — JSON detection status

ThingsBoard MQTT (port 1883):
    python3 run_camera.py --tb-token X8c1PxfnSFuUvsS0KhBI
    Stream runs continuously; detect + publish only on detect_salt=true.

Hotkeys (GUI mode):
    q  - quit
    s  - save a snapshot to snapshots/
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from datetime import datetime
import json
import socket

import cv2
import numpy as np

from version import SERVICE_VERSION
from camera import CsiCamera
from detect_salt import (FALLBACK_TERMINAL_ROIS, SaltDetector, draw_detecting_banner,
                         draw_live_banner, draw_overlay)
from thingsboard_mqtt import DEFAULT_BROKER, DEFAULT_PORT, ThingsBoardMqtt

WINDOW = "Salt Detector - Battery"
SNAPSHOT_DIR = "snapshots"
STREAM_PORT = 5800
RESULT_SHOW_S = 4.0    # seconds to show % + overlay after TB scan


STREAM_JPEG_QUALITY = 68


class _FrameGrabber:
    """Read CSI/USB camera in a background thread (avoids libcamera timeout)."""

    def __init__(self, cam: CsiCamera):
        self._cam = cam
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            frame = self._cam.read()
            if frame is not None:
                with self._lock:
                    self._frame = frame
            time.sleep(0.001)

    def read(self, copy: bool = True) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy() if copy else self._frame

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

# ---------------------------------------------------------------------------
# Shared state between camera thread and HTTP server
# ---------------------------------------------------------------------------
_shared_lock   = threading.Lock()
_latest_jpeg   = b''          # latest JPEG for streaming
_latest_status = {}           # latest detection result dict
_frame_seq     = 0            # incremented each new JPEG

# Wake stream clients when a new frame is ready
_new_frame_event = threading.Event()

INDEX_HTML = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Salt Camera</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;
align-items:center;min-height:100vh}img{max-width:100%;max-height:100vh}</style>
</head><body><img src="/stream" alt="live camera"></body></html>
"""


def _get_lan_ip() -> str:
    """Best-effort LAN IP detection for user-facing stream URL logs."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ---------------------------------------------------------------------------
# MJPEG HTTP server
# ---------------------------------------------------------------------------
def _handle_stream(conn: socket.socket) -> None:
    """Send MJPEG multipart stream to one connected client via raw socket."""
    last_sent = 0
    try:
        conn.sendall(
            b'HTTP/1.1 200 OK\r\n'
            b'Content-Type: multipart/x-mixed-replace; boundary=mjpegframe\r\n'
            b'Cache-Control: no-cache, no-store, must-revalidate\r\n'
            b'Pragma: no-cache\r\n'
            b'Access-Control-Allow-Origin: *\r\n'
            b'\r\n'
        )
        while True:
            _new_frame_event.wait(timeout=1.0)
            with _shared_lock:
                data = _latest_jpeg
                seq = _frame_seq
            if not data or seq == last_sent:
                continue
            last_sent = seq
            chunk = (
                b'--mjpegframe\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n'
                + data + b'\r\n'
            )
            conn.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _handle_once(conn: socket.socket, request_line: str, path: str) -> None:
    """Handle single-request endpoints: /snapshot and /status."""
    cors = b'Access-Control-Allow-Origin: *\r\n'
    if path in ('/', '/index.html'):
        conn.sendall(
            b'HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n'
            + cors
            + b'Content-Length: ' + str(len(INDEX_HTML)).encode() + b'\r\n\r\n'
            + INDEX_HTML
        )
    elif path == '/snapshot':
        with _shared_lock:
            data = _latest_jpeg
        if not data:
            conn.sendall(b'HTTP/1.1 503 Not Ready\r\n' + cors + b'\r\n')
        else:
            conn.sendall(
                b'HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n'
                + cors
                + b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n'
                + data
            )
    elif path == '/status':
        with _shared_lock:
            payload = dict(_latest_status)
        body = json.dumps(payload).encode()
        conn.sendall(
            b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
            + cors
            + b'Content-Length: ' + str(len(body)).encode() + b'\r\n\r\n'
            + body
        )
    else:
        conn.sendall(b'HTTP/1.1 404 Not Found\r\n' + cors + b'\r\n')
    try:
        conn.close()
    except OSError:
        pass


def _client_worker(conn: socket.socket) -> None:
    """Parse the first HTTP request line then dispatch."""
    try:
        conn.settimeout(5.0)
        buf = b''
        while b'\r\n' not in buf and len(buf) < 512:
            chunk = conn.recv(512)
            if not chunk:
                break
            buf += chunk
        conn.settimeout(None)
        line = buf.split(b'\r\n')[0].decode(errors='replace')
        parts = line.split()
        path = parts[1] if len(parts) >= 2 else '/'
        # strip query string
        path = path.split('?')[0]
    except OSError:
        conn.close()
        return

    if path == '/stream':
        _handle_stream(conn)
    else:
        _handle_once(conn, line, path)


def _start_stream_server(port: int = STREAM_PORT) -> None:
    """Accept loop — each client runs in its own daemon thread."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', port))
    srv.listen(8)
    lan_ip = _get_lan_ip()
    print(f'[stream] Viewer:  http://{lan_ip}:{port}/', flush=True)
    print(f'[stream] MJPEG:   http://{lan_ip}:{port}/stream', flush=True)
    while True:
        try:
            conn, addr = srv.accept()
            print(f'[stream] Client connected from {addr[0]}:{addr[1]}',
                  flush=True)
            t = threading.Thread(target=_client_worker, args=(conn,),
                                 daemon=True)
            t.start()
        except OSError:
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Salt detector on CSI camera.")
    p.add_argument("--prototypes", default="prototypes.npz")
    p.add_argument("--model", default=os.path.join("model",
                                                   "mobilenetv2-12.onnx"))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--infer-every", type=int, default=5,
                   help="Run inference every N frames (smoother preview).")
    p.add_argument("--headless", action="store_true",
                   help="No GUI window. Print results to stdout, save "
                        "snapshots when salt is detected.")
    p.add_argument("--no-csi", action="store_true",
                   help="Skip Picamera2 and use OpenCV VideoCapture only.")
    p.add_argument("--camera-format", choices=("rgb", "bgr"), default="rgb",
                   help="Picamera2 format: rgb=RGB888 no convert (default, Pi), "
                        "bgr=BGR888+convert.")
    p.add_argument("--swap-rb", action="store_true",
                   help="Swap red/blue channels (use if red/blue are inverted).")
    p.add_argument("--no-pole-roi", action="store_true",
                   help="Scan full frame for salt (default: auto terminal heads).")
    p.add_argument("--fixed-roi", action="store_true",
                   help="Use fixed corner ROIs instead of auto terminal detection.")
    p.add_argument("--pole-threshold", type=float, default=0.10,
                   help="Min terminal corrosion colour ratio to flag SALT "
                        "(default 0.10; on post crop only).")
    p.add_argument("--stream-port", type=int, default=STREAM_PORT,
                   help=f"Port for the MJPEG HTTP stream (default {STREAM_PORT}).")
    p.add_argument("--tb-token",
                   default=os.environ.get("TB_ACCESS_TOKEN", ""),
                   help="ThingsBoard device token (enables MQTT upload).")
    p.add_argument("--tb-broker",
                   default=os.environ.get("TB_BROKER_HOST", DEFAULT_BROKER))
    p.add_argument("--tb-port", type=int,
                   default=int(os.environ.get("TB_BROKER_PORT",
                                              str(DEFAULT_PORT))))
    p.add_argument("--tb-tls", action="store_true",
                   help="Use MQTTS (port 8883). Default is plain MQTT 1883.")
    p.add_argument("--tb-jpeg-quality", type=int, default=50,
                   help="JPEG quality for ThingsBoard image (lower = smaller).")
    return p.parse_args()


def main() -> None:
    global _latest_jpeg, _frame_seq

    args = parse_args()
    print(f"[run_camera] version {SERVICE_VERSION}", flush=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # Start MJPEG HTTP server in a daemon thread
    t = threading.Thread(target=_start_stream_server,
                         args=(args.stream_port,), daemon=True)
    t.start()

    on_demand_detect = bool(args.tb_token)
    detector: SaltDetector | None = None

    def _terminal_mode():
        if args.no_pole_roi:
            return None
        if args.fixed_roi:
            return FALLBACK_TERMINAL_ROIS
        return "auto"

    def get_detector() -> SaltDetector:
        nonlocal detector
        if detector is None:
            print("[i] Loading model and prototypes...", flush=True)
            detector = SaltDetector(
                prototypes_path=args.prototypes,
                model_path=args.model,
                pole_cv_threshold=args.pole_threshold,
                terminal_rois=_terminal_mode())
        return detector

    if on_demand_detect:
        print("[i] ThingsBoard mode: MJPEG stream only until detect_salt=true.",
              flush=True)
    else:
        get_detector()

    cam = CsiCamera(width=args.width, height=args.height,
                    prefer_csi=not args.no_csi,
                    swap_rb=args.swap_rb,
                    camera_format=args.camera_format)
    grabber = _FrameGrabber(cam)

    if on_demand_detect:
        def _preload_model() -> None:
            try:
                get_detector()
                print("[i] Model preloaded in background.", flush=True)
            except Exception as exc:
                print(f"[!] Model preload failed: {exc}", flush=True)
        threading.Thread(target=_preload_model, daemon=True).start()

    tb: ThingsBoardMqtt | None = None
    if args.tb_token:
        tb = ThingsBoardMqtt(
            token=args.tb_token,
            broker=args.tb_broker,
            port=args.tb_port,
            use_tls=args.tb_tls,
            jpeg_quality=args.tb_jpeg_quality,
        )
        tb.start()

    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, args.width, args.height)
        print("[i] GUI window enabled (press q to quit).", flush=True)

    last_result: dict = {
        "has_salt": False, "confidence": 0.0,
        "salt_sim": 0.0, "clean_sim": 0.0, "margin": 0.0,
        "cv_ratio": 0.0,
        "best_salt_match": "-", "best_clean_match": "-",
    }
    last_alert_time = 0.0
    ALERT_COOLDOWN_S = 5.0

    frame_idx = 0
    t0 = time.time()
    fps = 0.0
    stream_ready = False
    detect_busy = False
    show_overlay_until = 0.0
    result_lock = threading.Lock()

    def _update_status(result: dict) -> None:
        with _shared_lock:
            _latest_status.clear()
            _latest_status.update({
                k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                    float(v) if isinstance(v, (float, np.floating)) else v)
                for k, v in result.items()
                if k not in ('boxes', 'poles')
            })
            poles = result.get("poles", [])
            if poles:
                _latest_status["poles"] = [
                    {"name": p["name"],
                     "has_salt": bool(p["has_salt"]),
                     "confidence": float(p["confidence"]),
                     "cv_ratio": float(p["cv_ratio"])}
                    for p in poles
                ]

    def _run_detect_job(snapshot: np.ndarray, fps_now: float) -> None:
        nonlocal detect_busy, show_overlay_until, last_result
        try:
            result = get_detector().predict(snapshot)
            overlay = draw_overlay(snapshot, result, fps_now)
            ok, jpeg_buf = cv2.imencode(
                '.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg_bytes = jpeg_buf.tobytes() if ok else b''  # type: ignore
            with result_lock:
                last_result = result
                show_overlay_until = time.time() + RESULT_SHOW_S
            _update_status(result)
            if tb is not None and jpeg_bytes:
                if not tb.poll_publish(jpeg_bytes, result):
                    tb.clear_detect_pending()
                    tb.arm_retrigger()
        except Exception as exc:
            print(f"[!] Detection failed: {exc}", flush=True)
            if tb is not None:
                tb.clear_detect_pending()
                tb.arm_retrigger()
        finally:
            detect_busy = False

    try:
        while True:
            raw = grabber.read(copy=False)
            if raw is None:
                time.sleep(0.005)
                continue

            if on_demand_detect:
                if (tb is not None and tb.should_detect()
                        and not detect_busy):
                    detect_busy = True
                    snap = raw.copy()
                    threading.Thread(
                        target=_run_detect_job,
                        args=(snap, fps),
                        daemon=True,
                    ).start()
            else:
                if frame_idx % args.infer_every == 0:
                    last_result = get_detector().predict(raw.copy())
                    _update_status(last_result)

            frame_idx += 1
            if frame_idx % 5 == 0:
                t1 = time.time()
                fps = 5.0 / (t1 - t0)
                t0 = t1

            if on_demand_detect:
                if detect_busy:
                    view = draw_detecting_banner(raw, fps)
                elif time.time() < show_overlay_until:
                    with result_lock:
                        view = draw_overlay(raw, last_result, fps)
                else:
                    view = draw_live_banner(
                        raw, fps, waiting_tb=True, last_result=None)
            else:
                view = draw_overlay(raw, last_result, fps)

            ok, jpeg_buf = cv2.imencode(
                '.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
            if ok:
                jpeg_bytes = jpeg_buf.tobytes()  # type: ignore[attr-defined]
                with _shared_lock:
                    _latest_jpeg = jpeg_bytes
                    _frame_seq += 1
                _new_frame_event.set()
                if not stream_ready:
                    stream_ready = True
                    print("[stream] First frame ready - video is live.",
                          flush=True)

            if args.headless and not on_demand_detect:
                now = time.time()
                if (last_result["has_salt"]
                        and now - last_alert_time > ALERT_COOLDOWN_S):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn = os.path.join(SNAPSHOT_DIR, f"{ts}_salt.jpg")
                    cv2.imwrite(fn, raw.copy())
                    print(f"[ALERT] {ts}  conf={last_result['confidence']:.2f}"
                          f"  margin={last_result['margin']:+.2f}"
                          f"  saved={fn}")
                    last_alert_time = now
                time.sleep(0.01)   # avoid pegging CPU on the Pi
            else:
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tag = "salt" if last_result["has_salt"] else "clean"
                    fn = os.path.join(SNAPSHOT_DIR, f"{ts}_{tag}.jpg")
                    cv2.imwrite(fn, raw.copy())
                    print(f"[OK] Snapshot saved: {fn}")
    except KeyboardInterrupt:
        print("\n[i] Interrupted, shutting down.")
    finally:
        if tb is not None:
            tb.stop()
        grabber.stop()
        cam.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
