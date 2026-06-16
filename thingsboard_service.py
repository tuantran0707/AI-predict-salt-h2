"""
thingsboard_service.py — backward-compatible entry point.

Runs the live camera + detection loop with ThingsBoard MQTT enabled.
Camera and MJPEG stream stay active continuously; detection and
telemetry run only when shared attribute detect_salt becomes true.

Pi (co man hinh):
    python3 thingsboard_service.py --token <TOKEN>

Pi (SSH, khong co man hinh):
    python3 thingsboard_service.py --token <TOKEN> --headless

Stream: http://<Pi-IP>:5800/
"""

from __future__ import annotations

import argparse
import os
import sys

from version import SERVICE_VERSION


def _setup_hdmi_display() -> None:
    """OpenCV imshow needs DISPLAY when launched from SSH on Pi desktop."""
    if os.name != "posix" or os.environ.get("DISPLAY"):
        return
    os.environ["DISPLAY"] = ":0"
    print("[thingsboard_service] Set DISPLAY=:0 for HDMI.", flush=True)


def main() -> None:
    print(f"[thingsboard_service] version {SERVICE_VERSION}", flush=True)

    p = argparse.ArgumentParser(
        description="Camera + ThingsBoard MQTT (port 1883 by default).")
    p.add_argument(
        "--token",
        default=os.environ.get("TB_ACCESS_TOKEN", "X8c1PxfnSFuUvsS0KhBI"),
    )
    p.add_argument("--broker",
                   default=os.environ.get("TB_BROKER_HOST",
                                          "mqtt.thingsboard.cloud"))
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("TB_BROKER_PORT", "1883")))
    p.add_argument("--tls", action="store_true", help="Use MQTTS port 8883.")
    p.add_argument("--headless", action="store_true",
                   help="Khong mo cua so GUI (chi stream MJPEG).")
    args, rest = p.parse_known_args()

    argv = [sys.argv[0],
            "--tb-token", args.token,
            "--tb-broker", args.broker,
            "--tb-port", str(args.port)]
    if args.tls:
        argv.append("--tb-tls")
    if args.headless:
        argv.append("--headless")
        print("[!] --headless: KHONG hien thi tren man hinh HDMI.", flush=True)
        print("    Bo --headless neu muon xem tren TV/monitor Pi.", flush=True)
    else:
        _setup_hdmi_display()
    # 1440x810: slightly sharper, same 16:9 FOV/zoom as 1280x720
    argv.extend(["--width", "1440", "--height", "810"])
    argv.extend(rest)

    print("[thingsboard_service] Starting with on-demand detection.", flush=True)
    print(f"[thingsboard_service] Token: {args.token[:8]}...", flush=True)
    if args.headless:
        print("[thingsboard_service] Headless — no GUI window.", flush=True)
    else:
        print("[thingsboard_service] GUI window will open on the Pi display.",
              flush=True)
    sys.argv = argv
    from run_camera import main as run_main
    run_main()


if __name__ == "__main__":
    main()
