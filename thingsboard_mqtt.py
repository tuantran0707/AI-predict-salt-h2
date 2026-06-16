"""ThingsBoard MQTT client — on-demand detect when shared detect_salt=true."""

from __future__ import annotations

import json
import ssl
import threading
import time

import paho.mqtt.client as mqtt

from detect_salt import compress_jpeg_bytes, jpeg_to_data_uri, poles_to_telemetry

DEFAULT_BROKER = "mqtt.thingsboard.cloud"
DEFAULT_PORT = 1883
MAX_MQTT_PAYLOAD = 48000   # ThingsBoard limit ~64 KB; keep margin
ATTR_TOPIC = "v1/devices/me/attributes"
TELEMETRY_TOPIC = "v1/devices/me/telemetry"
ATTR_REQUEST_TOPIC = "v1/devices/me/attributes/request/1"
ATTR_RESPONSE_TOPIC = "v1/devices/me/attributes/response/+"


def _is_detect_trigger(value) -> bool:
    """True for ThingsBoard shared attribute detect_salt=true/start."""
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("true", "start", "1", "yes")
    return False


def _extract_detect_salt(payload: dict):
    """Read detect_salt from attribute update or attribute-response body."""
    if "detect_salt" in payload:
        return payload["detect_salt"]
    shared = payload.get("shared")
    if isinstance(shared, dict) and "detect_salt" in shared:
        return shared["detect_salt"]
    return None


class ThingsBoardMqtt:
    """MQTT in background; one-shot detect + publish on detect_salt=true."""

    def __init__(self, token: str, broker: str = DEFAULT_BROKER,
                 port: int = DEFAULT_PORT, use_tls: bool = False,
                 jpeg_quality: int = 50):
        self.token = token
        self.broker = broker
        self.port = port
        self.use_tls = use_tls
        self.jpeg_quality = jpeg_quality
        self._last_attr_value = None
        self._detect_pending = threading.Event()
        self._awaiting_retrigger = False
        self._last_queue_at = 0.0
        self._pub_lock = threading.Lock()
        self._logged_connect = False

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"salt-pi-{token[:8]}",
            protocol=mqtt.MQTTv311,
        )
        self._client.username_pw_set(token)
        if use_tls:
            self._client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        self._client.reconnect_delay_set(min_delay=2, max_delay=30)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            print(f"[!] MQTT connect failed: {reason_code}", flush=True)
            return
        if not self._logged_connect:
            print(f"[mqtt] Connected to {self.broker}:{self.port}", flush=True)
            self._logged_connect = True
        client.subscribe(ATTR_TOPIC)
        client.subscribe(ATTR_RESPONSE_TOPIC)
        client.publish(
            ATTR_REQUEST_TOPIC,
            json.dumps({"sharedKeys": "detect_salt"}),
            qos=0,
        )

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code,
                       properties):
        if reason_code != 0:
            print(f"[!] MQTT disconnected (rc={reason_code})", flush=True)

    def _queue_detection(self, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_queue_at < 1.5:
            return
        print(f"[mqtt] detect_salt=true - {reason}", flush=True)
        self._detect_pending.set()
        self._awaiting_retrigger = False
        self._last_queue_at = now

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        value = _extract_detect_salt(payload)
        if value is None:
            return

        if _is_detect_trigger(value):
            rising = not _is_detect_trigger(self._last_attr_value)
            repeat = (self._awaiting_retrigger
                      and not self._detect_pending.is_set())
            if rising:
                self._queue_detection("queued (new request)")
            elif repeat:
                self._queue_detection("queued (scan again)")
            return

        if isinstance(value, str) and value.strip().lower() in (
                "false", "done", "0", "no"):
            self._awaiting_retrigger = True
        self._last_attr_value = value

    def arm_retrigger(self) -> None:
        """Allow the next detect_salt=true to start a new scan."""
        self._awaiting_retrigger = True
        self._last_attr_value = "done"

    def should_detect(self) -> bool:
        """True while a detect_salt=true request is waiting to be processed."""
        return self._detect_pending.is_set()

    def clear_detect_pending(self) -> None:
        """Release detect lock after a failed inference."""
        self._detect_pending.clear()

    def start(self) -> None:
        print(f"[mqtt] Connecting to {self.broker}:{self.port} ...", flush=True)
        self._client.connect(self.broker, self.port, keepalive=120)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def _send(self, data: dict, retries: int = 5) -> bool:
        """Publish telemetry with qos=0 (no PUBACK wait — avoids Pi timeout)."""
        payload = json.dumps(data, separators=(",", ":"))
        size = len(payload.encode())
        if size > MAX_MQTT_PAYLOAD:
            print(f"[!] Payload too large: {size} bytes", flush=True)
            return False

        for attempt in range(1, retries + 1):
            if not self._client.is_connected():
                time.sleep(0.5)
                continue
            rc = self._client.publish(TELEMETRY_TOPIC, payload, qos=0).rc
            if rc == mqtt.MQTT_ERR_SUCCESS:
                time.sleep(0.3)   # let socket flush
                return True
            print(f"[!] MQTT publish rc={rc}, retry {attempt}/{retries}",
                  flush=True)
            time.sleep(0.5)
        return False

    def _build_payload(self, jpeg_bytes: bytes, result: dict) -> dict:
        small = compress_jpeg_bytes(
            jpeg_bytes, max_width=480, quality=self.jpeg_quality,
            max_bytes=22000)
        uri = jpeg_to_data_uri(small)
        return {
            "detect_salt": "done",
            **poles_to_telemetry(result),
            "image_html": (
                f'<img src="{uri}" '
                f'style="max-width:100%;height:auto;" alt="salt detection"/>'
            ),
        }

    def _text_keys(self, payload: dict) -> dict:
        return {
            "detect_salt": payload["detect_salt"],
            "terminal_L": payload["terminal_L"],
            "terminal_R": payload["terminal_R"],
            "has_salt": payload["has_salt"],
            "trang_thai": payload["trang_thai"],
        }

    def _payload_size(self, data: dict) -> int:
        return len(json.dumps(data, separators=(",", ":")).encode())

    def _do_publish(self, jpeg_bytes: bytes, result: dict) -> bool:
        payload = self._build_payload(jpeg_bytes, result)
        text = self._text_keys(payload)

        # ThingsBoard HTML card reads keys from the LAST telemetry message —
        # the final publish must include text + image together.
        if self._payload_size(payload) <= MAX_MQTT_PAYLOAD:
            if self._send(payload):
                print(f"[mqtt] OK  L={payload['terminal_L']}  "
                      f"R={payload['terminal_R']}  "
                      f"{payload['trang_thai']}", flush=True)
                return True
        else:
            widget = {**text, "image_html": payload["image_html"]}
            if self._payload_size(widget) <= MAX_MQTT_PAYLOAD:
                if self._send(widget):
                    print(f"[mqtt] OK  L={text['terminal_L']}  "
                          f"R={text['terminal_R']}  "
                          f"{text['trang_thai']}", flush=True)
                    return True

        print("[mqtt] FAILED to publish telemetry", flush=True)
        self._send({"detect_salt": "error", "error": "publish failed"})
        return False

    def poll_publish(self, jpeg_bytes: bytes, result: dict) -> bool:
        """Publish detection result; keep detect_pending set until success."""
        if not self._detect_pending.is_set():
            return False
        if not self._client.is_connected():
            return False

        with self._pub_lock:
            if not self._detect_pending.is_set():
                return False
            if self._do_publish(jpeg_bytes, result):
                self._detect_pending.clear()
                self.arm_retrigger()
                return True
        return False
