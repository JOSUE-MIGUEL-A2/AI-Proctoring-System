import json
import time
import threading
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path


class ViolationLogger:
    """
    Logs all violation types to a local JSONL file and optionally
    sends them to a remote webhook endpoint.

    Violation types logged:
      - gaze_violation      (head pose deviation)
      - gaze_eye_violation  (iris gaze deviation)
      - foreign_object      (forbidden item detected)
      - face_mismatch       (wrong person at camera)
    """

    def __init__(
        self,
        log_path:     str = "logs/violations.jsonl",
        snapshot_dir: str = "assets/snapshots",
        webhook_url:  str | None = None,
        student_id:   str = "unknown",
    ):
        self.log_path     = Path(log_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.webhook_url  = webhook_url
        self.student_id   = student_id

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._cooldown_sec      = 5.0
        self._last_logged       = 0.0
        self._obj_last_logged   = {}
        self._face_last_logged  = 0.0
        self._gaze_last_logged  = 0.0

    # ------------------------------------------------------------------
    # Head pose gaze violation
    # ------------------------------------------------------------------

    def log(self, frame: np.ndarray, status: dict):
        now = time.monotonic()
        if now - self._last_logged < self._cooldown_sec:
            return
        self._last_logged = now

        timestamp = self._timestamp()
        snap_path = self.snapshot_dir / f"gaze_{self._safe_ts(timestamp)}.jpg"

        record = {
            "type":         "gaze_violation",
            "timestamp":    timestamp,
            "student_id":   self.student_id,
            "snapshot":     str(snap_path),
            "yaw_delta":    round(status.get("yaw_delta", 0), 2),
            "pitch_delta":  round(status.get("pitch_delta", 0), 2),
            "duration_sec": round(status.get("violation_duration_sec", 0), 2),
        }

        threading.Thread(
            target=self._write,
            args=(frame.copy(), snap_path, record),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Eye gaze violation
    # ------------------------------------------------------------------

    def log_eye_gaze(self, frame: np.ndarray, gaze_status: dict):
        now = time.monotonic()
        if now - self._gaze_last_logged < self._cooldown_sec:
            return
        self._gaze_last_logged = now

        timestamp = self._timestamp()
        snap_path = self.snapshot_dir / f"eye_gaze_{self._safe_ts(timestamp)}.jpg"

        record = {
            "type":         "gaze_eye_violation",
            "timestamp":    timestamp,
            "student_id":   self.student_id,
            "snapshot":     str(snap_path),
            "zone":         gaze_status.get("zone", "UNKNOWN"),
            "duration_sec": round(gaze_status.get("duration_sec", 0), 2),
        }

        threading.Thread(
            target=self._write,
            args=(frame.copy(), snap_path, record),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Foreign object violation
    # ------------------------------------------------------------------

    def log_object(self, frame: np.ndarray, violations: list[dict]):
        now = time.monotonic()

        for v in violations:
            cls = v["class_name"]
            if now - self._obj_last_logged.get(cls, 0) < self._cooldown_sec:
                continue
            self._obj_last_logged[cls] = now

            timestamp = self._timestamp()
            snap_path = self.snapshot_dir / f"object_{cls}_{self._safe_ts(timestamp)}.jpg"

            record = {
                "type":       "foreign_object",
                "timestamp":  timestamp,
                "student_id": self.student_id,
                "object":     cls,
                "label":      v["label"],
                "confidence": round(v["confidence"], 3),
                "snapshot":   str(snap_path),
            }

            threading.Thread(
                target=self._write,
                args=(frame.copy(), snap_path, record),
                daemon=True,
            ).start()

    # ------------------------------------------------------------------
    # Face mismatch violation
    # ------------------------------------------------------------------

    def log_face_mismatch(self, frame: np.ndarray, face_result: dict):
        now = time.monotonic()
        if now - self._face_last_logged < self._cooldown_sec * 2:
            return
        self._face_last_logged = now

        timestamp = self._timestamp()
        snap_path = self.snapshot_dir / f"face_mismatch_{self._safe_ts(timestamp)}.jpg"

        record = {
            "type":       "face_mismatch",
            "timestamp":  timestamp,
            "student_id": self.student_id,
            "snapshot":   str(snap_path),
            "confidence": face_result.get("confidence", 0),
        }

        threading.Thread(
            target=self._write,
            args=(frame.copy(), snap_path, record),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    def _write(self, frame, snap_path, record):
        cv2.imwrite(str(snap_path), frame)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[Logger] {record['type']} @ {record['timestamp']}")
        if self.webhook_url:
            self._send_webhook(record)

    def _send_webhook(self, record: dict):
        try:
            import requests
            resp = requests.post(self.webhook_url, json=record, timeout=5)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Logger] Webhook failed: {e}")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_ts(ts: str) -> str:
        return ts.replace(":", "-").replace(".", "-")