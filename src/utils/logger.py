# src/utils/logger.py

import os
import json
import time
import threading
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path


class ViolationLogger:
    """
    Logs violations to a local JSON file and optionally sends them
    to a remote webhook endpoint.
    """

    def __init__(
        self,
        log_path:      str = "logs/violations.jsonl",
        snapshot_dir:  str = "assets/snapshots",
        webhook_url:   str | None = None,
        student_id:    str = "unknown",
    ):
        self.log_path     = Path(log_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.webhook_url  = webhook_url
        self.student_id   = student_id

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._cooldown_sec = 5.0
        self._last_logged  = 0.0

    def log(self, frame: np.ndarray, status: dict):
        """
        Call this every frame a violation is active.
        Internally rate-limits to avoid duplicate entries.
        """
        now = time.monotonic()
        if now - self._last_logged < self._cooldown_sec:
            return
        self._last_logged = now

        timestamp = datetime.now(timezone.utc).isoformat()
        snap_name = f"violation_{timestamp.replace(':', '-').replace('.', '-')}.jpg"
        snap_path = self.snapshot_dir / snap_name

        # Save snapshot in background so we don't block the video loop
        frame_copy = frame.copy()
        threading.Thread(
            target=self._save_and_send,
            args=(frame_copy, snap_path, timestamp, status, snap_name),
            daemon=True,
        ).start()
    
    def log_object(self, frame: np.ndarray, violations: list[dict]):
        """
        Logs foreign object detections separately from gaze violations.
        Rate-limited to once per 5 seconds per unique object class.
        """
        now = time.monotonic()

        # Per-class cooldown tracking
        if not hasattr(self, "_obj_last_logged"):
            self._obj_last_logged = {}

        for v in violations:
            cls = v["class_name"]
            if now - self._obj_last_logged.get(cls, 0) < self._cooldown_sec:
                continue
            self._obj_last_logged[cls] = now

            timestamp = datetime.now(timezone.utc).isoformat()
            snap_name = f"object_{cls}_{timestamp.replace(':', '-').replace('.', '-')}.jpg"
            snap_path = self.snapshot_dir / snap_name

            frame_copy = frame.copy()
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
                target=self._save_object_record,
                args=(frame_copy, snap_path, record),
                daemon=True,
            ).start()

    def _save_object_record(self, frame, snap_path, record):
        cv2.imwrite(str(snap_path), frame)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[Logger] Object violation: {record['label']} @ {record['timestamp']}")
        if self.webhook_url:
            self._send_webhook(record)

    # ------------------------------------------------------------------

    def _save_and_send(self, frame, snap_path, timestamp, status, snap_name):
        # 1. Save snapshot
        cv2.imwrite(str(snap_path), frame)

        # 2. Build log record
        record = {
            "timestamp":    timestamp,
            "student_id":   self.student_id,
            "snapshot":     str(snap_path),
            "yaw_delta":    round(status.get("yaw_delta", 0), 2),
            "pitch_delta":  round(status.get("pitch_delta", 0), 2),
            "duration_sec": round(status.get("violation_duration_sec", 0), 2),
        }

        # 3. Append to local JSONL file
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"[Logger] Violation logged @ {timestamp}  →  {snap_path}")

        # 4. Send webhook (if configured)
        if self.webhook_url:
            self._send_webhook(record)

    def _send_webhook(self, record: dict):
        try:
            import requests
            resp = requests.post(self.webhook_url, json=record, timeout=5)
            resp.raise_for_status()
            print(f"[Logger] Webhook sent: HTTP {resp.status_code}")
        except Exception as e:
            print(f"[Logger] Webhook failed: {e}")