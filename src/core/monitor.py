# src/core/monitor.py

import time
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MonitorConfig:
    # --- Calibration ---
    calibration_duration_sec: float = 3.0   # How long student must look straight

    # --- Tolerance thresholds (degrees) ---
    yaw_threshold:   float = 20.0   # Left/right head turn
    pitch_threshold: float = 15.0   # Up/down head tilt

    # --- Evaluation window (false positive prevention) ---
    violation_duration_sec: float = 1.5  # Must exceed threshold for this long

    # --- Smoothing ---
    smoothing_window: int = 5  # Rolling average over N frames


class GazeMonitor:
    """
    Orchestrates calibration, threshold checking, and violation detection.
    """

    def __init__(self, config: MonitorConfig = None):
        self.config = config or MonitorConfig()
        self.baseline_yaw:  float | None = None
        self.baseline_pitch: float | None = None

        self._calibration_readings: list[tuple[float, float]] = []
        self._calibration_start:    float | None = None
        self._is_calibrated: bool = False

        # Rolling buffers for angle smoothing
        self._yaw_buf   = deque(maxlen=self.config.smoothing_window)
        self._pitch_buf = deque(maxlen=self.config.smoothing_window)

        # Violation timing
        self._violation_start: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def reset_calibration(self):
        """Call this to restart the calibration phase."""
        self._calibration_readings.clear()
        self._calibration_start = None
        self._is_calibrated = False
        self.baseline_yaw = None
        self.baseline_pitch = None

    def update(self, yaw: float, pitch: float) -> dict:
        """
        Feed one frame's YPR values. Returns a status dict:
          {
            "phase":       "calibrating" | "monitoring",
            "calibration_progress": float (0.0–1.0, only during calibration),
            "smoothed_yaw":   float,
            "smoothed_pitch": float,
            "yaw_delta":   float,
            "pitch_delta": float,
            "violation":   bool,
            "violation_duration_sec": float,
          }
        """
        self._yaw_buf.append(yaw)
        self._pitch_buf.append(pitch)
        smooth_yaw   = float(np.mean(self._yaw_buf))
        smooth_pitch = float(np.mean(self._pitch_buf))

        if not self._is_calibrated:
            return self._run_calibration(smooth_yaw, smooth_pitch)
        else:
            return self._run_monitoring(smooth_yaw, smooth_pitch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_calibration(self, yaw: float, pitch: float) -> dict:
        now = time.monotonic()
        if self._calibration_start is None:
            self._calibration_start = now

        elapsed = now - self._calibration_start
        self._calibration_readings.append((yaw, pitch))
        progress = min(elapsed / self.config.calibration_duration_sec, 1.0)

        if elapsed >= self.config.calibration_duration_sec:
            yaws   = [r[0] for r in self._calibration_readings]
            pitches = [r[1] for r in self._calibration_readings]
            self.baseline_yaw   = float(np.mean(yaws))
            self.baseline_pitch = float(np.mean(pitches))
            self._is_calibrated = True

        return {
            "phase": "calibrating",
            "calibration_progress": progress,
            "smoothed_yaw": yaw,
            "smoothed_pitch": pitch,
            "yaw_delta": 0.0,
            "pitch_delta": 0.0,
            "violation": False,
            "violation_duration_sec": 0.0,
        }

    def _run_monitoring(self, yaw: float, pitch: float) -> dict:
        yaw_delta   = abs(yaw   - self.baseline_yaw)
        pitch_delta = abs(pitch - self.baseline_pitch)

        exceeds = (
            yaw_delta   > self.config.yaw_threshold or
            pitch_delta > self.config.pitch_threshold
        )

        now = time.monotonic()
        violation = False
        violation_duration = 0.0

        if exceeds:
            if self._violation_start is None:
                self._violation_start = now
            violation_duration = now - self._violation_start
            if violation_duration >= self.config.violation_duration_sec:
                violation = True
        else:
            self._violation_start = None

        return {
            "phase": "monitoring",
            "calibration_progress": 1.0,
            "smoothed_yaw": yaw,
            "smoothed_pitch": pitch,
            "yaw_delta": yaw_delta,
            "pitch_delta": pitch_delta,
            "violation": violation,
            "violation_duration_sec": violation_duration,
        }