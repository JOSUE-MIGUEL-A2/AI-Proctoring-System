# Thresholds and config loaded here
# config/settings.py
from dataclasses import dataclass


@dataclass
class AppConfig:
    # Camera resolution for display
    frame_width:  int = 640
    frame_height: int = 480

    # Smaller resolution for YOLO inference — significantly boosts FPS
    proc_width:  int = 416
    proc_height: int = 416

    # Optional webhook for remote logging (set to None to disable)
    webhook_url: str | None = None   # e.g. "https://your-backend.com/api/violations"

    # Student identifier for logging
    student_id: str = "student_001"


@dataclass
class MonitorConfig:
    calibration_duration_sec: float = 3.0
    yaw_threshold:            float = 20.0
    pitch_threshold:          float = 15.0
    violation_duration_sec:   float = 1.5
    smoothing_window:         int   = 5