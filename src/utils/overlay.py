# src/utils/overlay.py

import cv2
import numpy as np


def draw_hud(
    frame: np.ndarray,
    status: dict,
    config,
    violation: bool,
) -> np.ndarray:
    """
    Draws the heads-up display (HUD) onto the frame.
    Includes calibration progress bar, angle readouts, and violation banner.
    """
    h, w = frame.shape[:2]

    # ── Calibration phase ───────────────────────────────────────────────
    if status["phase"] == "calibrating":
        progress = status["calibration_progress"]
        bar_w = int(w * 0.6)
        bar_h = 20
        bar_x = (w - bar_w) // 2
        bar_y = h // 2 + 40

        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 50), (w, h // 2 + 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame,
                    "CALIBRATING — Look straight at the screen",
                    (w // 2 - 280, h // 2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)

        # Progress bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), -1)
        # Progress bar fill
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      (0, 200, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (200, 200, 200), 1)
        return frame

    # ── Monitoring phase ─────────────────────────────────────────────────
    yaw   = status["smoothed_yaw"]
    pitch = status["smoothed_pitch"]
    dy    = status["yaw_delta"]
    dp    = status["pitch_delta"]

    # Angle readout (top-left)
    def put(text, y, color=(200, 200, 200)):
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    put(f"Yaw:   {yaw:+.1f}°  (delta {dy:.1f}° / max {config.yaw_threshold}°)",
        25, color=(0, 200, 255) if dy > config.yaw_threshold else (200, 200, 200))
    put(f"Pitch: {pitch:+.1f}°  (delta {dp:.1f}° / max {config.pitch_threshold}°)",
        50, color=(0, 200, 255) if dp > config.pitch_threshold else (200, 200, 200))

    vd = status["violation_duration_sec"]
    if vd > 0:
        put(f"Away for: {vd:.1f}s  (limit {config.violation_duration_sec}s)",
            75, color=(0, 100, 255))

    # ── Violation banner ──────────────────────────────────────────────────
    if violation:
        # Red flashing border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)

        # Warning banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 45), (w, h // 2 + 45), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame,
                    "⚠  WARNING: LOOK AT THE SCREEN",
                    (w // 2 - 265, h // 2 + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # Green "OK" indicator
        cv2.circle(frame, (w - 20, 20), 10, (0, 200, 0), -1)

    return frame