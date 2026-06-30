import cv2
import numpy as np


def draw_hud(
    frame: np.ndarray,
    status: dict,
    config,
    violation: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Calibration phase
    if status["phase"] == "calibrating":
        progress = status["calibration_progress"]
        bar_w = int(w * 0.6)
        bar_h = 20
        bar_x = (w - bar_w) // 2
        bar_y = h // 2 + 40

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 50), (w, h // 2 + 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame,
                    "CALIBRATING - Look straight at the screen",
                    (w // 2 - 280, h // 2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), -1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      (0, 200, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (200, 200, 200), 1)
        return frame

    # Monitoring phase
    yaw   = status["smoothed_yaw"]
    pitch = status["smoothed_pitch"]
    dy    = status["yaw_delta"]
    dp    = status["pitch_delta"]

    def put(text, y, color=(200, 200, 200)):
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    put(f"Yaw:   {yaw:+.1f} deg  (delta {dy:.1f} / max {config.yaw_threshold})",
        25, color=(0, 200, 255) if dy > config.yaw_threshold else (200, 200, 200))
    put(f"Pitch: {pitch:+.1f} deg  (delta {dp:.1f} / max {config.pitch_threshold})",
        50, color=(0, 200, 255) if dp > config.pitch_threshold else (200, 200, 200))

    vd = status["violation_duration_sec"]
    if vd > 0:
        put(f"Away for: {vd:.1f}s  (limit {config.violation_duration_sec}s)",
            75, color=(0, 100, 255))

    if violation:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 45), (w, h // 2 + 45), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame,
                    "!! WARNING: LOOK AT THE SCREEN",
                    (w // 2 - 245, h // 2 + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.circle(frame, (w - 20, 20), 10, (0, 200, 0), -1)

    return frame


def draw_object_alert(frame: np.ndarray, violations: list[dict]) -> np.ndarray:
    if not violations:
        return frame

    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 120, 220), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    names = ", ".join(set(v["label"] for v in violations))
    cv2.putText(frame, f"OBJECT ALERT: {names}",
                (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_gaze_status(frame: np.ndarray, gaze_status: dict) -> np.ndarray:
    zone      = gaze_status.get("zone", "CENTER")
    violation = gaze_status.get("violation", False)
    duration  = gaze_status.get("duration_sec", 0.0)

    color = (0, 200, 0) if zone == "CENTER" else (0, 100, 255)
    if violation:
        color = (0, 0, 220)

    label = f"Gaze: {zone}"
    if duration > 0:
        label += f"  ({duration:.1f}s)"

    cv2.putText(frame, label, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    if violation:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 + 50), (w, h // 2 + 95), (0, 0, 160), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame,
                    "!! GAZE WARNING: Eyes off screen",
                    (w // 2 - 210, h // 2 + 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_face_status(frame: np.ndarray, face_result: dict) -> np.ndarray:
    status = face_result.get("status", "pending")

    if status == "verified":
        color = (0, 180, 0)
        label = "ID: Verified"
    elif status == "mismatch":
        conf  = face_result.get("confidence", 0)
        color = (0, 0, 220)
        label = f"ID: MISMATCH ({conf:.0f})"
        h, w  = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 + 100), (w, h // 2 + 145), (0, 0, 130), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame,
                    "!! ID MISMATCH: Wrong student detected",
                    (w // 2 - 230, h // 2 + 132),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    elif status == "no_face":
        color = (0, 140, 255)
        label = "ID: No face"
    elif status == "not_enrolled":
        color = (150, 150, 150)
        label = "ID: Not enrolled"
    elif status == "collecting":
        prog  = face_result.get("progress", 0)
        color = (200, 200, 0)
        label = f"Enrolling... {prog:.0%}"
    elif status == "complete":
        color = (0, 200, 100)
        label = "Enrollment complete!"
    else:
        color = (150, 150, 150)
        label = "ID: --"

    cv2.putText(frame, label, (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return frame