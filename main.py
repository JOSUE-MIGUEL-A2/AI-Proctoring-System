# main.py

import cv2
import time
from config.settings import MonitorConfig, AppConfig
from src.core.camera import CameraCapture
from src.core.detector import PoseDetector
from src.core.pose_estimator import HeadPoseEstimator
from src.core.monitor import GazeMonitor
from src.utils.overlay import draw_hud
from src.utils.audio_alert import AudioAlert
from src.utils.logger import ViolationLogger


def main():
    app_cfg = AppConfig()
    mon_cfg = MonitorConfig(
        calibration_duration_sec = 3.0,
        yaw_threshold            = 20.0,
        pitch_threshold          = 15.0,
        violation_duration_sec   = 1.5,
        smoothing_window         = 5,
    )

    detector  = PoseDetector(confidence=0.5)
    estimator = HeadPoseEstimator(
        frame_width  = app_cfg.frame_width,
        frame_height = app_cfg.frame_height,
    )
    monitor   = GazeMonitor(config=mon_cfg)
    alert     = AudioAlert(cooldown_sec=3.0)
    logger    = ViolationLogger(
        webhook_url = app_cfg.webhook_url,
        student_id  = app_cfg.student_id,
    )

    # ── FPS tracking ────────────────────────────────────────────────────
    frame_count = 0
    fps_timer   = time.perf_counter()
    current_fps = 0.0

    print("Starting AI Proctoring System… Press 'Q' to quit, 'R' to recalibrate.")

    with CameraCapture(width=app_cfg.frame_width, height=app_cfg.frame_height) as cam:
        while True:
            ok, frame = cam.read_frame()
            if not ok:
                print("[ERROR] Lost camera feed.")
                break

            # ── Resize for faster processing, keep display at full res ──
            small = cv2.resize(frame, (app_cfg.proc_width, app_cfg.proc_height))

            # ── Detection & estimation ────────────────────────────────────
            landmarks = detector.detect(small)

            if landmarks:
                # Scale keypoints back to display resolution
                scale_x = app_cfg.frame_width  / app_cfg.proc_width
                scale_y = app_cfg.frame_height / app_cfg.proc_height
                scaled = {
                    k: (v[0] * scale_x, v[1] * scale_y) if v else None
                    for k, v in landmarks.items()
                }
                result = estimator.estimate(scaled)
            else:
                result = None

            # ── Monitoring logic ───────────────────────────────────────────
            if result:
                yaw, pitch, roll = result
                print(f"Yaw: {yaw:+.1f}°  Pitch: {pitch:+.1f}°  Roll: {roll:+.1f}°", end="\r")
                status = monitor.update(yaw, pitch)
            else:
                # No face detected — use last known status or idle
                status = {
                    "phase": "monitoring" if monitor.is_calibrated else "calibrating",
                    "calibration_progress": 0.0,
                    "smoothed_yaw": 0.0, "smoothed_pitch": 0.0,
                    "yaw_delta": 0.0, "pitch_delta": 0.0,
                    "violation": False, "violation_duration_sec": 0.0,
                }

            violation = status.get("violation", False)

            # ── Alerts & logging ───────────────────────────────────────────
            if violation:
                alert.trigger()
                logger.log(frame, status)

            # ── HUD overlay ────────────────────────────────────────────────
            frame = draw_hud(frame, status, mon_cfg, violation)

            # ── FPS counter ────────────────────────────────────────────────
            frame_count += 1
            if frame_count % 30 == 0:
                now = time.perf_counter()
                current_fps = 30 / (now - fps_timer)
                fps_timer = now

            cv2.putText(frame, f"FPS: {current_fps:.1f}",
                        (frame.shape[1] - 100, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

            cv2.imshow("AI Proctoring System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting…")
                break
            if key == ord('r'):
                monitor.reset_calibration()
                print("[Calibration] Reset — look straight at the screen.")

    print("Session ended. Violations logged to logs/violations.jsonl")


if __name__ == "__main__":
    main()
    