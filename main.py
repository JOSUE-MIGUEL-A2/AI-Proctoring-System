import cv2
import time
from config.settings import MonitorConfig, AppConfig
from src.core.camera import CameraCapture
from src.core.detector import PoseDetector
from src.core.pose_estimator import HeadPoseEstimator
from src.core.monitor import GazeMonitor
from src.core.object_detector import ObjectDetectorAsync
from src.core.gaze_tracker import GazeTracker
from src.core.face_verifier import FaceVerifier
from src.utils.overlay import (
    draw_hud,
    draw_object_alert,
    draw_gaze_status,
    draw_face_status,
)
from src.utils.audio_alert import AudioAlert
from src.utils.logger import ViolationLogger


# Session phases
PHASE_ENROLLING   = "enrolling"
PHASE_CALIBRATING = "calibrating"
PHASE_MONITORING  = "monitoring"


def main():
    app_cfg = AppConfig()
    mon_cfg = MonitorConfig(
        calibration_duration_sec = 3.0,
        yaw_threshold            = 25.0,
        pitch_threshold          = 20.0,
        violation_duration_sec   = 2.0,
        smoothing_window         = 5,
    )

    # Core components
    detector  = PoseDetector(confidence=0.5)
    estimator = HeadPoseEstimator(
        frame_width  = app_cfg.frame_width,
        frame_height = app_cfg.frame_height,
    )
    monitor      = GazeMonitor(config=mon_cfg)
    alert        = AudioAlert(cooldown_sec=3.0)
    logger       = ViolationLogger(
        webhook_url = app_cfg.webhook_url,
        student_id  = app_cfg.student_id,
    )
    obj_detector = ObjectDetectorAsync(
        model_path  = "yolov8m.pt",
        confidence  = 0.45,
        proc_width  = app_cfg.proc_width,
        proc_height = app_cfg.proc_height,
    )
    gaze_tracker  = GazeTracker(violation_duration_sec=1.5)
    face_verifier = FaceVerifier(student_id=app_cfg.student_id)

    # Session state
    session_phase  = PHASE_ENROLLING if not face_verifier.is_enrolled else PHASE_CALIBRATING
    face_result    = {"status": "not_enrolled"}
    gaze_status    = {"zone": "CENTER", "violation": False, "duration_sec": 0.0}
    obj_violations = []

    # FPS tracking
    frame_count = 0
    fps_timer   = time.perf_counter()
    current_fps = 0.0

    print("Starting AI Proctoring System...")
    if session_phase == PHASE_ENROLLING:
        print("ENROLLMENT: Look at the camera for 3 seconds to register your face.")
    print("Controls: Q = quit | R = recalibrate | E = re-enroll face")

    with CameraCapture(width=app_cfg.frame_width, height=app_cfg.frame_height) as cam:
        while True:
            ok, frame = cam.read_frame()
            if not ok:
                print("[ERROR] Lost camera feed.")
                break

            small = cv2.resize(frame, (app_cfg.proc_width, app_cfg.proc_height))

            # ── PHASE: Face enrollment ────────────────────────────────
            if session_phase == PHASE_ENROLLING:
                enroll_result = face_verifier.enroll(frame)
                face_result   = enroll_result

                if enroll_result["status"] == "complete":
                    print("[Enrollment] Complete! Starting calibration...")
                    session_phase = PHASE_CALIBRATING

                # Draw enrollment screen
                h, w  = frame.shape[:2]
                prog  = enroll_result.get("progress", 0)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h // 2 - 60), (w, h // 2 + 100), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame,
                            "ENROLLMENT: Hold still and look at the camera",
                            (w // 2 - 310, h // 2 - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                bar_w = int(w * 0.6)
                bar_x = (w - bar_w) // 2
                bar_y = h // 2 + 20
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (60, 60, 60), -1)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + int(bar_w * prog), bar_y + 20), (0, 180, 255), -1)

                cv2.imshow("AI Proctoring System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ── Pose detection ────────────────────────────────────────
            landmarks = detector.detect(small)

            if landmarks:
                scale_x = app_cfg.frame_width  / app_cfg.proc_width
                scale_y = app_cfg.frame_height / app_cfg.proc_height
                scaled  = {
                    k: (v[0] * scale_x, v[1] * scale_y) if v else None
                    for k, v in landmarks.items()
                }
                result = estimator.estimate(scaled)
            else:
                result = None
                scaled = None

            # Head pose monitoring
            if result:
                yaw, pitch, roll = result
                print(f"Yaw: {yaw:+.1f}  Pitch: {pitch:+.1f}  Roll: {roll:+.1f}", end="\r")
                status = monitor.update(yaw, pitch)
                if session_phase == PHASE_CALIBRATING and monitor.is_calibrated:
                    session_phase = PHASE_MONITORING
                    print("\n[Calibration] Complete! Monitoring started.")
            else:
                status = {
                    "phase": "monitoring" if monitor.is_calibrated else "calibrating",
                    "calibration_progress": 0.0,
                    "smoothed_yaw": 0.0, "smoothed_pitch": 0.0,
                    "yaw_delta": 0.0, "pitch_delta": 0.0,
                    "violation": False, "violation_duration_sec": 0.0,
                }

            head_violation = status.get("violation", False)

            # ── Eye gaze tracking ─────────────────────────────────────
            if session_phase == PHASE_MONITORING and scaled:
                gaze_status = gaze_tracker.process(frame, scaled)
                if gaze_status["violation"]:
                    alert.trigger()
                    logger.log_eye_gaze(frame, gaze_status)

            # ── Face verification ─────────────────────────────────────
            if session_phase == PHASE_MONITORING:
                face_result = face_verifier.verify(frame)
                if face_result.get("status") == "mismatch":
                    alert.trigger()
                    logger.log_face_mismatch(frame, face_result)

            # ── Head pose alerts ──────────────────────────────────────
            if head_violation:
                alert.trigger()
                logger.log(frame, status)

            # ── Object detection ──────────────────────────────────────
            obj_detector.submit_frame(frame)
            obj_violations = obj_detector.get_violations()
            if obj_violations:
                frame = obj_detector.draw_violations(frame, obj_violations)
                logger.log_object(frame, obj_violations)
                alert.trigger()

            # ── HUD drawing ───────────────────────────────────────────
            frame = draw_hud(frame, status, mon_cfg, head_violation)
            frame = draw_gaze_status(frame, gaze_status)
            frame = draw_face_status(frame, face_result)
            frame = draw_object_alert(frame, obj_violations)

            # ── FPS counter ───────────────────────────────────────────
            frame_count += 1
            if frame_count % 30 == 0:
                now = time.perf_counter()
                current_fps = 30 / (now - fps_timer)
                fps_timer   = now

            cv2.putText(frame, f"FPS: {current_fps:.1f}",
                        (frame.shape[1] - 100, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

            cv2.imshow("AI Proctoring System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting...")
                break
            if key == ord('r'):
                monitor.reset_calibration()
                session_phase = PHASE_CALIBRATING
                print("\n[Calibration] Reset - look straight at the screen.")
            if key == ord('e'):
                face_verifier.reset_enrollment()
                session_phase = PHASE_ENROLLING
                print("\n[Enrollment] Reset - look at the camera to re-enroll.")

    print("Session ended. Violations logged to logs/violations.jsonl")


if __name__ == "__main__":
    main()