# scripts/test_conditions.py
"""
Run this script to measure FPS and detection rate under different conditions.
Press 'q' to quit, 'r' to reset stats.
"""

import cv2
import time
import numpy as np
from src.core.camera import CameraCapture
from src.core.detector import PoseDetector
from src.core.pose_estimator import HeadPoseEstimator

detector = PoseDetector(confidence=0.4)  # Lower conf for low-light testing
estimator = HeadPoseEstimator()

frame_times  = []
detect_count = 0
total_frames = 0

with CameraCapture() as cam:
    while True:
        t0 = time.perf_counter()
        ok, frame = cam.read_frame()
        if not ok:
            break

        landmarks = detector.detect(frame)
        total_frames += 1

        if landmarks:
            detect_count += 1
            result = estimator.estimate(landmarks)
            if result:
                yaw, pitch, roll = result
                cv2.putText(frame,
                    f"Yaw:{yaw:+.1f}  Pitch:{pitch:+.1f}  Roll:{roll:+.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)

        avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
        detection_rate = (detect_count / total_frames) * 100 if total_frames else 0

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Detection: {detection_rate:.0f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Condition Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            frame_times.clear()
            detect_count = total_frames = 0

cv2.destroyAllWindows()
print(f"Final avg FPS: {avg_fps:.1f} | Detection rate: {detection_rate:.1f}%")