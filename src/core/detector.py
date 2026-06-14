# src/core/detector.py

import numpy as np
from ultralytics import YOLO


# COCO keypoint indices for facial landmarks
KEYPOINT_INDICES = {
    "nose":      0,
    "left_eye":  1,
    "right_eye": 2,
    "left_ear":  3,
    "right_ear": 4,
}


class PoseDetector:
    """
    Wraps YOLOv8-Pose to extract facial keypoints per frame.
    """

    def __init__(self, model_path: str = "yolov8n-pose.pt", confidence: float = 0.5):
        # Model auto-downloads to ~/.cache/ultralytics on first run
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        Runs inference on a single BGR frame.

        Returns a dict with keys matching KEYPOINT_INDICES (e.g. 'nose', 'left_eye')
        where each value is (x, y) in pixel coords, or None if no person detected.
        Only returns the highest-confidence detection when multiple people are present.
        """
        results = self.model(frame, verbose=False, conf=self.confidence)

        if not results or results[0].keypoints is None:
            return None

        keypoints_data = results[0].keypoints

        # Pick the person with highest box confidence
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_idx = int(boxes.conf.argmax())
        kp_xy = keypoints_data.xy[best_idx].cpu().numpy()  # shape: (17, 2)

        # Extract only the 5 facial keypoints
        landmarks = {}
        for name, idx in KEYPOINT_INDICES.items():
            x, y = kp_xy[idx]
            # Treat (0, 0) as undetected — YOLO returns zeros for occluded points
            landmarks[name] = (float(x), float(y)) if (x > 0 and y > 0) else None

        # Require at least nose + both eyes to compute head pose reliably
        required = ["nose", "left_eye", "right_eye"]
        if any(landmarks[k] is None for k in required):
            return None

        return landmarks

    def draw_keypoints(self, frame: np.ndarray, landmarks: dict) -> np.ndarray:
        """Draws detected facial keypoints onto the frame for debugging."""
        import cv2
        colors = {
            "nose":      (0, 255, 255),
            "left_eye":  (255, 0, 0),
            "right_eye": (0, 0, 255),
            "left_ear":  (255, 255, 0),
            "right_ear": (0, 255, 0),
        }
        for name, pt in landmarks.items():
            if pt is not None:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, colors[name], -1)
                cv2.putText(frame, name, (int(pt[0]) + 6, int(pt[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1)
        return frame