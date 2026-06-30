import cv2
import numpy as np


class GazeTracker:
    """
    Tracks iris position within the eye region to detect if the student
    is looking away even while keeping their head relatively still.

    Uses OpenCV's blob detection on the eye ROI — no extra libraries needed.

    Gaze zones:
      CENTER  — looking at the screen (OK)
      LEFT    — looking left of screen
      RIGHT   — looking right of screen
      UP      — looking up (e.g. at notes above screen)
      DOWN    — looking down (e.g. at phone on desk)
    """

    # Gaze deviation thresholds (ratio of eye width/height)
    HORIZONTAL_THRESHOLD = 0.20   # iris center > 20% from eye center = left/right
    VERTICAL_THRESHOLD   = 0.15   # iris center > 15% from eye center = up/down

    def __init__(self, violation_duration_sec: float = 1.5):
        self.violation_duration_sec = violation_duration_sec
        self._violation_start = None
        self._last_zone = "CENTER"

        # Blob detector tuned for iris detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea        = True
        params.minArea             = 30
        params.maxArea             = 1500
        params.filterByCircularity = True
        params.minCircularity      = 0.4
        params.filterByConvexity   = True
        params.minConvexity        = 0.7
        params.filterByInertia     = True
        params.minInertiaRatio     = 0.3
        self._blob_detector = cv2.SimpleBlobDetector_create(params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray, landmarks: dict) -> dict:
        """
        Analyzes eye regions from the frame using YOLO landmarks.

        Returns:
          {
            "zone":         str,    CENTER | LEFT | RIGHT | UP | DOWN
            "violation":    bool,
            "duration_sec": float,
            "left_ratio":   (float, float) | None,   (h_ratio, v_ratio)
            "right_ratio":  (float, float) | None,
          }
        """
        import time

        left_ratio  = self._analyze_eye(frame, landmarks, "left")
        right_ratio = self._analyze_eye(frame, landmarks, "right")

        zone = self._determine_zone(left_ratio, right_ratio)
        self._last_zone = zone

        now = time.monotonic()
        violation = False
        duration  = 0.0

        if zone != "CENTER":
            if self._violation_start is None:
                self._violation_start = now
            duration = now - self._violation_start
            if duration >= self.violation_duration_sec:
                violation = True
        else:
            self._violation_start = None

        return {
            "zone":         zone,
            "violation":    violation,
            "duration_sec": duration,
            "left_ratio":   left_ratio,
            "right_ratio":  right_ratio,
        }

    def draw_debug(self, frame: np.ndarray, landmarks: dict, status: dict) -> np.ndarray:
        """Draws eye ROI boxes and gaze zone label for debugging."""
        h, w = frame.shape[:2]
        zone     = status["zone"]
        violated = status["violation"]

        color = (0, 200, 0) if zone == "CENTER" else (0, 100, 255)
        label = f"Gaze: {zone}"
        if violated:
            label += " (!)"
            color = (0, 0, 220)

        cv2.putText(frame, label, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # Draw eye ROI boxes
        for side in ("left", "right"):
            roi = self._get_eye_roi(frame, landmarks, side)
            if roi is not None:
                x, y, ew, eh = roi["rect"]
                cv2.rectangle(frame, (x, y), (x + ew, y + eh), color, 1)

        return frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_eye(self, frame, landmarks, side: str):
        """
        Returns (h_ratio, v_ratio) where each value is the normalized
        distance of the iris from the eye center (-1.0 to +1.0).
        Returns None if the eye region cannot be found.
        """
        roi_data = self._get_eye_roi(frame, landmarks, side)
        if roi_data is None:
            return None

        roi      = roi_data["roi"]
        x, y, ew, eh = roi_data["rect"]

        # Preprocess: grayscale, blur, threshold for dark iris
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

        # Find iris center via blob detection
        keypoints = self._blob_detector.detect(thresh)

        if not keypoints:
            # Fallback: use darkest region centroid
            iris_center = self._darkest_region_center(blurred)
        else:
            # Use largest blob
            largest = max(keypoints, key=lambda k: k.size)
            iris_center = (int(largest.pt[0]), int(largest.pt[1]))

        if iris_center is None:
            return None

        # Compute ratio: 0 = eye center, -1 = far left/up, +1 = far right/down
        eye_cx  = ew / 2
        eye_cy  = eh / 2
        h_ratio = (iris_center[0] - eye_cx) / (ew / 2 + 1e-6)
        v_ratio = (iris_center[1] - eye_cy) / (eh / 2 + 1e-6)

        return (h_ratio, v_ratio)

    def _get_eye_roi(self, frame, landmarks, side: str):
        """
        Extracts the eye region from the frame using YOLO landmarks.
        Returns dict with 'roi' and 'rect', or None if landmarks missing.
        """
        h, w = frame.shape[:2]

        eye_key = f"{side}_eye"
        ear_key = f"{side}_ear"

        eye_pt = landmarks.get(eye_key)
        ear_pt = landmarks.get(ear_key)

        if eye_pt is None:
            return None

        ex, ey = int(eye_pt[0]), int(eye_pt[1])

        # Estimate eye width from ear distance if available
        if ear_pt is not None:
            ear_dist = abs(ex - int(ear_pt[0]))
            ew = max(int(ear_dist * 0.6), 30)
        else:
            ew = 40   # default fallback

        eh = max(int(ew * 0.5), 20)

        # Crop eye ROI with boundary clamping
        x1 = max(ex - ew // 2, 0)
        y1 = max(ey - eh // 2, 0)
        x2 = min(ex + ew // 2, w)
        y2 = min(ey + eh // 2, h)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        return {
            "roi":  frame[y1:y2, x1:x2].copy(),
            "rect": (x1, y1, x2 - x1, y2 - y1),
        }

    def _determine_zone(self, left_ratio, right_ratio) -> str:
        """Combines left and right eye ratios into a single gaze zone."""
        ratios = [r for r in [left_ratio, right_ratio] if r is not None]
        if not ratios:
            return "CENTER"

        avg_h = sum(r[0] for r in ratios) / len(ratios)
        avg_v = sum(r[1] for r in ratios) / len(ratios)

        ht = self.HORIZONTAL_THRESHOLD
        vt = self.VERTICAL_THRESHOLD

        if abs(avg_h) > ht:
            return "RIGHT" if avg_h > 0 else "LEFT"
        if abs(avg_v) > vt:
            return "DOWN" if avg_v > 0 else "UP"

        return "CENTER"

    @staticmethod
    def _darkest_region_center(gray_roi: np.ndarray):
        """Finds the center of the darkest region — fallback iris locator."""
        _, _, _, max_loc = cv2.minMaxLoc(gray_roi)
        # minMaxLoc returns (minVal, maxVal, minLoc, maxLoc)
        # We want the darkest (min) location
        min_val, _, min_loc, _ = cv2.minMaxLoc(gray_roi)
        if min_val < 80:   # only trust if it's actually dark
            return min_loc
        return None