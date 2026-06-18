import cv2
import numpy as np
from ultralytics import YOLO
import threading


# COCO class names that are forbidden during an exam
FORBIDDEN_CLASSES = {
    "cell phone":   "Phone detected",
    "book":         "Book detected",
    "laptop":       "Laptop detected",
    "remote":       "Remote detected",
    "keyboard":     "Keyboard detected",
    "mouse":        "Mouse detected",
    "tv":           "Screen detected",
    "monitor":      "Monitor detected",
    "earphones":    "Earphones detected",
    "headphones":   "Headphones detected",
}

# COCO numeric IDs for the classes above (for fast lookup)
# Full list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
FORBIDDEN_IDS = {
    67: "cell phone",
    73: "book",
    63: "laptop",
    65: "remote",
    66: "keyboard",
    64: "mouse",
    62: "tv",
}

CLASS_CONFIDENCE = {
    73: 0.20,   # book — very low threshold, hard to detect
    67: 0.15,   # cell phone
    63: 0.20,   # laptop
    62: 0.20,   # tv
}


class ObjectDetector:
    """
    Runs YOLOv8 object detection on each frame to flag forbidden items.
    Uses yolov8n.pt (nano) — auto-downloads ~6 MB on first run.
    """

    def __init__(
        self,
        model_path:  str   = "yolov8n.pt",
        confidence:  float = 0.45,
        proc_width:  int   = 416,
        proc_height: int   = 416,
    ):
        self.model       = YOLO(model_path)
        self.confidence  = confidence
        self.proc_width  = proc_width
        self.proc_height = proc_height

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Runs detection on a BGR frame.

        Returns a list of violation dicts:
          {
            "label":      str,   # e.g. "Phone detected"
            "class_name": str,   # e.g. "cell phone"
            "confidence": float,
            "box":        (x1, y1, x2, y2)  # pixel coords on original frame
          }
        """
        h, w = frame.shape[:2]

        # Resize for faster inference
        small = cv2.resize(frame, (self.proc_width, self.proc_height))
        results = self.model(small, verbose=False, conf=self.confidence)

        if not results or results[0].boxes is None:
            return []

        scale_x = w / self.proc_width
        scale_y = h / self.proc_height

        violations = []
        boxes = results[0].boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in FORBIDDEN_IDS:
                continue

            class_name = FORBIDDEN_IDS[cls_id]
            conf       = float(box.conf[0])

            # Use per-class threshold if defined, else fall back to global
            min_conf = CLASS_CONFIDENCE.get(cls_id, self.confidence)
            if conf < min_conf:
                continue

            # Scale box back to original frame size
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            violations.append({
                "label":      FORBIDDEN_CLASSES.get(class_name, class_name),
                "class_name": class_name,
                "confidence": conf,
                "box":        (x1, y1, x2, y2),
            })

        return violations

    def draw_violations(
        self,
        frame: np.ndarray,
        violations: list[dict],
    ) -> np.ndarray:
        """
        Draws bright orange bounding boxes with labels on detected forbidden items.
        """
        for v in violations:
            x1, y1, x2, y2 = v["box"]
            conf  = v["confidence"]
            label = f"{v['label']} ({conf:.0%})"

            # Orange box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - 10),
                (x1 + tw + 8, y1),
                (0, 140, 255),
                -1,
            )
            cv2.putText(
                frame, label,
                (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return frame
class ObjectDetectorAsync:
    """
    Wraps ObjectDetector to run inference in a background thread.
    The video loop always gets the latest cached result instantly.
    """

    def __init__(self, **kwargs):
        self._detector    = ObjectDetector(**kwargs)
        self._violations  = []
        self._lock        = threading.Lock()
        self._running     = False
        self._frame       = None
        self._frame_lock  = threading.Lock()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit_frame(self, frame: np.ndarray):
        """Feed the latest frame. Non-blocking — drops if worker is busy."""
        with self._frame_lock:
            self._frame = frame.copy()

    def get_violations(self) -> list[dict]:
        """Returns the latest cached violation list. Always instant."""
        with self._lock:
            return list(self._violations)

    def draw_violations(self, frame, violations):
        return self._detector.draw_violations(frame, violations)

    def _worker(self):
        while True:
            frame = None
            with self._frame_lock:
                if self._frame is not None:
                    frame = self._frame
                    self._frame = None

            if frame is not None:
                result = self._detector.detect(frame)
                with self._lock:
                    self._violations = result
            else:
                threading.Event().wait(0.01)  # sleep 10ms if no frame queue
