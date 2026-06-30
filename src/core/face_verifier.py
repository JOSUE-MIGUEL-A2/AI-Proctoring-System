# src/core/face_verifier.py

import cv2
import os
import numpy as np
import pickle
from pathlib import Path


class FaceVerifier:
    """
    Verifies that the person in frame matches the registered student.

    Workflow:
      1. enroll(frame, student_id)  — called once at session start to save the face embedding
      2. verify(frame)              — called periodically during the session

    Uses OpenCV's LBPH (Local Binary Pattern Histogram) face recognizer.
    No internet required, no API keys, runs fully offline.

    LBPH is chosen over deep learning embeddings (DeepFace, face_recognition)
    because it requires no extra pip installs beyond opencv-contrib-python.
    """

    ENROLLMENT_DIR = Path("assets/enrolled_faces")
    MIN_CONFIDENCE = 80.0    # LBPH distance — lower = more similar. >80 = likely different person
    VERIFY_EVERY_N = 60      # Verify identity every N frames (not every frame)

    def __init__(self, student_id: str = "student_001"):
        self.student_id   = student_id
        self._recognizer  = cv2.face.LBPHFaceRecognizer_create()
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._enrolled    = False
        self._model_path  = self.ENROLLMENT_DIR / f"{student_id}_model.yml"
        self._frame_count = 0
        self._last_result = {"verified": True, "confidence": 0.0, "status": "pending"}

        self.ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)

        # Auto-load existing enrollment if available
        if self._model_path.exists():
            self._recognizer.read(str(self._model_path))
            self._enrolled = True
            print(f"[FaceVerifier] Loaded existing enrollment for {student_id}")

    @property
    def is_enrolled(self) -> bool:
        return self._enrolled

    def enroll(self, frame: np.ndarray, n_samples: int = 20) -> dict:
        """
        Collects face samples from a single frame and adds to enrollment.
        Call this repeatedly during the enrollment phase (e.g. 3 seconds of frames).

        Returns:
          { "status": "collecting" | "complete" | "no_face", "progress": float }
        """
        if not hasattr(self, "_enrollment_faces"):
            self._enrollment_faces = []

        faces = self._detect_faces(frame)
        if not faces:
            return {"status": "no_face", "progress": len(self._enrollment_faces) / n_samples}

        # Use the largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        face_roi = self._extract_roi(frame, largest)

        self._enrollment_faces.append(face_roi)

        progress = len(self._enrollment_faces) / n_samples

        if len(self._enrollment_faces) >= n_samples:
            # Train the recognizer on collected samples
            labels = [0] * len(self._enrollment_faces)
            self._recognizer.train(self._enrollment_faces, np.array(labels))
            self._recognizer.save(str(self._model_path))
            self._enrolled = True
            self._enrollment_faces = []
            print(f"[FaceVerifier] Enrollment complete for {self.student_id}")
            return {"status": "complete", "progress": 1.0}

        return {"status": "collecting", "progress": progress}

    def verify(self, frame: np.ndarray) -> dict:
        """
        Checks if the current face matches the enrolled student.
        Rate-limited — only runs every VERIFY_EVERY_N frames.

        Returns:
          {
            "verified":   bool,
            "confidence": float,   lower = better match
            "status":     "verified" | "mismatch" | "no_face" | "not_enrolled"
          }
        """
        self._frame_count += 1

        if not self._enrolled:
            return {"verified": True, "confidence": 0.0, "status": "not_enrolled"}

        # Rate limiting — return cached result on non-check frames
        if self._frame_count % self.VERIFY_EVERY_N != 0:
            return self._last_result

        faces = self._detect_faces(frame)
        if len(faces) == 0:
            self._last_result = {"verified": False, "confidence": 999.0, "status": "no_face"}
            return self._last_result

        largest = max(faces, key=lambda f: f[2] * f[3])
        face_roi = self._extract_roi(frame, largest)

        try:
            label, confidence = self._recognizer.predict(face_roi)
            verified = confidence < self.MIN_CONFIDENCE
            status   = "verified" if verified else "mismatch"
            self._last_result = {
                "verified":   verified,
                "confidence": round(float(confidence), 1),
                "status":     status,
            }
        except Exception as e:
            print(f"[FaceVerifier] Prediction error: {e}")
            self._last_result = {"verified": True, "confidence": 0.0, "status": "error"}

        return self._last_result

    def draw_status(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draws verification status in the top-right area of the frame."""
        h, w = frame.shape[:2]
        status = result.get("status", "pending")

        if status == "verified":
            color = (0, 180, 0)
            label = "ID: Verified"
        elif status == "mismatch":
            color = (0, 0, 220)
            label = f"ID: MISMATCH ({result['confidence']:.0f})"
        elif status == "no_face":
            color = (0, 140, 255)
            label = "ID: No face"
        elif status == "not_enrolled":
            color = (150, 150, 150)
            label = "ID: Not enrolled"
        else:
            color = (150, 150, 150)
            label = "ID: Checking..."

        cv2.putText(frame, label, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return frame

    def reset_enrollment(self):
        """Deletes stored enrollment so the student can re-enroll."""
        if self._model_path.exists():
            os.remove(self._model_path)
        self._enrolled = False
        if hasattr(self, "_enrollment_faces"):
            self._enrollment_faces = []
        print(f"[FaceVerifier] Enrollment reset for {self.student_id}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_faces(self, frame: np.ndarray):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor  = 1.1,
            minNeighbors = 5,
            minSize      = (60, 60),
        )
        if len(faces) == 0:
            return []
        return list(faces)

    def _extract_roi(self, frame: np.ndarray, face_rect) -> np.ndarray:
        x, y, fw, fh = face_rect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi  = gray[y:y + fh, x:x + fw]
        roi  = cv2.resize(roi, (100, 100))
        return roi