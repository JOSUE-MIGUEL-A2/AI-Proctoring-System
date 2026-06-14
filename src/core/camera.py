# src/core/camera.py

import cv2


class CameraCapture:
    """
    Manages the webcam feed lifecycle.
    Always call release() in a finally block.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check if your webcam is connected and not in use by another app."
            )

    def read_frame(self):
        """Returns (success: bool, frame: np.ndarray)."""
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()