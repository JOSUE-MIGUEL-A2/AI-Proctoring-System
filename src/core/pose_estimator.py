# src/core/pose_estimator.py

import cv2
import numpy as np


class HeadPoseEstimator:
    """
    Computes Yaw, Pitch, and Roll (in degrees) from 2D facial keypoints
    using OpenCV's solvePnP with a generic 3D face model.

    Coordinate convention (right-hand rule):
      Yaw   (+) = turning right,  (-) = turning left
      Pitch (+) = looking up,     (-) = looking down
      Roll  (+) = tilting right,  (-) = tilting left
    """

    # Generic 3D facial landmark positions in centimeters
    # Origin is at the tip of the nose. Axes: X=right, Y=up, Z=forward
    MODEL_3D_POINTS = np.array([
        [0.0,    0.0,    0.0],    # Nose tip
        [-1.0,   1.65,  -0.25],   # Left eye center
        [1.0,    1.65,  -0.25],   # Right eye center
        [-2.4,   1.20,  -1.3],    # Left ear
        [2.4,    1.20,  -1.3],    # Right ear
    ], dtype=np.float64)

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        # Approximate camera intrinsic matrix (works without calibration)
        focal_length = frame_width
        cx = frame_width / 2
        cy = frame_height / 2
        self.camera_matrix = np.array([
            [focal_length, 0,            cx],
            [0,            focal_length, cy],
            [0,            0,            1 ],
        ], dtype=np.float64)

        # Assuming no lens distortion (valid for most webcams at close range)
        self.dist_coeffs = np.zeros((4, 1))

    def estimate(self, landmarks: dict) -> tuple[float, float, float] | None:
        """
        Takes the landmarks dict from PoseDetector.detect() and returns
        (yaw, pitch, roll) in degrees, or None if solvePnP fails.
        """
        # Build 2D image point array in the same order as MODEL_3D_POINTS
        image_points_list = [
            landmarks["nose"],
            landmarks["left_eye"],
            landmarks["right_eye"],
            landmarks.get("left_ear"),
            landmarks.get("right_ear"),
        ]

        # Filter out any None (occluded) keypoints — must keep 3D/2D arrays aligned
        valid_3d, valid_2d = [], []
        for pt3d, pt2d in zip(self.MODEL_3D_POINTS, image_points_list):
            if pt2d is not None:
                valid_3d.append(pt3d)
                valid_2d.append(pt2d)

        if len(valid_2d) < 3:
            # solvePnP needs at least 3 correspondences for reliable results
            return None

        model_points = np.array(valid_3d, dtype=np.float64)
        image_points = np.array(valid_2d, dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(
            model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        if not success:
            return None

        # Convert rotation vector → rotation matrix → Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        yaw, pitch, roll = self._rotation_matrix_to_euler(rotation_mat)

        return yaw, pitch, roll

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
        """Decomposes a 3x3 rotation matrix into Yaw, Pitch, Roll in degrees."""
        # Clamp to avoid numerical issues with arcsin
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
            pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            roll  = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        else:
            yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
            pitch = 0.0
            roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))

        return yaw, pitch, roll