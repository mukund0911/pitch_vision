"""
Pixel → field coordinate mapping using homography estimation.

For tactical cam videos (mostly fixed camera), we estimate a perspective
transform using 4+ known field landmarks (corners, penalty spots, center circle).

Two modes:
  1. Manual calibration: user provides pixel↔field point correspondences
  2. From file: load a previously saved homography matrix

Standard pitch: 105m × 68m, origin at center of field.

Usage:
    h = HomographyEstimator()
    h.calibrate_manual(
        pixel_points=[[100, 50], [1800, 50], [100, 900], [1800, 900]],
        field_point_names=["top_left", "top_right", "bottom_left", "bottom_right"],
    )
    field_coords = h.pixel_to_field([[960, 540]])  # -> [x_meters, y_meters]
    h.save("match_homography.npz")
"""

import cv2
import numpy as np
from pathlib import Path


# Standard football pitch dimensions (meters)
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

# Known field landmarks in real-world coordinates (meters)
# Origin at center of field, x along length, y along width
FIELD_POINTS = {
    "top_left":           [-52.5, -34.0],
    "top_right":          [ 52.5, -34.0],
    "bottom_left":        [-52.5,  34.0],
    "bottom_right":       [ 52.5,  34.0],
    "center":             [  0.0,   0.0],
    "left_penalty_spot":  [-40.5,   0.0],
    "right_penalty_spot": [ 40.5,   0.0],
    # Penalty area corners (16.5m from goal line, 20.15m from center)
    "left_pa_top":        [-52.5, -20.15],
    "left_pa_bottom":     [-52.5,  20.15],
    "left_pa_top_edge":   [-36.0, -20.15],
    "left_pa_bottom_edge":[-36.0,  20.15],
    "right_pa_top":       [ 52.5, -20.15],
    "right_pa_bottom":    [ 52.5,  20.15],
    "right_pa_top_edge":  [ 36.0, -20.15],
    "right_pa_bottom_edge":[ 36.0,  20.15],
    # Center circle (approximate points on the circle)
    "center_circle_top":  [  0.0,  -9.15],
    "center_circle_bottom":[ 0.0,   9.15],
}


class HomographyEstimator:
    def __init__(self):
        self.H = None       # 3x3 pixel → field
        self.H_inv = None   # 3x3 field → pixel

    @property
    def is_calibrated(self) -> bool:
        return self.H is not None

    def calibrate_manual(self, pixel_points: list, field_point_names: list):
        """
        Calibrate from user-provided pixel ↔ field correspondences.

        Args:
            pixel_points: [[px1, py1], [px2, py2], ...] — at least 4 points.
            field_point_names: Matching landmark names from FIELD_POINTS dict.

        Raises:
            ValueError: If fewer than 4 points or unknown field point name.
        """
        if len(pixel_points) < 4:
            raise ValueError(f"Need at least 4 points, got {len(pixel_points)}")

        field_coords = []
        for name in field_point_names:
            if name not in FIELD_POINTS:
                raise ValueError(
                    f"Unknown field point '{name}'. "
                    f"Available: {list(FIELD_POINTS.keys())}"
                )
            field_coords.append(FIELD_POINTS[name])

        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(field_coords, dtype=np.float32)

        self.H, _ = cv2.findHomography(src, dst)
        self.H_inv, _ = cv2.findHomography(dst, src)

        print(f"Homography calibrated from {len(pixel_points)} points")

    def calibrate_from_points(self, pixel_points: list, field_points: list):
        """
        Calibrate from raw pixel ↔ field coordinate pairs (no name lookup).

        Args:
            pixel_points: [[px1, py1], ...] — at least 4 points.
            field_points: [[fx1, fy1], ...] — corresponding field coords in meters.
        """
        if len(pixel_points) < 4:
            raise ValueError(f"Need at least 4 points, got {len(pixel_points)}")

        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(field_points, dtype=np.float32)

        self.H, _ = cv2.findHomography(src, dst)
        self.H_inv, _ = cv2.findHomography(dst, src)

    def pixel_to_field(self, pixel_coords: list) -> np.ndarray:
        """
        Transform pixel coordinates to field coordinates (meters).

        Args:
            pixel_coords: List of [px, py] or (N, 2) array.

        Returns:
            (N, 2) array of [x_meters, y_meters] in field frame.
        """
        if not self.is_calibrated:
            raise RuntimeError("Homography not calibrated. Call calibrate_manual() first.")

        pts = np.array(pixel_coords, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed.reshape(-1, 2)

    def field_to_pixel(self, field_coords: list) -> np.ndarray:
        """
        Inverse transform: field coordinates (meters) → pixel coordinates.
        Useful for overlaying model predictions back onto video.

        Args:
            field_coords: List of [x_m, y_m] or (N, 2) array.

        Returns:
            (N, 2) array of [px, py].
        """
        if self.H_inv is None:
            raise RuntimeError("Homography not calibrated.")

        pts = np.array(field_coords, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H_inv)
        return transformed.reshape(-1, 2)

    def is_on_field(self, field_coords: np.ndarray, margin: float = 5.0) -> np.ndarray:
        """
        Check if field coordinates are within pitch bounds (with margin).

        Args:
            field_coords: (N, 2) array of field positions.
            margin: Extra meters beyond pitch boundary to allow.

        Returns:
            (N,) boolean array.
        """
        x_ok = np.abs(field_coords[:, 0]) <= (FIELD_LENGTH / 2 + margin)
        y_ok = np.abs(field_coords[:, 1]) <= (FIELD_WIDTH / 2 + margin)
        return x_ok & y_ok

    def save(self, path: str):
        """Save homography matrices to .npz file."""
        np.savez(path, H=self.H, H_inv=self.H_inv)
        print(f"Homography saved: {path}")

    def load(self, path: str):
        """Load homography matrices from .npz file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Homography file not found: {path}")
        data = np.load(path)
        self.H = data["H"]
        self.H_inv = data["H_inv"]
        print(f"Homography loaded: {path}")
