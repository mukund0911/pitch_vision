"""
End-to-end video processing pipeline.

Video file → per-frame tracking data (field coordinates) → JSON.

Output schema per frame:
{
    "frame_idx": int,
    "timestamp_s": float,
    "players": [
        {"track_id": int, "pos": [x, y], "bbox_px": [x1,y1,x2,y2]},
        ...
    ],
    "ball": {"pos": [x, y]} or null
}

Usage:
    processor = VideoProcessor(homography_path="match_h.npz")
    processor.process_video("match.mp4", "output/tracking.json")
"""

import sys
import cv2
import json
import time
import numpy as np
import torch
from pathlib import Path

from pitchvision.detection.detector import Detector
from pitchvision.tracking.tracker import DeepSORTTracker
from .homography import HomographyEstimator


class VideoProcessor:
    def __init__(self, homography_path: str = None, model_size: str = "yolov8m.pt",
                 device: str = "cuda"):
        """
        Args:
            homography_path: Path to saved .npz homography file.
                             If None, must call calibrate() before processing.
            model_size: YOLOv8 model variant.
            device: 'cuda' or 'cpu' for detection.
        """
        self.device = device
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device == "cuda" else None
        print(f"Device: {device}" + (f" ({gpu_name})" if gpu_name else ""))

        print("Loading detector...", end=" ", flush=True)
        self.detector = Detector(img_size=416, conf_threshold=0.3)
        print("done")

        print("Loading tracker (ReID)...", end=" ", flush=True)
        self.tracker = DeepSORTTracker()
        print("done")

        self.homography = HomographyEstimator()
        if homography_path is not None:
            self.homography.load(homography_path)
            print(f"Homography loaded from {homography_path}")

    def calibrate(self, pixel_points: list, field_point_names: list):
        """Calibrate homography with manual point correspondences."""
        self.homography.calibrate_manual(pixel_points, field_point_names)

    def process_video(self, video_path: str, output_path: str,
                      max_frames: int = None, subsample: int = 1,
                      progress_every: int = 100) -> str:
        """
        Process a full video and save tracking data as JSON.

        Args:
            video_path: Path to input video file.
            output_path: Path for output JSON file.
            max_frames: Optional frame limit (for testing).
            subsample: Process every Nth frame (e.g., 3 = every 3rd frame).
                       Useful for reducing computation on high-fps videos.
            progress_every: Print progress every N frames.

        Returns:
            Path to saved JSON file.
        """
        if not self.homography.is_calibrated:
            raise RuntimeError(
                "Homography not calibrated. Call calibrate() or provide "
                "homography_path in constructor."
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_frames = min(total_frames, max_frames) if max_frames else total_frames
        print(f"\nVideo: {video_path}")
        print(f"  Resolution: {width}x{height} @ {fps:.1f} fps")
        print(f"  Total frames: {total_frames}"
              + (f" (processing first {target_frames})" if max_frames else ""))
        print(f"  Subsample: every {subsample} frame(s) → ~{fps/subsample:.0f} Hz output")
        print(f"  Processing...", flush=True)

        frames_data = []
        frame_idx = 0
        t_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            # Subsample: skip frames we don't process
            if frame_idx % subsample != 0:
                frame_idx += 1
                continue

            frame_data = self._process_frame(frame, frame_idx, fps)
            frames_data.append(frame_data)

            if progress_every and frame_idx % progress_every == 0 and frame_idx > 0:
                elapsed = time.time() - t_start
                fps_actual = frame_idx / elapsed
                remaining = target_frames - frame_idx
                eta = remaining / fps_actual if fps_actual > 0 else 0
                pct = frame_idx / target_frames * 100
                bar = int(pct // 5)
                sys.stdout.write(
                    f"\r  [{'\u2588' * bar}{'.' * (20 - bar)}] "
                    f"{pct:5.1f}%  frame {frame_idx}/{target_frames}  "
                    f"{fps_actual:.1f} fps  ETA {eta:.0f}s   "
                )
                sys.stdout.flush()

            frame_idx += 1

        cap.release()
        print()  # newline after progress bar
        elapsed = time.time() - t_start

        # Save output
        output = {
            "video_path": str(video_path),
            "fps": fps,
            "resolution": [width, height],
            "total_frames_processed": len(frames_data),
            "subsample": subsample,
            "effective_hz": fps / subsample,
            "processing_time_s": round(elapsed, 2),
            "frames": frames_data,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f)

        print(f"\nDone: {len(frames_data)} frames in {elapsed:.1f}s "
              f"({len(frames_data)/elapsed:.1f} fps)")
        print(f"Output: {output_path}")
        return output_path

    def _process_frame(self, frame: np.ndarray, frame_idx: int,
                       fps: float) -> dict:
        """Process a single frame through detect → track → homography."""
        # Detect players and ball
        detections = self.detector.detect(frame)

        # Track players (assigns persistent IDs)
        tracks = self.tracker.update(detections["players"], frame)

        # Convert player positions to field coordinates
        players = []
        for track in tracks:
            bbox = track["bbox"]
            # Use bottom-center of bbox as approximate foot position
            foot_px = np.array([[(bbox[0] + bbox[2]) / 2, bbox[3]]])
            foot_field = self.homography.pixel_to_field(foot_px)[0]

            # Filter out detections that map outside the pitch
            if self.homography.is_on_field(foot_field.reshape(1, 2))[0]:
                players.append({
                    "track_id": track["track_id"],
                    "pos": [round(float(foot_field[0]), 2),
                            round(float(foot_field[1]), 2)],
                    "bbox_px": [round(c, 1) for c in bbox],
                })

        # Convert ball position to field coordinates
        ball = None
        if detections["ball"] is not None:
            bx1, by1, bx2, by2, _ = detections["ball"]
            ball_center_px = np.array([[(bx1 + bx2) / 2, (by1 + by2) / 2]])
            ball_field = self.homography.pixel_to_field(ball_center_px)[0]

            if self.homography.is_on_field(ball_field.reshape(1, 2), margin=10.0)[0]:
                ball = {
                    "pos": [round(float(ball_field[0]), 2),
                            round(float(ball_field[1]), 2)],
                }

        return {
            "frame_idx": frame_idx,
            "timestamp_s": round(frame_idx / fps, 3),
            "players": players,
            "ball": ball,
        }
