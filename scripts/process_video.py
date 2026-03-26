"""
Process a YouTube tactical cam video → tracking JSON.

Usage:
    # First time: calibrate homography
    python scripts/process_video.py --video data/videos/match.mp4 \
                                     --output data/processed/match_tracking.json \
                                     --calibrate

    # With saved homography:
    python scripts/process_video.py --video data/videos/match.mp4 \
                                     --output data/processed/match_tracking.json \
                                     --homography data/videos/match_homography.npz

    # Quick test (first 100 frames):
    python scripts/process_video.py --video data/videos/match.mp4 \
                                     --output test_tracking.json \
                                     --homography data/videos/match_homography.npz \
                                     --max-frames 100
"""

import argparse
import cv2
import json
import sys
from pathlib import Path

from pitchvision.video.pipeline import VideoProcessor
from pitchvision.video.homography import FIELD_POINTS


def interactive_calibrate(video_path: str, save_path: str):
    """
    Open the first frame of a video and let the user click field landmarks.
    Saves the resulting homography to save_path.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: cannot read video {video_path}")
        sys.exit(1)

    print("\n=== Homography Calibration ===")
    print("Available field landmarks:")
    for i, name in enumerate(FIELD_POINTS.keys()):
        coords = FIELD_POINTS[name]
        print(f"  {i:2d}. {name:30s} ({coords[0]:+6.1f}, {coords[1]:+6.1f}) meters")

    print("\nInstructions:")
    print("  1. A window will open showing the first frame of the video")
    print("  2. Click on 4+ field landmarks you can identify")
    print("  3. After each click, enter the landmark name")
    print("  4. Type 'done' when finished (minimum 4 points)")

    clicked_pixels = []
    clicked_names = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pixels.append([x, y])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(len(clicked_pixels)), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration", frame)
            print(f"\n  Point {len(clicked_pixels)} clicked at pixel ({x}, {y})")

    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", on_mouse)

    available = list(FIELD_POINTS.keys())

    while True:
        cv2.waitKey(100)
        if len(clicked_pixels) > len(clicked_names):
            name = input(f"  Enter landmark name for point {len(clicked_names)+1}: ").strip()
            if name == "done":
                break
            if name not in FIELD_POINTS:
                print(f"  Unknown landmark '{name}'. Skipping this point.")
                clicked_pixels.pop()
                continue
            clicked_names.append(name)
            print(f"  Mapped to {name} ({FIELD_POINTS[name]})")

        if len(clicked_names) >= 4:
            key = cv2.waitKey(1)
            cont = input("\n  Add more points? (y/n): ").strip().lower()
            if cont != "y":
                break

    cv2.destroyAllWindows()

    if len(clicked_names) < 4:
        print(f"Error: need at least 4 points, got {len(clicked_names)}")
        sys.exit(1)

    # Calibrate and save
    from pitchvision.video.homography import HomographyEstimator
    h = HomographyEstimator()
    h.calibrate_manual(clicked_pixels[:len(clicked_names)], clicked_names)
    h.save(save_path)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Process tactical cam video → tracking JSON")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path for output tracking JSON")
    parser.add_argument("--homography", help="Path to saved .npz homography file")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run interactive calibration on first frame")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames to process")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Process every Nth frame (default: 1)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for YOLO detection")
    parser.add_argument("--model", default="yolov8m.pt",
                        help="YOLOv8 model variant (default: yolov8m.pt)")
    args = parser.parse_args()

    homography_path = args.homography

    # Interactive calibration if requested
    if args.calibrate:
        save_dir = Path(args.video).parent
        save_path = str(save_dir / (Path(args.video).stem + "_homography.npz"))
        homography_path = interactive_calibrate(args.video, save_path)

    if homography_path is None:
        print("Error: provide --homography or use --calibrate")
        sys.exit(1)

    # Process video
    processor = VideoProcessor(
        homography_path=homography_path,
        model_size=args.model,
        device=args.device,
    )
    processor.process_video(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
        subsample=args.subsample,
    )


if __name__ == "__main__":
    main()
