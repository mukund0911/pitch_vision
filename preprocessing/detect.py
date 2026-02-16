import cv2
import numpy as np
from ultralytics import YOLO
import time

class PlayerDetector:
    def __init__(self, model_path: str = "yolo26x.pt"):
        """
        Initialize the YOLOv8 model for player detection.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.5

    def detect(self, frame):
        """
        Detect players in a single video frame.
        Returns:
            list: A list of detections [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        # YOLOv8 results extraction
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            # Class 0 is 'person' in COCO dataset
            if cls_id == 0 and conf > self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, conf, cls_id])
                
        return detections

if __name__ == "__main__":
    # Import the tracker and clip extractor
    try:
        from track import PlayerTracker
        from clip_extractor import ClipExtractor
    except ImportError:
        # If running from pycharm/vscode sometimes path needs adjustment or strict package structure
        import sys
        sys.path.append(".") 
        from track import PlayerTracker
        from clip_extractor import ClipExtractor

    # Test the detector on a video
    video_path = "./clips/argentina_vs_france2022.mp4" # Update with your actual video path!
    
    # Check if video exists, if not use webcam or exit
    import os
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}. Using webcam (0) instead.")
        video_path = 0
        
    cap = cv2.VideoCapture(video_path)
    
    # Initialize Detector AND Tracker AND ClipExtractor
    detector = PlayerDetector(model_path="./models/yolov8n.pt")
    tracker = PlayerTracker(max_age=30, n_init=3)
    
    # Clip Extraction Setup
    CLIP_DURATION = 5 # seconds
    FPS = 30
    BUFFER_SIZE = CLIP_DURATION * FPS
    # Create directory for clips if it doesn't exist
    OUTPUT_CLIPS_DIR = "d:/Personal Projects/pitch_vision/data/clips"
    clip_extractor = ClipExtractor(output_dir=OUTPUT_CLIPS_DIR, buffer_size=BUFFER_SIZE)
    
    print(f"Processing video: {video_path}")
    print(f"Saving clips to: {OUTPUT_CLIPS_DIR}")
    
    frame_count = 0
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
            
        # 1. Detect players
        detections = detector.detect(frame)
        
        # 2. Update Tracker
        tracks = tracker.update_tracks(detections, frame)

        # 3. Update Clip Extractor Buffer
        clip_extractor.update(frame, tracks)
        
        # 4. Auto-save clips every BUFFER_SIZE frames (e.g., every 5 seconds)
        if frame_count % BUFFER_SIZE == 0:
            print(f"Frame {frame_count}: Extracting clips for active players...")
            for track in tracks:
                # Only extract if track is confirmed and has recent update
                if track.is_confirmed() and track.time_since_update <= 1:
                    save_path = clip_extractor.extract_clip(track.track_id)
                    if save_path:
                        print(f"  Saved clip for ID {track.track_id}")
        
        # 5. Visualize Tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            # Get bounding box [left, top, right, bottom]
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for tracked players
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Track ID
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
            
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        cv2.imshow("PitchVision - Tracking & Extraction", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


