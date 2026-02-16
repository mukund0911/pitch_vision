
"""
Module for extracting player-centered video clips from tracked sequences.
"""
import os
import cv2
import numpy as np
from collections import deque

class ClipExtractor:
    def __init__(self, output_dir: str, buffer_size: int = 150):
        """
        Initialize the clip extractor.

        Args:
            output_dir (str): Directory to save the extracted clips.
            buffer_size (int): Number of frames to keep in history/buffer 
                               (e.g., 150 frames for 5 seconds at 30fps).
        """
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        
        # Deque for frame buffer (circular buffer)
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Dict to store tracking info for frames currently in buffer
        # Structure: {frame_idx: {track_id: bbox}}
        self.track_history = deque(maxlen=buffer_size)
        
        # Frame counter
        self.frame_count = 0
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def update(self, frame, tracks):
        """
        Update the buffer with the latest frame and tracking results.

        Args:
            frame (numpy.ndarray): Current video frame.
            tracks (list): List of Track objects from DeepSORT.
        """
        self.frame_buffer.append(frame.copy())
        
        # Store detections for this frame
        current_frame_detections = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb() # [left, top, right, bottom]
            current_frame_detections[track_id] = ltrb
            
        self.track_history.append(current_frame_detections)
        self.frame_count += 1

    def extract_clip(self, track_id, save_filename=None):
        """
        Extract and save a clip for a specific player ID.
        
        Args:
            track_id (int): The ID of the player to extract.
            save_filename (str, optional): Filename for the output video. 
                                           If None, generates one based on ID.
        
        Returns:
            str: Path to the saved video file, or None if failed.
        """
        # Ensure we have enough frames
        if len(self.frame_buffer) < self.buffer_size:
            print(f"Warning: Buffer not full ({len(self.frame_buffer)}/{self.buffer_size}). extracting partial clip.")
        
        if save_filename is None:
            save_filename = f"video_{track_id}_{self.frame_count}.mp4"
            
        save_path = os.path.join(self.output_dir, save_filename)
        
        # Video writer setup
        height, width, _ = self.frame_buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (224, 224))
        
        frames_written = 0
        
        # Check if track exists in our history
        # We need to iterate through buffer and history synchronously
        for frame, detections in zip(self.frame_buffer, self.track_history):
            if track_id in detections:
                bbox = detections[track_id]
                # Crop and resize
                processed_frame = self._crop_and_resize(frame, bbox)
                out.write(processed_frame)
                frames_written += 1
            else:
                # Handle missing frames (e.g., occlusion)
                # Option 1: Skip
                # Option 2: Write black frame
                # Option 3: Use last known position (if available) - complex
                # For now, we skip, but ideally we want fixed length.
                # Let's write a black frame to maintain temporal consistency? 
                # Or just skip. Let's write a black frame for now.
                black_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                out.write(black_frame)
                
        out.release()
        
        if frames_written == 0:
            os.remove(save_path) # Don't save empty files
            return None
            
        return save_path

    def _crop_and_resize(self, frame, bbox, target_size=(224, 224)):
        """
        Helper method to crop a frame around a bbox and resize it.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        
        # Add padding/context?
        # Let's take the bounding box directly for now, maybe with slight padding
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Crop
        crop = frame[y1:y2, x1:x2]
        
        # Resize
        if crop.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            
        return cv2.resize(crop, target_size)
