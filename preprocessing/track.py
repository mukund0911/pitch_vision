from deep_sort_realtime.deepsort_tracker import DeepSort

class PlayerTracker:
    def __init__(self, max_age: int = 30, n_init: int = 3):
        """
        Initialize the DeepSORT tracker.
        
        Args:
            max_age (int): Maximum number of missed frames before a track is deleted.
            n_init (int): Number of consecutive detections to confirm a track.
        """
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update_tracks(self, detections, frame):
        """
        Update the tracker with new detections from the current frame.

        Args:
            detections (list): List of detections [x1, y1, x2, y2, confidence, class_id].
            frame (numpy.ndarray): The current video frame.

        Returns:
            list: A list of active Track objects.
        """
        # Convert detections to format expected by deep_sort_realtime: 
        # [[left, top, w, h], confidence, detection_class]
        formatted_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            w = x2 - x1
            h = y2 - y1
            # DeepSort expects (x, y, w, h), conf, class_id
            formatted_detections.append(([x1, y1, w, h], conf, cls_id))
            
        # Update tracker
        tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
        return tracks
