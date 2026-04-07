"""
Hand Tracker - MediaPipe Hand Detection
Uses MediaPipe Tasks API (hand_landmarker.task model required).
Download: curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
Save as 'hand_landmarker.task' in same folder as main.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class HandTracker:
    """MediaPipe hand detection handler using Tasks API."""

    MODEL_PATH = 'hand_landmarker.task'

    def __init__(self):
        """Initialize MediaPipe HandLandmarker via Tasks API."""
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._timestamp = 0

    def process_frame(self, rgb_frame):
        """
        Process a frame and detect hand landmarks.
        Returns raw HandLandmarkerResult from Tasks API.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect_for_video(mp_image, self._timestamp)
        self._timestamp += 33  # ~30fps → 33ms per frame
        return results

    def draw_landmarks(self, image, results):
        """
        Draw hand landmarks and connections on image.
        Optimized: only draws dots, no lines (faster).
        """
        if results.hand_landmarks is None:
            return image

        h, w = image.shape[:2]

        for hand_lms in results.hand_landmarks:
            for lm in hand_lms:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1)

        return image

    def close(self):
        """Clean up MediaPipe resources."""
        self.detector.close()
