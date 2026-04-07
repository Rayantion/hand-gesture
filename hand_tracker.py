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


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (5, 9), (9, 13), (13, 17),
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

        Args:
            rgb_frame: RGB image frame

        Returns:
            object with multi_hand_landmarks attribute (or None)
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect_for_video(mp_image, self._timestamp)
        self._timestamp += 33  # ~30fps → 33ms per frame

        class LegacyResults:
            def __init__(self, hand_landmarks):
                self.multi_hand_landmarks = hand_landmarks

        return LegacyResults(results.hand_landmarks if results.hand_landmarks else None)

    def draw_landmarks(self, image, hand_landmarks):
        """
        Draw hand landmarks and connections on image.

        Args:
            image: BGR image to draw on
            hand_landmarks: Tasks API landmark list from results.hand_landmarks

        Returns:
            Image with drawn landmarks
        """
        if not hand_landmarks:
            return image

        for landmarks in hand_landmarks:
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                pt1 = landmarks[start_idx]
                pt2 = landmarks[end_idx]
                pt1 = (int(pt1.x * image.shape[1]), int(pt1.y * image.shape[0]))
                pt2 = (int(pt2.x * image.shape[1]), int(pt2.y * image.shape[0]))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

            for lm in landmarks:
                cx = int(lm.x * image.shape[1])
                cy = int(lm.y * image.shape[0])
                cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)

        return image

    def close(self):
        """Clean up MediaPipe resources."""
        self.detector.close()
