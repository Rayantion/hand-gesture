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


# MediaPipe hand landmark connections (21 points per hand)
HAND_LM = {
    'wrist': 0,
    'thumb_cmc': 1, 'thumb_mcp': 2, 'thumb_ip': 3, 'thumb_tip': 4,
    'index_mcp': 5, 'index_pip': 6, 'index_dip': 7, 'index_tip': 8,
    'middle_mcp': 9, 'middle_pip': 10, 'middle_dip': 11, 'middle_tip': 12,
    'ring_mcp': 13, 'ring_pip': 14, 'ring_dip': 15, 'ring_tip': 16,
    'pinky_mcp': 17, 'pinky_pip': 18, 'pinky_dip': 19, 'pinky_tip': 20,
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),                # palm
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
            """Wrap Tasks result in legacy-compatible format."""
            def __init__(self, results):
                # results.hand_landmarks: list of list of NormalizedLandmark
                # Nest as [hand] to match original multi_hand_landmarks format
                self.multi_hand_landmarks = results.hand_landmarks if results.hand_landmarks else []

        return LegacyResults(results)

    def draw_landmarks(self, image, hand_landmarks):
        """
        Draw hand landmarks and connections on image.

        Args:
            image: BGR image to draw on
            hand_landmarks: list of list of NormalizedLandmark (from Tasks API)

        Returns:
            Image with drawn landmarks
        """
        if not hand_landmarks:
            return image

        h, w = image.shape[:2]

        # hand_landmarks is list of hands; each hand is a list of 21 landmarks
        for hand_lms in hand_landmarks:
            # Draw connections
            for (a, b) in HAND_CONNECTIONS:
                pt1 = hand_lms[a]
                pt2 = hand_lms[b]
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw landmark points
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)

        return image

    def close(self):
        """Clean up MediaPipe resources."""
        self.detector.close()
