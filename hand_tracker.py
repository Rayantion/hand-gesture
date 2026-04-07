"""
Hand Tracker - MediaPipe Hand Detection
Handles camera input and hand landmark detection
"""

import cv2
import mediapipe as mp


class HandTracker:
    """MediaPipe hand detection handler."""
    
    def __init__(self):
        """Initialize MediaPipe Hands."""
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def process_frame(self, rgb_frame):
        """
        Process a frame and detect hand landmarks.
        
        Args:
            rgb_frame: RGB image frame
            
        Returns:
            MediaPipe hand landmarks or None
        """
        return self.hands.process(rgb_frame)
    
    def draw_landmarks(self, image, hand_landmarks):
        """
        Draw hand landmarks and connections on image.
        
        Args:
            image: BGR image to draw on
            hand_landmarks: Detected hand landmarks
            
        Returns:
            Image with drawn landmarks
        """
        self.mp_draw.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(
                color=(0, 255, 0),
                thickness=2,
                circle_radius=2
            ),
            self.mp_draw.DrawingSpec(
                color=(0, 255, 0),
                thickness=1,
                circle_radius=1
            )
        )
        return image
    
    def close(self):
        """Clean up MediaPipe resources."""
        self.hands.close()