"""
Gesture Recognition - Detect Hand Gestures
With hysteresis, debouncing, and EMA smoothing to prevent false positives and jitter.
"""

import numpy as np
import time
from config import (
    PINCH_ON_THRESHOLD, PINCH_OFF_THRESHOLD, PINCH_DEBOUNCE_FRAMES,
    SMOOTHING_ALPHA, CURSOR_SENSITIVITY
)


class GestureRecognizer:
    """Recognize hand gestures with anti-false-positive measures."""

    THUMB_TIP = 4
    INDEX_TIP = 8

    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_KNUCKLES = [5, 9, 13, 17]

    def __init__(self):
        # Hysteresis state for pinch
        self._is_pinched = False
        self._pinch_debounce_count = 0

        # Open palm timer (2 second delay before activation)
        self._open_palm_start = None
        self._palm_active = False

    @staticmethod
    def get_distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def is_pinched(self, landmarks):
        """
        Pinch detection with HYSTERESIS and DEBOUNCING.
        - Tighter threshold to START pinch (prevents false triggers)
        - Looser threshold to END pinch (prevents rapid flickering)
        - Requires N consecutive frames to confirm state change
        """
        if not landmarks:
            self._pinch_debounce_count = 0
            return False

        thumb = landmarks[self.THUMB_TIP]
        index = landmarks[self.INDEX_TIP]
        distance = self.get_distance(thumb, index)

        if self._is_pinched:
            # Currently pinched — use LOOSER threshold to stay pinched
            if distance > PINCH_OFF_THRESHOLD:
                self._pinch_debounce_count -= 1
                if self._pinch_debounce_count <= 0:
                    self._is_pinched = False
                    self._pinch_debounce_count = 0
            else:
                self._pinch_debounce_count = PINCH_DEBOUNCE_FRAMES
            return True
        else:
            # Not pinched — use TIGHTER threshold to start pinch
            if distance < PINCH_ON_THRESHOLD:
                self._pinch_debounce_count += 1
                if self._pinch_debounce_count >= PINCH_DEBOUNCE_FRAMES:
                    self._is_pinched = True
                return self._pinch_debounce_count >= PINCH_DEBOUNCE_FRAMES
            else:
                self._pinch_debounce_count = max(0, self._pinch_debounce_count - 1)
                return False

    def get_cursor_position(self, landmarks, home_position=None):
        """
        Cursor position based on WRIST (landmark 0).
        Returns (x, y) in 0-1 screen space.
        No smoothing - direct mapping for zero lag.
        """
        if not landmarks:
            return None

        # Use wrist as cursor position (landmark 0)
        wrist = landmarks[0]
        wx, wy = wrist.x, wrist.y

        if home_position is None:
            self._smooth_x = wx
            self._smooth_y = wy
            return (0.5, 0.5)

        home_x, home_y = home_position
        dx = wx - home_x
        dy = wy - home_y

        # Apply sensitivity
        target_x = 0.5 + dx * CURSOR_SENSITIVITY
        target_y = 0.5 + dy * CURSOR_SENSITIVITY

        # Clamp to 0-1
        target_x = max(0.0, min(1.0, target_x))
        target_y = max(0.0, min(1.0, target_y))

        return (target_x, target_y)

    def reset(self):
        """Reset all state (call when hand leaves frame)."""
        self._is_pinched = False
        self._pinch_debounce_count = 0
        self._smooth_x = None
        self._smooth_y = None
        self._open_palm_start = None
        self._palm_active = False

    def is_open_palm(self, landmarks, current_time=None):
        """
        Detect open palm with 2-second activation delay.
        Palm must be open AND facing camera for 2 seconds before cursor activates.
        """
        if not landmarks:
            self._open_palm_start = None
            self._palm_active = False
            return False

        # Check if all fingers are extended (open palm)
        for tip_idx, knuckle_idx in zip(
            GestureRecognizer.FINGER_TIPS,
            GestureRecognizer.FINGER_KNUCKLES
        ):
            tip = landmarks[tip_idx]
            knuckle = landmarks[knuckle_idx]
            if tip.y > knuckle.y:  # Finger is curled
                self._open_palm_start = None
                self._palm_active = False
                return False

        # Check if palm is facing camera (thumb and pinky on opposite sides of index)
        if not self.is_palm_facing(landmarks):
            self._open_palm_start = None
            self._palm_active = False
            return False

        # Palm is open and facing - start/continue timer
        if self._open_palm_start is None:
            self._open_palm_start = current_time or time.time()

        # Activate after 2 seconds
        elapsed = (current_time or time.time()) - self._open_palm_start
        if elapsed > 2.0:
            self._palm_active = True

        return self._palm_active

    @staticmethod
    def is_fist(landmarks):
        if not landmarks:
            return False
        for tip_idx, knuckle_idx in zip(
            GestureRecognizer.FINGER_TIPS,
            GestureRecognizer.FINGER_KNUCKLES
        ):
            tip = landmarks[tip_idx]
            knuckle = landmarks[knuckle_idx]
            if tip.y < knuckle.y:
                return False
        return True

    @staticmethod
    def is_middle_finger(landmarks):
        if not landmarks:
            return False
        middle_tip = landmarks[12]
        middle_pip = landmarks[11]
        index_tip = landmarks[8]
        index_pip = landmarks[7]
        ring_tip = landmarks[16]
        ring_pip = landmarks[15]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[19]

        middle_extended = middle_tip.y < middle_pip.y
        index_curled = index_tip.y > index_pip.y
        ring_curled = ring_tip.y > ring_pip.y
        pinky_curled = pinky_tip.y > pinky_pip.y

        return middle_extended and index_curled and ring_curled and pinky_curled

    @staticmethod
    def is_palm_facing(landmarks):
        """
        Check if palm is facing the camera (not the back of hand).
        Uses z-depth: fingertips should be closer to camera than wrist.
        """
        if not landmarks:
            return False

        wrist = landmarks[0]
        # Check if fingertips are in front of wrist (lower z = closer to camera)
        finger_z_sum = 0
        for tip_idx in [4, 8, 12, 16, 20]:  # thumb, index, middle, ring, pinky tips
            finger_z_sum += landmarks[tip_idx].z

        avg_finger_z = finger_z_sum / 5.0
        # Palm facing camera: fingers have lower z (closer) than wrist
        return avg_finger_z < wrist.z
