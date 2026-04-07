"""
Gesture Recognition - Detect Hand Gestures
With hysteresis, debouncing, and EMA smoothing to prevent false positives and jitter.
"""

import numpy as np
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
        # Hysteresis state
        self._is_pinched = False
        self._pinch_debounce_count = 0

        # EMA smoothing state
        self._smooth_x = None
        self._smooth_y = None

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
        Cursor position with EMA smoothing.
        Returns (x, y) in 0-1 screen space.
        """
        if not landmarks:
            return None

        thumb = landmarks[self.THUMB_TIP]
        index = landmarks[self.INDEX_TIP]

        mx = (thumb.x + index.x) / 2.0
        my = (thumb.y + index.y) / 2.0

        if home_position is None:
            self._smooth_x = mx
            self._smooth_y = my
            return (0.5, 0.5)

        home_x, home_y = home_position
        dx = mx - home_x
        dy = my - home_y

        # Apply sensitivity
        target_x = 0.5 + dx * CURSOR_SENSITIVITY
        target_y = 0.5 + dy * CURSOR_SENSITIVITY

        # Clamp
        target_x = max(0.0, min(1.0, target_x))
        target_y = max(0.0, min(1.0, target_y))

        # EMA smoothing
        if self._smooth_x is None:
            self._smooth_x = target_x
            self._smooth_y = target_y
        else:
            self._smooth_x = SMOOTHING_ALPHA * target_x + (1 - SMOOTHING_ALPHA) * self._smooth_x
            self._smooth_y = SMOOTHING_ALPHA * target_y + (1 - SMOOTHING_ALPHA) * self._smooth_y

        return (self._smooth_x, self._smooth_y)

    def reset(self):
        """Reset all state (call when hand leaves frame)."""
        self._is_pinched = False
        self._pinch_debounce_count = 0
        self._smooth_x = None
        self._smooth_y = None

    @staticmethod
    def is_open_palm(landmarks):
        if not landmarks:
            return False
        for tip_idx, knuckle_idx in zip(
            GestureRecognizer.FINGER_TIPS,
            GestureRecognizer.FINGER_KNUCKLES
        ):
            tip = landmarks[tip_idx]
            knuckle = landmarks[knuckle_idx]
            if tip.y > knuckle.y:
                return False
        return True

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
        if not landmarks:
            return False
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]

        to_index = (index_mcp.x - wrist.x, index_mcp.y - wrist.y)
        to_thumb = (thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
        to_pinky = (pinky_tip.x - wrist.x, pinky_tip.y - wrist.y)

        cross = to_index[0] * to_thumb[1] - to_index[1] * to_thumb[0]
        cross_pinky = to_index[0] * to_pinky[1] - to_index[1] * to_pinky[0]

        return (cross * cross_pinky) > 0
