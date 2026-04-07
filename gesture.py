"""
Gesture Recognition - Detect Hand Gestures
Analyzes hand landmarks to recognize gestures
"""

import numpy as np
from config import PINCH_THRESHOLD, CURSOR_SENSITIVITY


class GestureRecognizer:
    """Recognize hand gestures from landmarks."""

    THUMB_TIP = 4
    INDEX_TIP = 8

    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_KNUCKLES = [5, 9, 13, 17]

    @staticmethod
    def get_distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    @staticmethod
    def is_pinched(landmarks):
        if not landmarks:
            return False
        thumb = landmarks[GestureRecognizer.THUMB_TIP]
        index = landmarks[GestureRecognizer.INDEX_TIP]
        return GestureRecognizer.get_distance(thumb, index) < PINCH_THRESHOLD

    @staticmethod
    def get_cursor_position(landmarks, home_position=None):
        """
        Cursor position relative to home (center of frame).
        home_position: (x, y) of the hand when it first entered frame.
        Sensitivity applied so ~3cm hand movement reaches screen edge.
        Returns (x, y) in 0-1 screen space, or None if no landmarks.
        """
        if not landmarks:
            return None

        thumb = landmarks[GestureRecognizer.THUMB_TIP]
        index = landmarks[GestureRecognizer.INDEX_TIP]

        mx = (thumb.x + index.x) / 2.0
        my = (thumb.y + index.y) / 2.0

        if home_position is None:
            # First frame — set this as home, cursor at center
            return (0.5, 0.5)

        home_x, home_y = home_position

        # Delta from home position
        dx = mx - home_x
        dy = my - home_y

        # Scale delta by sensitivity so small movements = full screen
        # With CURSOR_SENSITIVITY=3.0, ~15cm movement from home reaches edge
        cursor_x = 0.5 + dx * CURSOR_SENSITIVITY
        cursor_y = 0.5 + dy * CURSOR_SENSITIVITY

        cursor_x = max(0.0, min(1.0, cursor_x))
        cursor_y = max(0.0, min(1.0, cursor_y))

        return (cursor_x, cursor_y)

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
