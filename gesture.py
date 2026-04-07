"""
Gesture Recognition - Detect Hand Gestures
Analyzes hand landmarks to recognize gestures
"""

import numpy as np
from config import PINCH_THRESHOLD, CURSOR_SENSITIVITY


class GestureRecognizer:
    """Recognize hand gestures from landmarks."""

    # Landmark indices
    THUMB_TIP = 4
    INDEX_TIP = 8
    THUMB_MCP = 2
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17

    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_KNUCKLES = [5, 9, 13, 17]
    FINGER_PIPS = [6, 10, 14, 18]  # middle joints

    @staticmethod
    def get_distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    @staticmethod
    def is_pinched(landmarks):
        """
        Check if thumb and index finger are pinched together.
        Only checks the two relevant fingers — no extra finger checks.
        """
        if not landmarks:
            return False

        thumb = landmarks[GestureRecognizer.THUMB_TIP]
        index = landmarks[GestureRecognizer.INDEX_TIP]
        distance = GestureRecognizer.get_distance(thumb, index)

        return distance < PINCH_THRESHOLD

    @staticmethod
    def is_pinched_strict(landmarks):
        """Strict pinch — just distance check, for drag release."""
        if not landmarks:
            return False
        thumb = landmarks[GestureRecognizer.THUMB_TIP]
        index = landmarks[GestureRecognizer.INDEX_TIP]
        return GestureRecognizer.get_distance(thumb, index) < PINCH_THRESHOLD

    @staticmethod
    def get_cursor_position(landmarks):
        """
        Get cursor position from midpoint of thumb and index tips.
        Applies sensitivity so small hand movements = full screen travel.
        """
        if not landmarks:
            return None

        thumb = landmarks[GestureRecognizer.THUMB_TIP]
        index = landmarks[GestureRecognizer.INDEX_TIP]

        mx = (thumb.x + index.x) / 2.0
        my = (thumb.y + index.y) / 2.0

        # Amplify: x^0.4 means hand at ~40% of frame reaches 75% of screen
        # Adjust power: lower = more sensitive, higher = more restrained
        power = 1.0 / CURSOR_SENSITIVITY  # CURSOR_SENSITIVITY is > 1
        mx = mx ** power
        my = my ** power

        return (mx, my)

    @staticmethod
    def is_open_palm(landmarks):
        """Check if hand is open (all fingers extended)."""
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
        """Check if hand is in a fist position."""
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
        """Check if showing middle finger (中指)."""
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
        """Check if palm is facing the camera."""
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
