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

    def is_ready_to_move(self, landmarks, current_time=None):
        """
        Detect "ready" gesture: index and thumb extended (like a gun/pistol shape).
        This is more natural than full open palm - user can relax other fingers.
        Requires 2 seconds of steady ready gesture before cursor activates.
        Uses HYSTERESIS: once activated, stays active until gesture is clearly broken.
        """
        if not landmarks:
            self._open_palm_start = None
            self._palm_active = False
            return False

        index_tip = landmarks[8]
        index_pip = landmarks[7]
        index_dip = landmarks[6]
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]

        # Index finger: tip should be above DIP joint (more lenient)
        # Use relative Y position - lower value = higher in image = extended
        index_extended = index_tip.y < index_dip.y

        # Thumb: just check it's not fully curled
        thumb_extended = thumb_tip.y < thumb_ip.y * 1.2

        # Check palm facing camera
        palm_facing = self.is_palm_facing(landmarks)

        # Ready gesture detected
        if index_extended and thumb_extended and palm_facing:
            if self._open_palm_start is None:
                self._open_palm_start = current_time or time.time()

            # Activate after 2 seconds
            elapsed = (current_time or time.time()) - self._open_palm_start
            if elapsed > 2.0:
                self._palm_active = True

            return self._palm_active
        elif self._palm_active:
            # HYSTERESIS: Once activated, stay active unless gesture is VERY broken
            # Only deactivate if BOTH index AND thumb are clearly curled
            index_curled = index_tip.y > index_pip.y
            thumb_curled = thumb_tip.y > thumb_ip.y * 1.5

            if index_curled and thumb_curled:
                self._palm_active = False
                self._open_palm_start = None
                return False
            else:
                # Keep active - user is still gesturing
                return True
        else:
            # Not yet activated, reset timer
            self._open_palm_start = None
            return False

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
        Check if palm is facing the camera (not back of hand).
        Uses 2D geometry: when palm faces camera, thumb is on the side.
        This works at any distance - no z-depth dependency.
        """
        if not landmarks:
            return False

        wrist = landmarks[0]
        index_mcp = landmarks[5]  # Base of index finger
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]

        # Vector from wrist to index MCP
        to_index_x = index_mcp.x - wrist.x
        to_index_y = index_mcp.y - wrist.y

        # Vector from wrist to thumb tip
        to_thumb_x = thumb_tip.x - wrist.x
        to_thumb_y = thumb_tip.y - wrist.y

        # Vector from wrist to pinky tip
        to_pinky_x = pinky_tip.x - wrist.x
        to_pinky_y = pinky_tip.y - wrist.y

        # Cross products tell us which side thumb/pinky are on
        cross_thumb = to_index_x * to_thumb_y - to_index_y * to_thumb_x
        cross_pinky = to_index_x * to_pinky_y - to_index_y * to_pinky_x

        # Palm facing: thumb and pinky on OPPOSITE sides of index line
        # (for right hand: thumb right, pinky left; for left hand: reversed)
        return cross_thumb * cross_pinky < 0
