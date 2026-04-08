"""
Gesture Recognition - Simplified
Based on proven working implementations.
Most logic now in main.py for clarity and debugging.
"""

import numpy as np


class GestureRecognizer:
    """Simple gesture recognition - most logic in main.py."""

    THUMB_TIP = 4
    INDEX_TIP = 8

    @staticmethod
    def get_distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    @staticmethod
    def is_index_extended(landmarks):
        """Check if index finger is extended (for cursor control)."""
        if not landmarks:
            return False
        index_tip = landmarks[8]
        index_pip = landmarks[7]
        return index_tip.y < index_pip.y
