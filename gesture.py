"""
Gesture Recognition - Detect Hand Gestures
Analyzes hand landmarks to recognize gestures
"""

import numpy as np
from config import PINCH_THRESHOLD


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
    
    FINGER_TIPS = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    FINGER_KNUCKLES = [5, 9, 13, 17]  # Corresponding MCP joints
    
    @staticmethod
    def get_distance(p1, p2):
        """
        Calculate Euclidean distance between two landmarks.
        
        Args:
            p1, p2: Landmark objects with .x and .y attributes
            
        Returns:
            Distance value
        """
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    @staticmethod
    def is_pinched(landmarks):
        """
        Check if thumb and index finger are pinched together.
        
        Args:
            landmarks: List of 21 hand landmarks
            
        Returns:
            True if pinched, False otherwise
        """
        if not landmarks:
            return False
        
        thumb = landmarks[GestureRecognizer.THUMB_TIP]
        index = landmarks[GestureRecognizer.INDEX_TIP]
        
        distance = GestureRecognizer.get_distance(thumb, index)
        return distance < PINCH_THRESHOLD
    
    @staticmethod
    def is_palm_facing(landmarks):
        """
        Check if palm is facing the camera (not back of hand).

        Uses handedness cross-product: if thumb is on the "other side"
        of the wrist from the fingers, palm is facing camera.
        When back of hand faces camera the fingers and thumb
        are on the same side of the wrist.
        """
        if not landmarks:
            return False

        wrist = landmarks[0]
        index_mcp = landmarks[5]
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]

        # Vector from wrist to index_mcp (pointing toward fingers)
        to_index = (index_mcp.x - wrist.x, index_mcp.y - wrist.y)
        # Vector from wrist to thumb_tip
        to_thumb = (thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
        # Vector from wrist to pinky_tip
        to_pinky = (pinky_tip.x - wrist.x, pinky_tip.y - wrist.y)

        # Cross product tells us which side thumb is on vs fingers
        # cross > 0 means thumb on left side of finger direction (pinky side)
        # cross < 0 means thumb on right side (index side)
        cross = to_index[0] * to_thumb[1] - to_index[1] * to_thumb[0]

        # Cross for pinky as well
        cross_pinky = to_index[0] * to_pinky[1] - to_index[1] * to_pinky[0]

        # Palm-facing: thumb and pinky are on similar side (same sign of cross product)
        # Back-facing: thumb and pinky are on opposite sides
        # Use absolute comparison
        same_sign_thumb_pinky = (cross * cross_pinky) > 0

        # Also check: thumb should be on the "outside" (opposite from pinky)
        # Back of hand: fingers and thumb cluster together on same side
        # Palm: fingers and thumb spread apart

        return same_sign_thumb_pinky

    @staticmethod
    def is_open_palm(landmarks):
        """
        Check if hand is open (all fingers extended).

        Args:
            landmarks: List of 21 hand landmarks

        Returns:
            True if open palm, False otherwise
        """
        if not landmarks:
            return False

        # Check if all fingertips are above their knuckles
        for tip_idx, knuckle_idx in zip(
            GestureRecognizer.FINGER_TIPS,
            GestureRecognizer.FINGER_KNUCKLES
        ):
            tip = landmarks[tip_idx]
            knuckle = landmarks[knuckle_idx]

            # If any fingertip is below its knuckle, hand is not open
            if tip.y > knuckle.y:
                return False

        return True
    
    @staticmethod
    def is_fist(landmarks):
        """
        Check if hand is in a fist position.
        
        Args:
            landmarks: List of 21 hand landmarks
            
        Returns:
            True if fist, False otherwise
        """
        if not landmarks:
            return False
        
        # Fist: fingertips below knuckles
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
        """
        Check if showing middle finger (中指).
        
        Detection: Middle finger extended, other fingers curled
        
        Args:
            landmarks: List of 21 hand landmarks
            
        Returns:
            True if middle finger gesture, False otherwise
        """
        if not landmarks:
            return False
        
        # Landmarks for each finger (tip, pip, mcp)
        # Index: 8, 7, 5
        # Middle: 12, 11, 9
        # Ring: 16, 15, 13
        # Pinky: 20, 19, 17
        
        middle_tip = landmarks[12]
        middle_pip = landmarks[11]
        middle_mcp = landmarks[9]
        
        index_tip = landmarks[8]
        index_pip = landmarks[7]
        
        ring_tip = landmarks[16]
        ring_pip = landmarks[15]
        
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[19]
        
        # Middle finger extended (tip above pip)
        middle_extended = middle_tip.y < middle_pip.y
        
        # Other fingers curled (tips below their pip joints)
        index_curled = index_tip.y > index_pip.y
        ring_curled = ring_tip.y > ring_pip.y
        pinky_curled = pinky_tip.y > pinky_pip.y
        
        return middle_extended and index_curled and ring_curled and pinky_curled
    
    @staticmethod
    def get_cursor_position(landmarks):
        """
        Get cursor position from index finger tip.
        
        Args:
            landmarks: List of 21 hand landmarks
            
        Returns:
            (x, y) normalized coordinates (0-1)
        """
        if not landmarks:
            return None
        
        index_tip = landmarks[GestureRecognizer.INDEX_TIP]
        return (index_tip.x, index_tip.y)