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