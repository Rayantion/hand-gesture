"""
Cursor Control - Mouse Action Functions
Handles all mouse cursor movements and clicks
"""

import pyautogui
import numpy as np
from collections import deque
from config import SMOOTHING_WINDOW, SCREEN_WIDTH, SCREEN_HEIGHT


class CursorController:
    """Control mouse cursor based on hand position."""
    
    def __init__(self):
        """Initialize cursor controller with screen size."""
        # Get actual screen size
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Smoothing buffer
        self.position_buffer = deque(maxlen=SMOOTHING_WINDOW)
        
        # Click timing
        self.last_click_time = 0
        self.click_count = 0
        
        # Drag state
        self.is_dragging = False
    
    def move_to(self, normalized_x, normalized_y):
        """
        Move cursor to position with smoothing.
        
        Args:
            normalized_x: X position (0-1)
            normalized_y: Y position (0-1)
        """
        # Convert to screen coordinates
        screen_x = int(normalized_x * self.screen_width)
        screen_y = int(normalized_y * self.screen_height)
        
        # Add to smoothing buffer
        self.position_buffer.append((screen_x, screen_y))
        
        # Calculate average position
        avg_x = int(np.mean([p[0] for p in self.position_buffer]))
        avg_y = int(np.mean([p[1] for p in self.position_buffer]))
        
        # Move cursor
        pyautogui.moveTo(avg_x, avg_y, duration=0)
    
    def click(self):
        """Perform single click."""
        pyautogui.click()
    
    def double_click(self):
        """Perform double click."""
        pyautogui.doubleClick()
    
    def right_click(self):
        """Perform right click."""
        pyautogui.rightClick()
    
    def mouse_down(self, button='left'):
        """Hold mouse button down."""
        pyautogui.mouseDown(button=button)
        self.is_dragging = True
    
    def mouse_up(self, button='left'):
        """Release mouse button."""
        pyautogui.mouseUp(button=button)
        self.is_dragging = False
    
    def scroll(self, amount):
        """
        Scroll the page.
        
        Args:
            amount: Positive=up, Negative=down
        """
        pyautogui.scroll(amount)
    
    def is_moving(self, normalized_x, normalized_y):
        """
        Check if cursor is actively moving.
        
        Args:
            normalized_x, normalized_y: Current position
            
        Returns:
            True if cursor is moving significantly
        """
        if len(self.position_buffer) < 2:
            return True
        
        last_pos = self.position_buffer[-1]
        prev_pos = self.position_buffer[-2]
        
        distance = np.sqrt(
            (last_pos[0] - prev_pos[0])**2 +
            (last_pos[1] - prev_pos[1])**2
        )
        
        return distance > 3  # More than 3 pixels moved