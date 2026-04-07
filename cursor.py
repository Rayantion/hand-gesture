"""
Cursor Control - Mouse Action Functions
Handles all mouse cursor movements and clicks
"""

import pyautogui
pyautogui.FAILSAFE = False


class CursorController:
    """Control mouse cursor based on hand position."""
    
    def __init__(self):
        """Initialize cursor controller with screen size."""
        self.screen_width, self.screen_height = pyautogui.size()
        self.is_dragging = False
    
    def move_to(self, normalized_x, normalized_y):
        """
        Move cursor to position.
        Args:
            normalized_x: X position (0-1)
            normalized_y: Y position (0-1)
        """
        screen_x = int(normalized_x * self.screen_width)
        screen_y = int(normalized_y * self.screen_height)
        pyautogui.moveTo(screen_x, screen_y, duration=0)
    
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
    
