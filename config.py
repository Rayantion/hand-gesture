"""
Configuration settings for Hand Gesture Control
Easy to adjust sensitivity and parameters
"""

# Pinch detection threshold (0-1, smaller = more strict, larger = more sensitive)
# Distance between thumb and index finger tips
PINCH_THRESHOLD = 0.07

# Double click timing (seconds)
# Max time between pinches to register as double click
DOUBLE_CLICK_INTERVAL = 0.4

# Hold duration for drag action (seconds)
# How long to hold pinch before drag activates
HOLD_DURATION = 0.8

# Smoothing window size
# Number of frames to average for smooth cursor movement
SMOOTHING_WINDOW = 5

# Camera resolution
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Preview window title
WINDOW_NAME = "Hand Gesture Control"

# Debug mode (show extra info)
DEBUG = False

# Cursor sensitivity (0.65 = amplified small movements, lower = more sensitive)
CURSOR_SENSITIVITY = 0.65