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

# Smoothing window size (1 = no smoothing, instant response)
SMOOTHING_WINDOW = 1

# Camera resolution
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Preview window title
WINDOW_NAME = "Hand Gesture Control"

# Debug mode (show extra info)
DEBUG = False

# Cursor sensitivity (higher = smaller movement needed to reach screen edge)
# 1.0 = normal, 2.0 = hand at mid-frame reaches screen edge, 3.0 = hand at 1/3 reaches edge
CURSOR_SENSITIVITY = 2.5