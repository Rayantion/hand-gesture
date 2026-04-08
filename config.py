# Pinch detection with hysteresis (prevents false positives)
PINCH_ON_THRESHOLD = 0.03      # Distance to START pinch (tighter - less false triggers)
PINCH_OFF_THRESHOLD = 0.06     # Distance to END pinch (looser)
PINCH_DEBOUNCE_FRAMES = 5      # Consecutive frames needed to confirm pinch

# Drag activation
HOLD_DURATION = 0.5            # Seconds to hold pinch before drag activates (longer = fewer false triggers)

# Click timing
DOUBLE_CLICK_INTERVAL = 0.3    # Max seconds between clicks for double-click

# Cursor - direct mapping with sensitivity
CURSOR_SENSITIVITY = 2.5       # Multiplier for cursor movement (higher = more sensitive)

# Camera
CAM_WIDTH = 640
CAM_HEIGHT = 480
WINDOW_NAME = "Hand Gesture Control"

# Debug
DEBUG = False
