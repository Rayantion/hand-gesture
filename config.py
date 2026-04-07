# Pinch detection with hysteresis (prevents false positives)
PINCH_ON_THRESHOLD = 0.04      # Distance to START pinch (tighter)
PINCH_OFF_THRESHOLD = 0.08     # Distance to END pinch (looser)
PINCH_DEBOUNCE_FRAMES = 3      # Consecutive frames needed to confirm pinch
HOLD_DURATION = 0.2            # Seconds to hold before drag activates
DOUBLE_CLICK_INTERVAL = 0.3    # Max seconds between clicks for double-click

# Cursor smoothing (EMA)
SMOOTHING_ALPHA = 0.25         # 0.1=very smooth, 0.5=snappy
CURSOR_SENSITIVITY = 3.0       # Delta multiplier from home position

# Camera
CAM_WIDTH = 640
CAM_HEIGHT = 480
WINDOW_NAME = "Hand Gesture Control"

# Debug
DEBUG = False
