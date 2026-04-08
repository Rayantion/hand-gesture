# Pinch detection with hysteresis (prevents false positives)
PINCH_ON_THRESHOLD = 0.05      # Distance to START pinch (tighter)
PINCH_OFF_THRESHOLD = 0.08     # Distance to END pinch (looser)
PINCH_DEBOUNCE_FRAMES = 3      # Consecutive frames needed to confirm pinch

# Drag activation
HOLD_DURATION = 0.3            # Seconds to hold pinch before drag activates

# Click timing
DOUBLE_CLICK_INTERVAL = 0.3    # Max seconds between clicks for double-click

# Cursor smoothing (0.1=very smooth, 0.9=snappy)
SMOOTHING_FACTOR = 0.75
CURSOR_SENSITIVITY = 1.0  # Kept for compatibility, not used in new implementation

# Camera
CAM_WIDTH = 640
CAM_HEIGHT = 480
WINDOW_NAME = "Hand Gesture Control"

# Debug
DEBUG = False
