"""
Hand Gesture Cursor Control - Main Application
=================================================
Controls mouse cursor using hand gestures from webcam.

Controls:
- Open palm: Move cursor
- Pinch (thumb + index): Click
- Double pinch: Double click
- Pinch + hold: Drag/hold

Requirements:
    pip install -r requirements.txt

Usage:
    python main.py
"""

import cv2
import time
from collections import deque

from config import (
    CAM_WIDTH, CAM_HEIGHT, WINDOW_NAME,
    DOUBLE_CLICK_INTERVAL, HOLD_DURATION, DEBUG
)
from hand_tracker import HandTracker
from gesture import GestureRecognizer
from cursor import CursorController


class HandGestureApp:
    """Main application for hand gesture cursor control."""
    
    def __init__(self):
        """Initialize all components."""
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAM_WIDTH)
        self.cap.set(4, CAM_HEIGHT)
        
        # Check camera opened
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera. Please check your webcam.")
        
        # Initialize components
        self.hand_tracker = HandTracker()
        self.gesture_recognizer = GestureRecognizer()
        self.cursor = CursorController()
        
        # State tracking
        self.last_pinch_time = 0
        self.pinch_count = 0
        self.is_holding = False
        self.hold_start_time = None
        self.position_buffer = deque(maxlen=5)
        
        # Running flag
        self.running = True
        
        print(f"🖐️ {WINDOW_NAME} Started")
        print("   Controls:")
        print("   - Open palm: Move cursor")
        print("   - Pinch: Click")
        print("   - Double pinch: Double click")
        print("   - Pinch + hold: Drag")
        print("   - Press 'q' to quit")
    
    def run(self):
        """Main application loop."""
        while self.running:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hand_tracker.process_frame(rgb_frame)
            
            # Get gesture and act
            action_text = "Waiting..."
            
            if results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.hand_tracker.draw_landmarks(
                    frame, 
                    results.multi_hand_landmarks[0]
                )
                
                # Get landmarks (list of 21 NormalizedLandmark objects)
                landmarks = results.multi_hand_landmarks[0]
                
                # Get cursor position from index finger
                cursor_pos = self.gesture_recognizer.get_cursor_position(landmarks)
                
                if cursor_pos:
                    # Move cursor
                    self.cursor.move_to(cursor_pos[0], cursor_pos[1])
                    action_text = "Moving cursor"
                
                # Gesture detection
                current_time = time.time()
                
                # Check pinch
                is_pinched = self.gesture_recognizer.is_pinched(landmarks)
                is_open_palm = self.gesture_recognizer.is_open_palm(landmarks)
                is_middle_finger = self.gesture_recognizer.is_middle_finger(landmarks)
                
                # � middle finger = 靠北
                if is_middle_finger:
                    action_text = "靠北 😂"
                    self._show_kaobei(frame)
                
                elif is_pinched:
                    # Handle hold/drag
                    if not self.is_holding:
                        self.is_holding = True
                        self.hold_start_time = current_time
                    else:
                        # Check hold duration
                        hold_time = current_time - self.hold_start_time
                        if hold_time > HOLD_DURATION:
                            if not self.cursor.is_dragging:
                                self.cursor.mouse_down()
                                action_text = "DRAGGING"
                            else:
                                action_text = "DRAGGING"
                    
                    # Check for click timing
                    time_since_last = current_time - self.last_pinch_time
                    
                    if time_since_last < DOUBLE_CLICK_INTERVAL:
                        # Double click
                        self.cursor.double_click()
                        action_text = "DOUBLE CLICK!"
                        self.pinch_count = 0
                    else:
                        # Single click (on pinch release)
                        if self.pinch_count == 0:
                            self.cursor.click()
                            action_text = "CLICK"
                    
                    self.last_pinch_time = current_time
                    self.pinch_count += 1
                
                else:
                    # Not pinched - release drag if holding
                    if self.is_holding and self.cursor.is_dragging:
                        self.cursor.mouse_up()
                        action_text = "RELEASED"
                    
                    self.is_holding = False
                    self.hold_start_time = None
                    self.pinch_count = 0
                    
                    if is_open_palm:
                        action_text = "Palm open - moving"
                
                # Debug info
                if DEBUG:
                    cv2.putText(
                        frame,
                        f"Pinched: {is_pinched} | Open: {is_open_palm}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1
                    )
            
            # Draw UI overlay
            self._draw_overlay(frame, action_text)
            
            # Show preview window
            cv2.imshow(WINDOW_NAME, frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
        
        self.cleanup()
    
    def _draw_overlay(self, frame, action_text):
        """Draw instructions and status on frame."""
        # Status bar background
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        
        # Instructions
        cv2.putText(
            frame,
            "Pinch=Click | DoublePinch=DoubleClick | PinchHold=Drag | Press 'q' to quit",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Action text
        cv2.putText(
            frame,
            action_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    
    def _show_kaobei(self, frame):
        """Show 靠北 on screen when middle finger detected."""
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Draw large 靠北 text in center
        cv2.putText(
            frame,
            "靠北",
            (w//2 - 100, h//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 255),
            10
        )
    
    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_tracker.close()
        print(f"👋 {WINDOW_NAME} Stopped")


def main():
    """Entry point."""
    try:
        app = HandGestureApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()