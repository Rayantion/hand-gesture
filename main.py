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
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAM_WIDTH)
        self.cap.set(4, CAM_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera. Please check your webcam.")

        self.hand_tracker = HandTracker()
        self.gesture_recognizer = GestureRecognizer()
        self.cursor = CursorController()

        self.last_pinch_time = 0
        self.pinch_count = 0
        self.is_holding = False
        self.hold_start_time = None
        self.position_buffer = deque(maxlen=5)
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
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hand_tracker.process_frame(rgb_frame)

            action_text = "Waiting..."

            if results.hand_landmarks:
                self.hand_tracker.draw_landmarks(frame, results)

                # List of 21 NormalizedLandmark objects
                landmarks = results.hand_landmarks[0]

                # Get cursor position from index finger tip (landmark 8)
                cursor_pos = (landmarks[8].x, landmarks[8].y)
                self.cursor.move_to(cursor_pos[0], cursor_pos[1])
                action_text = "Moving cursor"

                current_time = time.time()

                is_pinched = self.gesture_recognizer.is_pinched(landmarks)
                is_open_palm = self.gesture_recognizer.is_open_palm(landmarks)
                is_middle_finger = self.gesture_recognizer.is_middle_finger(landmarks)

                if is_middle_finger:
                    action_text = "靠北 😂"
                    self._show_kaobei(frame)

                elif is_pinched:
                    if not self.is_holding:
                        self.is_holding = True
                        self.hold_start_time = current_time
                    else:
                        hold_time = current_time - self.hold_start_time
                        if hold_time > HOLD_DURATION:
                            if not self.cursor.is_dragging:
                                self.cursor.mouse_down()
                                action_text = "DRAGGING"
                            else:
                                action_text = "DRAGGING"

                    time_since_last = current_time - self.last_pinch_time

                    if time_since_last < DOUBLE_CLICK_INTERVAL:
                        self.cursor.double_click()
                        action_text = "DOUBLE CLICK!"
                        self.pinch_count = 0
                    else:
                        if self.pinch_count == 0:
                            self.cursor.click()
                            action_text = "CLICK"

                    self.last_pinch_time = current_time
                    self.pinch_count += 1

                else:
                    if self.is_holding and self.cursor.is_dragging:
                        self.cursor.mouse_up()
                        action_text = "RELEASED"

                    self.is_holding = False
                    self.hold_start_time = None
                    self.pinch_count = 0

                    if is_open_palm:
                        action_text = "Palm open - moving"

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

            self._draw_overlay(frame, action_text)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False

        self.cleanup()

    def _draw_overlay(self, frame, action_text):
        """Draw instructions and status on frame."""
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)

        cv2.putText(
            frame,
            "Pinch=Click | DoublePinch=DoubleClick | PinchHold=Drag | Press 'q' to quit",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

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
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            "靠北",
            (w // 2 - 100, h // 2),
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
