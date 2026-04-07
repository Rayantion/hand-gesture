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

        self.was_pinched = False   # was pinched last frame?
        self.pinch_released = False  # just released pinch this frame?
        self.last_pinch_release = 0  # timestamp of last release
        self.is_holding = False
        self.hold_start_time = None
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

                # Move cursor
                cursor_pos = (landmarks[8].x, landmarks[8].y)
                self.cursor.move_to(cursor_pos[0], cursor_pos[1])
                action_text = "Moving cursor"

                current_time = time.time()
                is_pinched = self.gesture_recognizer.is_pinched(landmarks)
                is_open_palm = self.gesture_recognizer.is_open_palm(landmarks)
                is_middle_finger = self.gesture_recognizer.is_middle_finger(landmarks)
                is_palm_facing = self.gesture_recognizer.is_palm_facing(landmarks)

                # Reject if back of hand is showing
                if not is_palm_facing:
                    action_text = "Show palm"
                    if self.is_holding and self.cursor.is_dragging:
                        self.cursor.mouse_up()
                        self.is_holding = False
                        self.hold_start_time = None
                    self.was_pinched = False
                elif is_middle_finger:
                    action_text = "靠北 😂"
                    self._show_kaobei(frame)

                # === Detect pinch RELEASE ===
                elif self.was_pinched and not is_pinched:
                    self.pinch_released = True
                    self.was_pinched = False

                    # Click on RELEASE
                    time_since_last = current_time - self.last_pinch_release
                    if time_since_last < DOUBLE_CLICK_INTERVAL:
                        self.cursor.double_click()
                        action_text = "DOUBLE CLICK!"
                    else:
                        # Only single-click if we weren't doing a drag
                        if not (self.is_holding and self.cursor.is_dragging):
                            self.cursor.click()
                            action_text = "CLICK"

                    self.last_pinch_release = current_time

                    # End hold on release
                    if self.is_holding and self.cursor.is_dragging:
                        self.cursor.mouse_up()
                        action_text = "RELEASED"
                        self.is_holding = False
                        self.hold_start_time = None

                elif is_pinched:
                    self.was_pinched = True
                    self.pinch_released = False

                    # Start hold timer on first pinched frame
                    if not self.is_holding:
                        self.is_holding = True
                        self.hold_start_time = current_time

                    # Drag after HOLD_DURATION
                    hold_time = current_time - self.hold_start_time
                    if hold_time > HOLD_DURATION and not self.cursor.is_dragging:
                        self.cursor.mouse_down()
                        action_text = "DRAGGING"

                    if self.cursor.is_dragging:
                        action_text = "DRAGGING"

                else:
                    # Not pinched
                    if self.was_pinched:
                        self.pinch_released = False
                        self.was_pinched = False
                    if self.is_holding and not self.cursor.is_dragging:
                        self.is_holding = False
                        self.hold_start_time = None

                    if is_open_palm:
                        action_text = "Palm open - moving"

                if is_middle_finger:
                    action_text = "靠北 😂"
                    self._show_kaobei(frame)

                if DEBUG:
                    state = f"Pinch:{int(is_pinched)} Hold:{int(self.is_holding)} Drag:{int(self.cursor.is_dragging)}"
                    cv2.putText(frame, state, (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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
