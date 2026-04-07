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

from config import (
    CAM_WIDTH, CAM_HEIGHT, WINDOW_NAME,
    DOUBLE_CLICK_INTERVAL, HOLD_DURATION, DEBUG
)
from hand_tracker import HandTracker
from gesture import GestureRecognizer
from cursor import CursorController


class HandGestureApp:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAM_WIDTH)
        self.cap.set(4, CAM_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.hand_tracker = HandTracker()
        self.gesture_recognizer = GestureRecognizer()
        self.cursor = CursorController()

        self.was_pinched = False
        self.last_pinch_release = 0
        self.is_holding = False
        self.hold_start_time = None
        self.home_position = None
        self.running = True

        print(f"🖐️ {WINDOW_NAME} Started")
        print("   Controls:")
        print("   - Open palm: Move cursor")
        print("   - Pinch: Click")
        print("   - Double pinch: Double click")
        print("   - Pinch + hold: Drag")
        print("   - Press 'q' to quit")

    def run(self):
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

                landmarks = results.hand_landmarks[0]
                current_time = time.time()

                is_pinched = self.gesture_recognizer.is_pinched(landmarks)
                is_open_palm = self.gesture_recognizer.is_open_palm(landmarks)
                is_middle_finger = self.gesture_recognizer.is_middle_finger(landmarks)

                # Set home position on first frame seen
                if self.home_position is None:
                    thumb = landmarks[4]
                    index = landmarks[8]
                    self.home_position = (
                        (thumb.x + index.x) / 2.0,
                        (thumb.y + index.y) / 2.0
                    )

                # Cursor follows hand relative to home position
                cursor_pos = self.gesture_recognizer.get_cursor_position(
                    landmarks, self.home_position
                )
                if cursor_pos:
                    self.cursor.move_to(cursor_pos[0], cursor_pos[1])
                action_text = "Moving cursor"

                if is_middle_finger:
                    action_text = "靠北 😂"
                    self._show_kaobei(frame)

                elif is_pinched:
                    if not self.was_pinched:
                        # First pinch frame
                        self.was_pinched = True
                        self.is_holding = True
                        self.hold_start_time = current_time
                    else:
                        # Held pinch → drag after HOLD_DURATION
                        hold_time = current_time - self.hold_start_time
                        if hold_time > HOLD_DURATION and not self.cursor.is_dragging:
                            self.cursor.mouse_down()
                            action_text = "DRAGGING"
                        elif self.cursor.is_dragging:
                            action_text = "DRAGGING"

                else:
                    if self.was_pinched:
                        # Pinch released
                        time_since_last = current_time - self.last_pinch_release
                        if time_since_last < DOUBLE_CLICK_INTERVAL:
                            self.cursor.double_click()
                            action_text = "DOUBLE CLICK!"
                        else:
                            if not (self.is_holding and self.cursor.is_dragging):
                                self.cursor.click()
                                action_text = "CLICK"

                        self.last_pinch_release = current_time

                        if self.is_holding and self.cursor.is_dragging:
                            self.cursor.mouse_up()
                            action_text = "RELEASED"

                        self.was_pinched = False
                        self.is_holding = False
                        self.hold_start_time = None

                    if is_open_palm:
                        action_text = "Palm open - moving"

                if DEBUG:
                    state = f"Pinch:{int(is_pinched)} Drag:{int(self.cursor.is_dragging)}"
                    cv2.putText(frame, state, (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            else:
                # No hand detected — reset
                self.home_position = None
                if self.is_holding and self.cursor.is_dragging:
                    self.cursor.mouse_up()
                self.is_holding = False
                self.was_pinched = False

            self._draw_overlay(frame, action_text)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False

        self.cleanup()

    def _draw_overlay(self, frame, action_text):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "Pinch=Click | DoublePinch=DoubleClick | PinchHold=Drag | Press 'q' to quit",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            frame, action_text,
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    def _show_kaobei(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(
            frame, "靠北",
            (w // 2 - 100, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10
        )

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_tracker.close()
        print(f"👋 {WINDOW_NAME} Stopped")


def main():
    try:
        app = HandGestureApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
