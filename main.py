"""
Hand Gesture Cursor Control - Main Application
Based on proven working implementations from:
- https://github.com/varadganjoo/hand-tracking-with-controls
- https://github.com/aleafarrel-id/alea-aircursor
- https://towardsdatascience.com/i-ditched-my-mouse-how-i-control-my-computer-with-hand-gestures-in-60-lines-of-python/
"""

import cv2
import time
import numpy as np
import pyautogui

from config import (
    CAM_WIDTH, CAM_HEIGHT, WINDOW_NAME,
    PINCH_ON_THRESHOLD, PINCH_OFF_THRESHOLD, PINCH_DEBOUNCE_FRAMES,
    HOLD_DURATION, DOUBLE_CLICK_INTERVAL, CURSOR_SENSITIVITY
)
from hand_tracker import HandTracker

pyautogui.FAILSAFE = False


class HandGestureApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for lower latency
        self.cap.set(3, CAM_WIDTH)
        self.cap.set(4, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Force 30fps for smoother capture
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.hand_tracker = HandTracker()
        self.screen_width, self.screen_height = pyautogui.size()

        # Pinch state
        self.is_pinched = False
        self.pinch_frame_count = 0
        self.last_click_time = 0
        self.last_release_time = 0

        # Drag state
        self.is_dragging = False
        self.drag_start_time = 0

        print(f"🖐️ {WINDOW_NAME} Started")
        print("   Controls:")
        print("   - Index finger: Move cursor")
        print("   - Pinch (thumb + index): Click")
        print("   - Hold pinch: Drag")
        print("   - Press 'q' to quit")

    def get_distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hand_tracker.process_frame(rgb_frame)
            current_time = time.time()

            action_text = "Waiting for hand..."

            if results.hand_landmarks:
                self.hand_tracker.draw_landmarks(frame, results)
                landmarks = results.hand_landmarks[0]

                index_tip = landmarks[8]   # Index finger tip
                thumb_tip = landmarks[4]   # Thumb tip

                # Draw landmarks
                cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 15, (0, 255, 0), -1)
                cv2.circle(frame, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 15, (0, 0, 255), -1)

                # Calculate cursor position from index finger with sensitivity
                # Use wrist as reference for relative movement
                wrist = landmarks[0]
                dx = (index_tip.x - wrist.x) * CURSOR_SENSITIVITY
                dy = (index_tip.y - wrist.y) * CURSOR_SENSITIVITY

                # Map to screen coordinates (centered on wrist position)
                cursor_x = int(np.interp(0.5 + dx, (0, 1), (0, self.screen_width)))
                cursor_y = int(np.interp(0.5 + dy, (0, 1), (0, self.screen_height)))

                pyautogui.moveTo(cursor_x, cursor_y, duration=0)
                action_text = "Moving cursor"

                # Pinch detection
                distance = self.get_distance(thumb_tip, index_tip)
                pinch_threshold = PINCH_ON_THRESHOLD if not self.is_pinched else PINCH_OFF_THRESHOLD

                if distance < pinch_threshold:
                    self.pinch_frame_count += 1
                    if self.pinch_frame_count >= PINCH_DEBOUNCE_FRAMES:
                        if not self.is_pinched:
                            # Just started pinching
                            self.is_pinched = True
                            self.drag_start_time = current_time
                            action_text = "Pinch detected"
                        else:
                            # Holding pinch - check for drag
                            hold_time = current_time - self.drag_start_time
                            if hold_time > HOLD_DURATION and not self.is_dragging:
                                self.is_dragging = True
                                pyautogui.mouseDown()
                                action_text = "DRAGGING"
                            elif self.is_dragging:
                                action_text = "DRAGGING"
                else:
                    if self.is_pinched:
                        # Released pinch
                        time_since_release = current_time - self.last_release_time

                        if self.is_dragging:
                            # End drag
                            pyautogui.mouseUp()
                            action_text = "Released drag"
                        elif time_since_release < DOUBLE_CLICK_INTERVAL:
                            # Double click
                            pyautogui.doubleClick()
                            action_text = "DOUBLE CLICK!"
                        else:
                            # Single click
                            if current_time - self.last_click_time > DOUBLE_CLICK_INTERVAL:
                                pyautogui.click()
                                action_text = "CLICK"

                        self.last_release_time = current_time
                        self.is_pinched = False
                        self.is_dragging = False
                        self.pinch_frame_count = 0

                    action_text = "Hand open"

                # Show pinch distance
                cv2.putText(frame, f"Distance: {distance:.3f}", (10, h - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            else:
                # No hand - reset
                self.is_pinched = False
                self.is_dragging = False
                self.pinch_frame_count = 0
                if self.is_dragging:
                    pyautogui.mouseUp()

            # Draw status
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(frame, "Index=Move | Pinch=Click | Hold=Drag | 'q'=Quit",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, action_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        if self.is_dragging:
            pyautogui.mouseUp()
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
