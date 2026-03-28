import cv2
import mediapipe as mp
import numpy as np
import sys
import time
from pynput.mouse import Controller, Button
from pynput import keyboard
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QRect
from PyQt6.QtGui import QPainter, QColor, QPen

# Smoothing Filter
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev, self.dx_prev, self.last_time = None, 0, None

    def __call__(self, x):
        t = time.time()
        if self.last_time is None:
            self.last_time, self.x_prev = t, x
            return x
        te = t - self.last_time
        dx = (x - self.x_prev) / te
        edx = self._low_pass(self.dx_prev, dx, self._alpha(te, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        res = self._low_pass(self.x_prev, x, self._alpha(te, cutoff))
        self.x_prev, self.dx_prev, self.last_time = res, edx, t
        return res

    def _alpha(self, te, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def _low_pass(self, prev, curr, alpha):
        return alpha * curr + (1.0 - alpha) * prev

# The Tracking Engine (Runs in background)
class TrackingThread(QThread):
    # Sends: (landmarks, is_clicked, is_scrolling, is_switching)
    data_signal = pyqtSignal(list, int, int, bool, bool, bool, bool)

    def __init__(self):
        super().__init__()
        self.mouse = Controller()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.prev_wrist = None
        self.cap = cv2.VideoCapture(0)
        self.running = True
        
        # State & Sensitivity
        self.is_calibrated = False
        self.should_calibrate = False
        self.is_clicked = False
        self.is_right_clicked = False
        self.is_tracking_paused = False
        self.origin_cx, self.origin_cy = 0.5, 0.5
        self.sensitivity = 1.5
        
        self.physical_center_x, self.physical_center_y = 0, 0
        self.current_screen_idx = 0
        
        # Filters
        self.filter_x = OneEuroFilter(min_cutoff=0.05, beta=0.05)
        self.filter_y = OneEuroFilter(min_cutoff=0.05, beta=0.05)
        
        # Gesture Timing
        self.prev_scroll_y = 0
        self.prev_palm_x = 0
        self.last_switch_time = 0
        
    def run(self):
        while self.running:
            success, img = self.cap.read()
            if not success: continue
            
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            lms_out, is_scrolling, px, py = [], False, 0.5, 0.5

            if results.multi_hand_landmarks:
                best_hand_lms = None
                max_score = -1
                
                for hand_landmarks in results.multi_hand_landmarks:
                    lms_eval = hand_landmarks.landmark
                    
                    # Calculate openness score (number of fingers extended)
                    fingers_up = 0
                    if lms_eval[8].y < lms_eval[6].y: fingers_up += 1
                    if lms_eval[12].y < lms_eval[10].y: fingers_up += 1
                    if lms_eval[16].y < lms_eval[14].y: fingers_up += 1
                    if lms_eval[20].y < lms_eval[18].y: fingers_up += 1
                    
                    score = fingers_up * 10.0
                    
                    # Apply historical hysteresis bias to prevent flickering between two equally open hands
                    if self.prev_wrist and np.hypot(lms_eval[0].x - self.prev_wrist[0], lms_eval[0].y - self.prev_wrist[1]) < 0.3:
                        score += 5.0
                        
                    if score > max_score:
                        max_score = score
                        best_hand_lms = lms_eval
                
                # Funnel the dominant active hand directly into the application pipeline
                lms = best_hand_lms
                self.prev_wrist = (lms[0].x, lms[0].y)
                
                # Finger States
                index_up = lms[8].y < lms[5].y
                middle_up = lms[12].y < lms[9].y
                ring_up = lms[16].y < lms[13].y
                pinky_up = lms[20].y < lms[17].y
                
                # Palm Centroid
                palm_x = (lms[0].x + lms[5].x + lms[17].x) / 3
                palm_y = (lms[0].y + lms[5].y + lms[17].y) / 3

                # 1. Calibration
                if self.should_calibrate:
                    # Anchor directly to wherever the mouse cursor is located physically right now
                    mx, my = self.mouse.position
                    self.physical_center_x, self.physical_center_y = mx, my
                    
                    self.origin_cx, self.origin_cy = palm_x, palm_y
                    self.is_calibrated = True
                    self.should_calibrate = False

                # 2. Gesture: Scrolling (2 Fingers)
                is_scrolling = index_up and middle_up and not ring_up
                if is_scrolling:
                    if self.prev_scroll_y == 0: self.prev_scroll_y = palm_y
                    dy = self.prev_scroll_y - palm_y
                    if abs(dy) > 0.005:
                        self.mouse.scroll(0, int(-dy * 300)) # Natural scroll
                        self.prev_scroll_y = palm_y
                else:
                    self.prev_scroll_y = 0

                # 3. Movement and Pause Logic
                fingers_extended = sum([index_up, middle_up, ring_up, pinky_up])
                
                if fingers_extended <= 0:
                    self.is_tracking_paused = True
                elif fingers_extended >= 4:
                    if self.is_tracking_paused:
                        # Re-anchor tracking immediately to where the hand is physically located to prevent jumping
                        self.origin_cx, self.origin_cy = palm_x, palm_y
                        mx, my = self.mouse.position
                        self.physical_center_x, self.physical_center_y = mx, my
                    self.is_tracking_paused = False

                if self.is_calibrated and not is_scrolling and not self.is_tracking_paused:
                    offset_x = (palm_x - self.origin_cx) * self.sensitivity
                    offset_y = (palm_y - self.origin_cy) * self.sensitivity
                    
                    v_rect = QApplication.primaryScreen().virtualGeometry()
                    ratio = QApplication.primaryScreen().devicePixelRatio()
                    
                    # Compute unified array size spanning all monitors completely
                    phys_width = v_rect.width() * ratio
                    phys_height = v_rect.height() * ratio
                    
                    target_x = self.physical_center_x + (offset_x * phys_width)
                    target_y = self.physical_center_y + (offset_y * phys_height)
                    
                    self.mouse.position = (int(self.filter_x(target_x)),
                                           int(self.filter_y(target_y)))

                # 5. Clicking
                dist_left = np.hypot(lms[8].x - lms[4].x, lms[8].y - lms[4].y)
                is_left_clicked = dist_left < 0.04
                if is_left_clicked and not self.is_clicked and not self.is_tracking_paused:
                    self.mouse.press(Button.left)
                    self.is_clicked = True
                elif not is_left_clicked and self.is_clicked:
                    self.mouse.release(Button.left)
                    self.is_clicked = False
                    
                dist_right = np.hypot(lms[12].x - lms[4].x, lms[12].y - lms[4].y)
                is_right_clicked = dist_right < 0.04
                if is_right_clicked and not self.is_right_clicked and not self.is_tracking_paused:
                    self.mouse.press(Button.right)
                    self.is_right_clicked = True
                elif not is_right_clicked and self.is_right_clicked:
                    self.mouse.release(Button.right)
                    self.is_right_clicked = False
                    
                mx, my = self.mouse.position
                
                # Pull logical pos directly from QCursor to completely sidestep DPI offset mismatches
                from PyQt6.QtGui import QCursor
                logical_pos = QCursor.pos()
                
                # Emit any active clicking state to turn the UI red
                any_clicked = self.is_clicked or self.is_right_clicked
                self.data_signal.emit(list(lms), logical_pos.x(), logical_pos.y(), any_clicked, is_scrolling, self.is_calibrated, self.is_tracking_paused)
            
            time.sleep(0.01)

        # Cleanly release the camera and mediapipe resources on exit
        self.cap.release()
        self.hands.close()



# Individual Overlay Window
class ScreenOverlay(QMainWindow):
    def __init__(self, screen_obj):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(screen_obj.geometry())
        
        self.hand_points = []
        self.mode_text = ""
        self.mode_color = QColor(0, 255, 255, 120)
        self.is_calibrated = False
        self.show()

    def update_frame(self, lms, mx, my, clicked, scrolling, calibrated, is_paused):
        self.is_calibrated = calibrated
        self.is_paused = is_paused
        
        # Focus Logic: Only draw hand if mouse is on THIS screen
        if lms and self.geometry().contains(mx, my):
            # Calculate translation to perfectly align the visual palm with the internal tracking mouse
            palm_x = (lms[0].x + lms[5].x + lms[17].x) / 3.0
            palm_y = (lms[0].y + lms[5].y + lms[17].y) / 3.0
            
            local_mx = mx - self.x()
            local_my = my - self.y()
            
            shift_x = local_mx - (palm_x * self.width())
            shift_y = local_my - (palm_y * self.height())
            
            self.hand_points = [(int(lm.x * self.width() + shift_x), int(lm.y * self.height() + shift_y)) for lm in lms]
            
            if getattr(self, 'is_paused', False): self.mode_color = QColor(100, 100, 100, 180)
            elif clicked: self.mode_color = QColor(255, 0, 0, 180)
            elif scrolling: self.mode_color = QColor(255, 255, 0, 180)
            else: self.mode_color = QColor(0, 255, 255, 120)
        else:
            self.hand_points = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Always draw the calibration text prompt (useful on all screens)
        if not self.is_calibrated:
            painter.setPen(QColor(255, 255, 255, 200))
            painter.drawText(30, 50, "❌ NOT CALIBRATED - Press 'ESC'")
        elif getattr(self, 'is_paused', False):
            painter.setPen(QColor(255, 255, 255, 200))
            painter.drawText(30, 50, "⏸ PAUSED - Open Palm to Resume")
        
        # Draw Ghost Hand (only if hand_points list is populated by the focused check)
        if self.hand_points:
            painter.setPen(QPen(self.mode_color, 3))
            for pt in self.hand_points:
                painter.drawEllipse(pt[0], pt[1], 6, 6)

# Main Controller
class GestureApp:
    def __init__(self):
        self.qt_app = QApplication(sys.argv)
        self.overlays = [ScreenOverlay(s) for s in self.qt_app.screens()]
        
        self.tracker = TrackingThread()
        self.tracker.data_signal.connect(self.sync_overlays)
        self.tracker.finished.connect(self.qt_app.quit)
        self.tracker.start()

        self.listener = keyboard.GlobalHotKeys({
            '<esc>': self.calibrate_app,
            '<ctrl>+<esc>': self.quit_app
        })
        self.listener.start()

    def calibrate_app(self):
        print("Global Hotkey: Calibrating...")
        self.tracker.should_calibrate = True

    def quit_app(self):
        print("Global Hotkey: Quitting...")
        self.tracker.running = False
        return False

    def sync_overlays(self, lms, mx, my, clicked, scrolling, calibrated, switching):
        for o in self.overlays:
            o.update_frame(lms, mx, my, clicked, scrolling, calibrated, switching)

    def run(self):
        # Global key listener for calibration (ESC) and quit (CTRL+ESC)
        self.qt_app.exec()
        sys.exit(0)

if __name__ == "__main__":
    GestureApp().run()