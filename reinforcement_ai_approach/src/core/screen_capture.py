"""Optimized screen capture system for Guitar Hero."""

import time
import threading
import cv2
import numpy as np
import pyautogui
import configparser

from src.utils.logger import setup_logger


class ScreenCapture:
    """Handles screen capture in a dedicated thread for maximum performance."""

    def __init__(self, capture_config: dict):
        self.logger = setup_logger('ScreenCapture')
        self.config = capture_config

        self.capture_left = int(self.config['left'])
        self.capture_top = int(self.config['top'])
        self.capture_width = int(self.config['width'])
        self.capture_height = int(self.config['height'])

        self.latest_frame = np.zeros(
            (self.capture_height, self.capture_width, 3), dtype=np.uint8)
        self.frame_lock = threading.Lock()

        self.is_running = False
        self.capture_thread = None

        self.mss_instance = None
        self.use_mss = self._initialize_mss()

    def _initialize_mss(self) -> bool:
        """Attempts to initialize MSS for fast capture."""
        try:
            import mss
            self.mss_instance = mss.mss()
            self.logger.info(
                "✅ MSS initialized correctly for threaded capture.")
            return True
        except ImportError:
            self.logger.warning(
                "⚠️ MSS not available, using PyAutoGUI (slower).")
            return False

    def start(self):
        """Starts the capture thread in the background."""
        if self.is_running:
            self.logger.warning("The capture thread is already running.")
            return

        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.logger.info("▶️ Screen capture thread started.")

    def stop(self):
        """Stops the capture thread."""
        self.is_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        if self.use_mss and self.mss_instance:
            self.mss_instance.close()

        self.logger.info("⏹️ Screen capture thread stopped.")

    def _capture_loop(self):
        """The main loop that runs in the thread."""
        region = {
            'left': self.capture_left,
            'top': self.capture_top,
            'width': self.capture_width,
            'height': self.capture_height
        }

        while self.is_running:
            try:
                if self.use_mss and self.mss_instance:
                    screenshot = self.mss_instance.grab(region)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    screenshot = pyautogui.screenshot(
                        region=(
                            self.capture_left,
                            self.capture_top,
                            self.capture_width,
                            self.capture_height))
                    frame = cv2.cvtColor(
                        np.array(screenshot), cv2.COLOR_RGB2BGR)

                with self.frame_lock:
                    self.latest_frame = frame

            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                self.is_running = False  # Stop in case of serious error
                break
            time.sleep(0.001)

    def get_latest_frame(self) -> np.ndarray:
        """
        Gets the latest captured frame safely.
        This is the function that the environment will call. It's very fast.
        """
        with self.frame_lock:
            return self.latest_frame.copy()

    def calibrate_region(self, display_duration=3.0):
        """Calibrate capture region by showing preview."""
        self.logger.info("Starting capture region calibration...")
        self.start()
        time.sleep(1)

        start_time = time.time()
        while time.time() - start_time < display_duration:
            frame = self.get_latest_frame()
            if frame.size == 0:
                self.logger.error("❌ Could not get frame for calibration")
                break

            preview = frame.copy()
            cv2.rectangle(preview, (0, 0), (self.capture_width - 1,
                          self.capture_height - 1), (0, 255, 0), 2)
            cv2.putText(preview, "Capture Region", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture Calibration", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.stop()
        self.logger.info("Calibration finished.")

    def get_capture_info(self):
        """Get current capture information"""
        return {
            'left': self.capture_left,
            'top': self.capture_top,
            'width': self.capture_width,
            'height': self.capture_height,
            'method': 'MSS' if self.use_mss else 'PyAutoGUI',
            'active': self.is_running
        }

    def __del__(self):
        """Cleanup when destroying the instance"""
        self.stop()

    def load_configuration(self):
        """Load configuration from the configuration dictionary."""
        try:
            self.capture_left = int(self.config['left'])
            self.capture_top = int(self.config['top'])
            self.capture_width = int(self.config['width'])
            self.capture_height = int(self.config['height'])

            self.logger.info("Capture configuration loaded: %dx%d at (%d, %d)",
                             self.capture_width, self.capture_height,
                             self.capture_left, self.capture_top)

        except (KeyError, TypeError) as error:
            self.logger.error(
                "Fatal error in capture configuration: %s", error)
            self.logger.error(
                "Make sure the configuration dictionary contains 'left', 'top', 'width', 'height'.")
            raise
        except ValueError as error:
            self.logger.error(
                "Fatal error: one of the capture values is not a valid integer: %s", error)
            raise

    def get_monitor_region(self):
        """Get specific region of the target monitor"""
        try:
            return {
                'left': self.capture_left,
                'top': self.capture_top,
                'width': self.capture_width,
                'height': self.capture_height
            }

        except (IndexError, AttributeError, ValueError) as error:
            self.logger.warning("⚠️ Error getting monitor region: %s", error)
            return None

    def update_region(self, left, top, width, height):
        """Update capture region"""
        self.capture_left = left
        self.capture_top = top
        self.capture_width = width
        self.capture_height = height

        self.config['left'] = str(left)
        self.config['top'] = str(top)
        self.config['width'] = str(width)
        self.config['height'] = str(height)

        self.logger.info("📐 Region updated: %dx%d at (%d, %d)",
                         width, height, left, top)

    def get_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

        self.frame_count += 1
        return self.fps
