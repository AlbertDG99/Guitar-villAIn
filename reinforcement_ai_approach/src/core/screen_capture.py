"""Optimized screen capture system for Guitar Hero."""

import logging
import threading
import time

import cv2
import numpy as np
import pyautogui

# Try to import mss at module level
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

# Basic logging configuration to see errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s:%(lineno)d | %(levelname)s | %(message)s')


class ScreenCapture:
    """Handles screen capture in a dedicated thread for maximum performance."""

    def __init__(self, config_manager):
        self.logger = logging.getLogger('ScreenCapture')
        self.config_manager = config_manager

        capture_config = self.config_manager.get_capture_area_config()
        if not capture_config:
            self.logger.error(
                "❌ Capture configuration is empty. Cannot start.")
            raise ValueError("Capture configuration cannot be None.")

        self.capture_left = int(capture_config['left'])
        self.capture_top = int(capture_config['top'])
        self.capture_width = int(capture_config['width'])
        self.capture_height = int(capture_config['height'])

        self.latest_frame = np.zeros(
            (self.capture_height, self.capture_width, 3), dtype=np.uint8)
        self.frame_lock = threading.Lock()

        self.is_running = False
        self.capture_thread = None

        # Initialize FPS tracking attributes
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

        self.use_mss = self._check_mss_availability()

    def _check_mss_availability(self) -> bool:
        """Checks if MSS is available for capture."""
        if MSS_AVAILABLE:
            self.logger.info(
                "✅ MSS is available and will be used for capture.")
            return True
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

        # No global MSS instance to close; handled in thread

        self.logger.info("⏹️ Screen capture thread stopped.")

    def _capture_loop(self):
        """The main loop that runs in the thread."""
        region = {
            'left': self.capture_left,
            'top': self.capture_top,
            'width': self.capture_width,
            'height': self.capture_height
        }

        # --- MSS capture logic (preferred) ---
        if self.use_mss:
            try:
                with mss.mss() as sct:
                    self.logger.info("🚀 Capture started with MSS in thread.")
                    while self.is_running:
                        screenshot = sct.grab(region)
                        frame = np.array(screenshot)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                        if frame.size == 0:
                            self.logger.warning(
                                "⚠️ Frame captured with MSS is empty. Skipping.")
                            continue

                        with self.frame_lock:
                            self.latest_frame = frame

                        time.sleep(0.001)
                return  # End thread execution if loop ends
            except Exception as e:
                self.logger.error(f"❌ Fatal error during MSS capture: {e}.")
                self.is_running = False
                # Don't return, so it can try with PyAutoGUI if still running

        # --- PyAutoGUI capture logic (fallback) ---
        if not self.is_running:
            return  # Exit if thread was stopped by MSS error

        self.logger.info(
            "🚀 Using PyAutoGUI for capture (fallback or primary method).")
        while self.is_running:
            try:
                screenshot = pyautogui.screenshot(
                    region=(
                        self.capture_left,
                        self.capture_top,
                        self.capture_width,
                        self.capture_height))
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                if frame.size == 0:
                    self.logger.warning(
                        "⚠️ Frame captured with PyAutoGUI is empty. Skipping.")
                    continue

                with self.frame_lock:
                    self.latest_frame = frame

            except Exception as e:
                self.logger.error(f"❌ Error in PyAutoGUI capture loop: {e}")
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

        # Save to configuration
        self.config_manager.config.set('CAPTURE', 'game_left', str(left))
        self.config_manager.config.set('CAPTURE', 'game_top', str(top))
        self.config_manager.config.set('CAPTURE', 'game_width', str(width))
        self.config_manager.config.set('CAPTURE', 'game_height', str(height))

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
