#!/usr/bin/env python3
"""
Real-time note detection and automatic key pressing for Guitar Hero.
"""

import os
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

import cv2
import numpy as np
import pydirectinput
from mss import mss

from .screen_capture import ScreenCapture
from .config_manager import ConfigManager
from .metrics import PerformanceTracker


class PolygonVisualizer:
    """Real-time note detection and automatic key pressing."""

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_manager = ConfigManager()
        self.screen_capture = ScreenCapture(self.config_manager)

        hsv_ranges = self.config_manager.get_hsv_ranges()
        self.morphology_params = self.config_manager.get_morphology_params()

        green_range = hsv_ranges['green']
        self.green_hsv = {
            'lower': np.array([green_range['h_min'], green_range['s_min'], green_range['v_min']]),
            'upper': np.array([green_range['h_max'], green_range['s_max'], green_range['v_max']])
        }

        yellow_range = hsv_ranges['yellow']
        self.yellow_hsv = {
            'lower': np.array([yellow_range['h_min'], yellow_range['s_min'], yellow_range['v_min']]),
            'upper': np.array([yellow_range['h_max'], yellow_range['s_max'], yellow_range['v_max']])
        }

        self.polygons = self.config_manager.get_note_lane_polygons_relative()

        self.running = True
        self.total_green_count = 0
        self.total_yellow_count = 0

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        self.lane_colors = {
            'S': (0, 255, 0), 'D': (0, 255, 255), 'F': (255, 0, 0),
            'J': (255, 0, 255), 'K': (0, 128, 255), 'L': (128, 255, 0)
        }

        self.input_enabled = True
        self.setup_input_system()

        # Performance tracking
        self.perf = PerformanceTracker(report_interval_seconds=1.0)

    def setup_input_system(self):
        """Configure input system and key mappings."""
        self.lane_to_key = {
            'S': 's', 'D': 'd', 'F': 'f',
            'J': 'j', 'K': 'k', 'L': 'l'
        }

        self.keys_pressed = {lane: False for lane in self.lane_to_key}
        self.green_hold_active = {lane: False for lane in self.lane_to_key}
        self.key_cooldowns = {lane: 0.0 for lane in self.lane_to_key}

        # Anti-ban configuration
        self.action_delay_range = (0.01, 0.03)
        self.yellow_press_probability = 1
        self.green_press_probability = 1
        self.cooldown_duration = 0.15
        self.green_note_cooldown = 0.1
        self.yellow_duration_range = (0.08, 0.15)
        self.fast_green_double_tap_duration = (0.05, 0.1)

        self.active_press_threads = {}
        self.press_thread_lock = threading.Lock()

        # Configure pydirectinput
        pydirectinput.FAILSAFE = False
        pydirectinput.PAUSE = 0.001

    def press_key_with_duration(self, key: str, duration: float, lane: str):
        """Press a key for a specific duration in a separate thread."""
        try:
            time.sleep(random.uniform(*self.action_delay_range))

            with self.press_thread_lock:
                self.active_press_threads[lane] = threading.current_thread()

            pydirectinput.keyDown(key)
            self.keys_pressed[lane] = True
            time.sleep(duration)
            pydirectinput.keyUp(key)
            self.keys_pressed[lane] = False

        except Exception as e:
            print(f"ERROR Error en pulsación de {key}: {e}")
        finally:
            with self.press_thread_lock:
                if lane in self.active_press_threads:
                    del self.active_press_threads[lane]

    def _execute_start_green_hold(self, key: str, lane: str):
        """Start a green note hold in a separate thread."""
        try:
            time.sleep(random.uniform(*self.action_delay_range))
            pydirectinput.keyDown(key)
            self.keys_pressed[lane] = True
            print(f"GREEN HOLD START en carril {lane} (Tecla: {key})")
        except Exception as e:
            print(f"ERROR Error al iniciar hold verde de {key}: {e}")
            self.green_hold_active[lane] = False
            self.keys_pressed[lane] = False

    def _execute_end_green_hold(self, key: str, lane: str):
        """End a green note hold in a separate thread."""
        try:
            time.sleep(random.uniform(*self.action_delay_range))
            pydirectinput.keyUp(key)
            self.keys_pressed[lane] = False
            print(f"RED HOLD END en carril {lane} (Tecla: {key})")
        except Exception as e:
            print(f"ERROR Error al finalizar hold verde de {key}: {e}")
            self.keys_pressed[lane] = False

    def panic_release_all_keys(self):
        """Emergency function: release all keys immediately."""
        print("PANIC PANICO: Soltando todas las teclas...")
        try:
            for lane, key in self.lane_to_key.items():
                pydirectinput.keyUp(key)
                self.keys_pressed[lane] = False
                self.green_hold_active[lane] = False

            with self.press_thread_lock:
                self.active_press_threads.clear()

            print("OK Todas las teclas liberadas correctamente")
        except Exception as e:
            print(f"ERROR Error en función de pánico: {e}")

    def handle_yellow_note(self, lane: str):
        """Handle yellow note detection with anti-ban logic."""
        current_time = time.time()

        if current_time - self.key_cooldowns[lane] < self.cooldown_duration:
            return

        if self.keys_pressed[lane]:
            return

        if random.random() > self.yellow_press_probability:
            return

        duration = random.uniform(*self.yellow_duration_range)
        self.key_cooldowns[lane] = current_time

        key = self.lane_to_key[lane]
        press_thread = threading.Thread(
            target=self.press_key_with_duration,
            args=(key, duration, lane),
            daemon=True
        )
        press_thread.start()

    def handle_green_note(self, lane: str, count: int):
        """Handles green note detection, with special logic for doubles."""
        current_time = time.time()

        # --- Special case: Double green note detected simultaneously ---
        if count >= 2:
            # Use fast general cooldown for this special action
            if current_time - \
                    self.key_cooldowns[lane] < self.cooldown_duration:
                return

            # Don't act if there's already a key pressed to avoid conflicts
            if self.keys_pressed[lane]:
                return

            print(f"LIGHTNING Double green in {lane}. Executing fast tap.")

            key = self.lane_to_key[lane]
            duration = random.uniform(*self.fast_green_double_tap_duration)

            self.key_cooldowns[lane] = current_time

            # Reuse press thread with duration, ideal for this task
            press_thread = threading.Thread(
                target=self.press_key_with_duration,
                args=(key, duration, lane),
                daemon=True
            )
            press_thread.start()
            return  # Action for this lane is decided

        # --- Normal green note logic (start/end of hold) ---

        # Check specific cooldown for green notes to avoid re-triggering
        if current_time - self.key_cooldowns[lane] < self.green_note_cooldown:
            return

        # Anti-ban logic: random probability
        if random.random() > self.green_press_probability:
            return

        key = self.lane_to_key[lane]

        # If no hold is active, it's the first green note (start hold)
        if not self.green_hold_active[lane]:
            # Update state synchronously to avoid race conditions
            self.green_hold_active[lane] = True
            self.key_cooldowns[lane] = current_time

            # Launch press action in a thread to avoid blocking
            threading.Thread(
                target=self._execute_start_green_hold,
                args=(key, lane),
                daemon=True
            ).start()

        # If hold is already active, it's the second green note (end hold)
        else:
            # Update state synchronously
            self.green_hold_active[lane] = False
            self.key_cooldowns[lane] = current_time

            # Launch release action in a thread
            threading.Thread(
                target=self._execute_end_green_hold,
                args=(key, lane),
                daemon=True
            ).start()

    def process_ai_actions(self, detections: Dict):
        """Processes detections and executes AI actions"""
        # Process each lane
        for lane_name, lane_data in detections['lanes'].items():
            green_count = lane_data.get('green', 0)
            yellow_count = lane_data.get('yellow', 0)

            # Execute action based on detected note type
            if green_count > 0:
                self.handle_green_note(lane_name, green_count)
            if yellow_count > 0:
                self.handle_yellow_note(lane_name)

    def process_lane_micro_image(self, lane_data):
        """Process a complete lane in a micro-image."""
        lane_name, points, frame = lane_data

        pts = np.array(points, np.int32)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(frame.shape[1], x_max + margin)
        y_max = min(frame.shape[0], y_max + margin)

        micro_frame = frame[y_min:y_max, x_min:x_max]

        if micro_frame.size == 0:
            return {
                'lane_name': lane_name,
                'green_boxes': [],
                'yellow_boxes': [],
                'green_count': 0,
                'yellow_count': 0
            }

        local_polygon = pts - np.array([x_min, y_min])
        hsv_micro = cv2.cvtColor(micro_frame, cv2.COLOR_BGR2HSV)

        # Green note detection
        green_mask = cv2.inRange(
            hsv_micro,
            self.green_hsv['lower'],
            self.green_hsv['upper'])

        # Simplified green detection - same as yellow
        close_size_green = max(3, self.morphology_params['close_size'] // 2)  # Same as yellow
        dilate_size_green = max(2, self.morphology_params['dilate_size'] // 2)  # Same as yellow

        close_kernel_green = np.ones(
            (close_size_green, close_size_green), np.uint8)
        green_mask = cv2.morphologyEx(
            green_mask, cv2.MORPH_CLOSE, close_kernel_green)

        dilate_kernel_green = np.ones(
            (dilate_size_green, dilate_size_green), np.uint8)
        green_mask = cv2.dilate(
            green_mask,
            dilate_kernel_green,
            iterations=1)

        open_kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(
            green_mask, cv2.MORPH_OPEN, open_kernel)

        green_contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lane_results = {
            'lane_name': lane_name,
            'green_boxes': [],
            'yellow_boxes': [],
            'green_count': 0,
            'yellow_count': 0
        }

        for contour in green_contours:
            area = cv2.contourArea(contour)
            if self.morphology_params['min_area'] <= area <= self.morphology_params['max_area']:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if cv2.pointPolygonTest(
                            local_polygon, (cx, cy), False) >= 0:
                        x, y, w, h = cv2.boundingRect(contour)
                        global_x = x + x_min
                        global_y = y + y_min

                        lane_results['green_boxes'].append({
                            'x': global_x, 'y': global_y, 'w': w, 'h': h, 'area': area
                        })
                        lane_results['green_count'] += 1

        if self.green_hold_active[lane_name]:
            return lane_results

        if lane_results['green_count'] > 0:
            return lane_results

        # Yellow note detection
        yellow_mask = cv2.inRange(
            hsv_micro,
            self.yellow_hsv['lower'],
            self.yellow_hsv['upper'])

        close_size_yellow = max(3, self.morphology_params['close_size'] // 2)
        dilate_size_yellow = max(2, self.morphology_params['dilate_size'] // 2)

        close_kernel_yellow = np.ones(
            (close_size_yellow, close_size_yellow), np.uint8)
        yellow_mask = cv2.morphologyEx(
            yellow_mask, cv2.MORPH_CLOSE, close_kernel_yellow)

        dilate_kernel_yellow = np.ones(
            (dilate_size_yellow, dilate_size_yellow), np.uint8)
        yellow_mask = cv2.dilate(
            yellow_mask,
            dilate_kernel_yellow,
            iterations=1)

        yellow_mask = cv2.morphologyEx(
            yellow_mask, cv2.MORPH_OPEN, open_kernel)

        # 11. Yellow contour search
        yellow_contours, _ = cv2.findContours(
            yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 12. Filter yellow contours
        for contour in yellow_contours:
            area = cv2.contourArea(contour)
            if self.morphology_params['min_area'] <= area <= self.morphology_params['max_area']:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if cv2.pointPolygonTest(
                            local_polygon, (cx, cy), False) >= 0:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Convert local coordinates to global
                        global_x = x + x_min
                        global_y = y + y_min

                        lane_results['yellow_boxes'].append({
                            'x': global_x, 'y': global_y, 'w': w, 'h': h, 'area': area
                        })
                        lane_results['yellow_count'] += 1

        return lane_results

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame using real parallelism per lane"""
        detections = {
            'yellow': 0,
            'green': 0,
            'lanes': {}
        }

        # --- PREPARATION ---
        frame_start_t = time.perf_counter()
        output_frame = frame.copy()

        # --- REAL PARALLELISM: Each thread processes a complete lane ---
        lane_tasks = []
        for lane_name, points in self.polygons.items():
            if points:
                lane_tasks.append((lane_name, points, frame))

        total_green_detections = 0
        total_yellow_detections = 0

        if lane_tasks:
            # Use all available cores (up to 6 lanes)
            max_workers = min(len(lane_tasks), 6)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_results = [
                    executor.submit(
                        self.process_lane_micro_image,
                        task) for task in lane_tasks]

                # Collect results
                for future in future_results:
                    result = future.result()
                    lane_name = result['lane_name']

                    # Save lane statistics
                    detections['lanes'][lane_name] = {
                        'yellow': result['yellow_count'],
                        'green': result['green_count']
                    }

                    # Accumulate totals
                    total_green_detections += result['green_count']
                    total_yellow_detections += result['yellow_count']

                    # Draw detection boxes
                    for box in result['green_boxes']:
                        cv2.rectangle(
                            output_frame,
                            (box['x'],
                             box['y']),
                            (box['x'] +
                             box['w'],
                                box['y'] +
                                box['h']),
                            (0,
                             255,
                             0),
                            2)

                    for box in result['yellow_boxes']:
                        cv2.rectangle(
                            output_frame,
                            (box['x'],
                             box['y']),
                            (box['x'] +
                             box['w'],
                                box['y'] +
                                box['h']),
                            (0,
                             255,
                             255),
                            2)

        # --- AI PROCESSING ---
        if self.input_enabled:
            self.process_ai_actions(detections)

        # --- POLYGON AND EFFECTS DRAWING ---
        overlay = output_frame.copy()

        # First, draw the fills of pressed keys in the overlay
        for lane_name, points in self.polygons.items():
            if self.keys_pressed.get(lane_name, False):
                pts = np.array(points, np.int32)
                color = self.lane_colors.get(lane_name, (255, 255, 255))
                if color:
                    cv2.fillPoly(overlay, [pts], color)

        # Blend overlay with output frame
        alpha = 0.5  # 50% transparency
        cv2.addWeighted(
            overlay,
            alpha,
            output_frame,
            1 - alpha,
            0,
            output_frame)

        # Now, draw contours and labels on the already blended frame
        for lane_name, points in self.polygons.items():
            pts = np.array(points, np.int32)
            color = self.lane_colors.get(lane_name, (255, 255, 255))

            # Draw contour
            cv2.polylines(
                output_frame,
                [pts],
                isClosed=True,
                color=color,
                thickness=2)

            # Lane label
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(output_frame, lane_name, tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Update global counters
        detections['yellow'] = total_yellow_detections
        detections['green'] = total_green_detections
        self.total_green_count = total_green_detections
        self.total_yellow_count = total_yellow_detections

        # ADD TOP COUNTER
        self.add_top_counter(output_frame)

        # Calculate FPS
        self.fps_counter += 1
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = time.time()

        # Only show FPS (bottom right corner)
        fps_text = f"FPS: {self.current_fps:.1f}"
        text_size = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

        # Bottom right corner position
        fps_x = output_frame.shape[1] - text_size[0] - 10
        fps_y = output_frame.shape[0] - 10

        # Background for FPS
        overlay_fps = output_frame.copy()
        cv2.rectangle(overlay_fps, (fps_x - 5, fps_y - text_size[1] - 5),
                      (fps_x + text_size[0] + 5, fps_y + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay_fps, 0.7, output_frame, 0.3, 0, output_frame)

        # FPS text
        cv2.putText(output_frame, fps_text, (fps_x, fps_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Report latency
        latency_ms = (time.perf_counter() - frame_start_t) * 1000.0
        self.perf.on_frame_processed(latency_ms)

        return output_frame, detections

    def add_top_counter(self, frame: np.ndarray):
        """Add top counter with real-time detections"""
        # Background for counter (top part)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Main counter text
        counter_text = f"GREENS: {self.total_green_count}  |  YELLOWS: {self.total_yellow_count}"

        # Calculate centered position
        text_size = cv2.getTextSize(
            counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        x = (frame.shape[1] - text_size[0]) // 2
        y = 45

        # Draw text with shadow
        cv2.putText(frame, counter_text, (x + 2, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)  # Shadow
        cv2.putText(
            frame,
            counter_text,
            (x,
             y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255,
             255,
             255),
            3)  # Main text

        # AI system status
        if self.input_enabled:
            ai_text = "ROBOT AI ACTIVE | Anti-Ban: ON"
            ai_color = (0, 255, 0)  # Green
        else:
            ai_text = "EYE VISUALIZATION ONLY"
            ai_color = (0, 255, 255)  # Yellow

        cv2.putText(frame, ai_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ai_color, 2)

        # Pressed keys indicator
        pressed_keys = [
            lane for lane,
            pressed in self.keys_pressed.items() if pressed]
        if pressed_keys:
            keys_text = f"KEYS: {', '.join(pressed_keys)}"
            cv2.putText(frame, keys_text, (frame.shape[1] - 200, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        """Main bot execution loop."""
        print("\n" + "=" * 50)
        print("ROCKET Guitar Hero AI Bot started.")
        print("FIRE REAL key presses activated!")
        print("Press 'Q' in the capture window to exit.")
        print("Press 'SPACE' in the capture window for panic function (release all keys).")
        print("=" * 50 + "\n")

        window_name = 'Guitar Hero AI - Color Pattern Approach'
        try:
            # Create a named window to be able to manipulate it
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            with mss() as sct:
                monitors = sct.monitors
                target_monitor = None
                # Monitor [0] is the virtual one that covers all. [1] is usually the primary.
                # Look for a monitor that's not at position (0,0) (secondary)
                if len(monitors) > 2:
                    for monitor in monitors[1:]:
                        if monitor['left'] != 0 or monitor['top'] != 0:
                            target_monitor = monitor
                            break

                # If no secondary monitor found, use the primary
                if not target_monitor and len(monitors) > 1:
                    target_monitor = monitors[1]

                if target_monitor:
                    print(
                        f"MONITOR Moving and maximizing window on monitor ({target_monitor['left']}, {target_monitor['top']})")
                    cv2.moveWindow(
                        window_name,
                        target_monitor['left'],
                        target_monitor['top'])
                    cv2.resizeWindow(
                        window_name,
                        target_monitor['width'],
                        target_monitor['height'])
                else:
                    print("WARNING No monitor detected. Using default size.")

        except Exception as e:
            print(f"ERROR Error configuring window on secondary monitor: {e}")
            print("   Will continue with default window.")

        self.screen_capture.start()
        time.sleep(1)  # Give time for capture to start

        try:
            while self.running:
                frame = self.screen_capture.get_latest_frame()

                if frame is not None:
                    output_frame, detections = self.process_frame(frame)

                    cv2.imshow(window_name, output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord(' '):
                    self.panic_release_all_keys()

        except KeyboardInterrupt:
            print("\nSTOP User interruption detected.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleans all resources before exiting."""
        print("REFRESH Cleaning resources...")
        self.running = False
        self.screen_capture.stop()
        self.panic_release_all_keys()  # Ensures all keys are released
        cv2.destroyAllWindows()
        print("OK Cleanup complete. Goodbye!")


def main():
    """Main function"""
    try:
        visualizer = PolygonVisualizer()
        visualizer.run()

    except Exception as e:
        print(f"\nERROR A fatal error has occurred: {e}")
        traceback.print_exc()
        input("\nPress ENTER to exit.")


if __name__ == "__main__":
    main()
