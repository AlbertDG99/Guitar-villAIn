#!/usr/bin/env python3
"""Visualizes configured polygons and real-time note detection."""

import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, Tuple

from src.core.screen_capture import ScreenCapture
from src.utils.config_manager import ConfigManager
from src.core.score_detector import ScoreDetector


class PolygonVisualizer:
    """Real-time polygon, detection and score visualizer."""

    def __init__(self):
        self.config_manager = ConfigManager()
        # Initialize capture with dict config and start capture thread
        capture_cfg = self.config_manager.get_capture_area_config()
        self.screen_capture = ScreenCapture(capture_cfg)
        self.screen_capture.start()
        # Initialize score detector with configured region
        self.score_detector = ScoreDetector(self.config_manager.get_score_region())

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
        self.score_region = self.config_manager.get_score_region()

        self.total_green_count = 0
        self.total_yellow_count = 0

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        self.score_thread = None
        self.last_score_check_time = 0
        self.score_update_interval = 0.2
        self.last_known_score = 0

        self.view_mode = 0
        self.view_modes = ["NORMAL", "YELLOW MASK", "GREEN MASK"]

        self.lane_colors = {
            'S': (0, 255, 0), 'D': (0, 255, 255), 'F': (255, 0, 0),
            'J': (255, 0, 255), 'K': (0, 128, 255), 'L': (128, 255, 0)
        }

    def detect_yellow_notes(self, hsv: np.ndarray) -> np.ndarray:
        """Detect yellow notes using exact values from Plus file"""
        mask = cv2.inRange(
            hsv,
            self.yellow_hsv['lower'],
            self.yellow_hsv['upper'])

        close_size = max(3, self.morphology_params['close_size'] // 2)
        dilate_size = max(2, self.morphology_params['dilate_size'] // 2)

        close_kernel = np.ones((close_size, close_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

        open_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        return mask

    def detect_green_notes(self, hsv: np.ndarray) -> np.ndarray:
        """Detect green notes using EXACT values from Plus file"""
        mask = cv2.inRange(
            hsv,
            self.green_hsv['lower'],
            self.green_hsv['upper'])

        close_size = self.morphology_params['close_size']
        dilate_size = self.morphology_params['dilate_size']

        close_kernel = np.ones((close_size, close_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

        open_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        return mask

    def process_lane_optimized(self, lane_data):
        """Process a specific lane in optimized way (for threading)"""
        lane_name, points, green_contours, yellow_contours = lane_data

        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        pts = np.array(points, np.int32)
        cv2.fillPoly(mask, [pts], (255,))

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

                    if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                        x, y, w, h = cv2.boundingRect(contour)
                        lane_results['green_boxes'].append({
                            'x': x, 'y': y, 'w': w, 'h': h, 'area': area
                        })
                        lane_results['green_count'] += 1

        if lane_results['green_count'] > 0:
            return lane_results

        for contour in yellow_contours:
            area = cv2.contourArea(contour)
            if self.morphology_params['min_area'] <= area <= self.morphology_params['max_area']:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                        x, y, w, h = cv2.boundingRect(contour)
                        lane_results['yellow_boxes'].append({
                            'x': x, 'y': y, 'w': w, 'h': h, 'area': area
                        })
                        lane_results['yellow_count'] += 1

        return lane_results

    def update_score_async(self, frame: np.ndarray):
        """Function to execute score detection in a separate thread using full frame."""
        score = self.score_detector.update_score(frame)
        if score > self.last_known_score:
            self.last_known_score = score

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame and return image with visualization"""
        detections = {
            'yellow': 0,
            'green': 0,
            'lanes': {}
        }

        if self.view_mode == 0:
            output_frame = frame.copy()
        else:
            hsv_visual_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if self.view_mode == 1:
                mask = self.detect_yellow_notes(hsv_visual_frame)
            else:
                mask = self.detect_green_notes(hsv_visual_frame)
            output_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        green_mask = self.detect_green_notes(hsv_frame)
        yellow_mask = self.detect_yellow_notes(hsv_frame)

        green_contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow_contours, _ = cv2.findContours(
            yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lane_tasks = []
        for lane_name, points in self.polygons.items():
            if points:
                lane_tasks.append(
                    (lane_name, points, green_contours, yellow_contours))

        total_green_detections = 0
        total_yellow_detections = 0

        if lane_tasks:
            with ThreadPoolExecutor(max_workers=min(6, len(lane_tasks))) as executor:
                self.frame_height, self.frame_width = frame.shape[:2]

                future_results = [
                    executor.submit(
                        self.process_lane_optimized,
                        task) for task in lane_tasks]

                for future in future_results:
                    result = future.result()
                    lane_name = result['lane_name']

                    detections['lanes'][lane_name] = {
                        'yellow': result['yellow_count'],
                        'green': result['green_count']
                    }

                    total_green_detections += result['green_count']
                    total_yellow_detections += result['yellow_count']

                    if self.view_mode == 0:
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
                            cv2.putText(
                                output_frame,
                                f"G:{int(box['area'])}",
                                (box['x'],
                                 box['y'] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,
                                 255,
                                 0),
                                1)

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
                            cv2.putText(
                                output_frame,
                                f"Y:{int(box['area'])}",
                                (box['x'],
                                 box['y'] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,
                                 255,
                                 255),
                                1)

        for lane_name, points in self.polygons.items():
            pts = np.array(points, np.int32)
            color = self.lane_colors.get(lane_name, (255, 255, 255))
            cv2.polylines(
                output_frame,
                [pts],
                isClosed=True,
                color=color,
                thickness=2)

            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(output_frame, lane_name, tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        detections['yellow'] = total_yellow_detections
        detections['green'] = total_green_detections
        self.total_green_count = total_green_detections
        self.total_yellow_count = total_yellow_detections

        current_time = time.time()
        if self.score_region and (
                current_time -
                self.last_score_check_time > self.score_update_interval):
            if self.score_thread is None or not self.score_thread.is_alive():
                self.last_score_check_time = current_time

                x = self.score_region['x']
                y = self.score_region['y']
                w = self.score_region['width']
                h = self.score_region['height']
                score_roi = frame[y:y + h, x:x + w]

                # Pass full frame for internal ROI crop in ScoreDetector
                self.score_thread = threading.Thread(
                    target=self.update_score_async, args=(frame,))
                self.score_thread.start()

        score_roi_config = self.config_manager.get_score_region()
        if score_roi_config:
            x = score_roi_config.get('x', 0)
            y = score_roi_config.get('y', 0)
            w = score_roi_config.get('width', 0)
            h = score_roi_config.get('height', 0)

            cv2.rectangle(output_frame, (x, y),
                          (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(output_frame, "Score ROI", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            self.draw_score(output_frame)
        else:
            cv2.putText(output_frame, "SCORE ROI NO CONFIGURADA", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        self.add_top_counter(output_frame)

        self.fps_counter += 1
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time >= 1.0:
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = time.time()

        fps_text = f"FPS: {self.current_fps:.1f}"
        text_size = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

        fps_x = output_frame.shape[1] - text_size[0] - 10
        fps_y = output_frame.shape[0] - 10

        overlay = output_frame.copy()
        cv2.rectangle(overlay, (fps_x - 5, fps_y - text_size[1] - 5),
                      (fps_x + text_size[0] + 5, fps_y + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)

        cv2.putText(output_frame, fps_text, (fps_x, fps_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return output_frame, detections

    def add_top_counter(self, frame: np.ndarray):
        """Add top counter with real-time detections"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        counter_text = f"GREENS: {self.total_green_count}  |  YELLOWS: {self.total_yellow_count}"

        text_size = cv2.getTextSize(
            counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        x = (frame.shape[1] - text_size[0]) // 2
        y = 45

        cv2.putText(frame, counter_text, (x + 2, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
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
            3)

        if hasattr(self, 'morphology_params'):
            green_close = self.morphology_params['close_size']
            green_dilate = self.morphology_params['dilate_size']
            config_text = f"HSV Plus | V_min: {self.green_hsv['lower'][2]} | Close: {green_close} | Dilate: {green_dilate}"
        else:
            config_text = "Standard configuration"

        cv2.putText(frame, config_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    def draw_score(self, frame: np.ndarray):
        """Draws the last known score on the frame."""
        score_text = f"Score: {self.last_known_score}"
        cv2.putText(
            frame,
            score_text,
            (frame.shape[1] -
             250,
             30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,
             255,
             255),
            2,
            cv2.LINE_AA)

    def draw_view_mode(self, frame: np.ndarray):
        """Draws the current view mode."""
        mode_text = f"View: {self.view_modes[self.view_mode]}"
        cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_fps(self, frame: np.ndarray):
        """Draws current FPS on the frame."""
        fps_text = f"FPS: {self.current_fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def run(self):
        """Run visualizer"""
        print("\n" + "=" * 60)
        print("🎯 POLYGON VISUALIZER PLUS")
        print("=" * 60)

        print("HSV and Morphology configuration loaded from config.ini:")
        print(f"  Green V_min: {self.green_hsv['lower'][2]}")
        print(
            f"  GREEN Morphology: Close={self.morphology_params['close_size']}, Dilate={self.morphology_params['dilate_size']}")
        print(
            f"  YELLOW Morphology: Close={self.morphology_params['close_size']//2}, Dilate={self.morphology_params['dilate_size']//2}")
        print(
            f"  Detection Area: {self.morphology_params['min_area']}-{self.morphology_params['max_area']}")

        print("\nCONTROLS:")
        print("- 'q': Exit")
        print("- 's': Capture frame")
        print("- '+': Change view (Normal → Yellow → Green)")
        print("- SPACE: Pause/Resume")
        print("\nFEATURES:")
        print("- Lane detection with optimized THREADING")
        print("- Single HSV conversion + global morphology per frame")
        print("- Optimized game logic (if green, don't search for yellow)")
        print("- Parallel processing of up to 6 lanes")
        print("- Score OCR in separate thread (non-blocking)")
        print("- Real-time FPS counter")
        print("=" * 60)

        paused = False
        frame_count = 0

        try:
            while True:
                if not paused:
                    frame = self.screen_capture.get_latest_frame()

                if frame is not None:
                    output_frame, detections = self.process_frame(frame)

                    self.add_top_counter(output_frame)
                    self.draw_fps(output_frame)
                    self.draw_view_mode(output_frame)
                    self.draw_score(output_frame)

                    cv2.imshow('Guitar Hero IA - Visualizador', output_frame)
                else:
                    print("Frame not captured, waiting...")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    cv2.imwrite(f"polygon_viz_{timestamp}.png", output_frame)
                    print(f"Frame saved: polygon_viz_{timestamp}.png")
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                # Both + and = (for US keyboards)
                elif key == ord('+') or key == ord('='):
                    self.view_mode = (self.view_mode + 1) % 3
                    print(
                        f"📺 View changed to: {self.view_modes[self.view_mode]}")

        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            cv2.destroyAllWindows()
            if self.score_thread and self.score_thread.is_alive():
                self.score_thread.join()
            # Ensure capture thread stops
            self.screen_capture.stop()


def main():
    """Main function"""
    try:
        visualizer = PolygonVisualizer()
        visualizer.run()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
