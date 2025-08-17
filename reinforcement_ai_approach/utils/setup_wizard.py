#!/usr/bin/env python3
"""
Interactive setup wizard for RL approach:
- Select capture region (draw rectangle)
- Define lane polygons (S,D,F,J,K,L) by clicking points and pressing ENTER to confirm each lane
- Define SCORE and COMBO ROIs (draw rectangles)

Saves values into config.ini so they persist between runs.
"""

import cv2
import numpy as np
from mss import mss
from pathlib import Path
from typing import List, Tuple, Optional

from src.utils.config_manager import ConfigManager


def draw_rect(window: str, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    drawing = {"active": False, "ix": 0, "iy": 0, "rect": None}

    def _cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["active"] = True
            drawing["ix"], drawing["iy"] = x, y
            drawing["rect"] = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
            img = param.copy()
            cv2.rectangle(img, (drawing["ix"], drawing["iy"]), (x, y), (0, 255, 0), 2)
            cv2.imshow(window, img)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing["active"] = False
            x0, y0 = drawing["ix"], drawing["iy"]
            rect = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            drawing["rect"] = rect

    cv2.setMouseCallback(window, _cb, frame)
    preview = frame.copy()
    while True:
        cv2.imshow(window, preview)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return None
        if key == ord('s') and drawing["rect"]:
            return drawing["rect"]


def draw_polygon(window: str, frame: np.ndarray, label: str) -> Optional[List[Tuple[int, int]]]:
    points: List[Tuple[int, int]] = []

    def _cb(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.setMouseCallback(window, _cb, frame)
    while True:
        img = frame.copy()
        if len(points) > 0:
            for p in points:
                cv2.circle(img, p, 3, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.polylines(img, [np.array(points, np.int32)], False, (0, 255, 0), 1)
        cv2.putText(img, f"Lane {label}: click points, ENTER to confirm, 'r' reset, 'q' cancel",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window, img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return None
        if key == ord('r'):
            points.clear()
        if key == 13 and len(points) >= 3:  # ENTER
            return points


def main():
    cfg_path = Path(__file__).parent.parent / 'config' / 'config.ini'
    config = ConfigManager(config_path=str(cfg_path))

    with mss() as sct:
        # Start from full capture region from config or primary monitor as fallback
        cap_region = config.get_capture_area_config() or sct.monitors[1]
        region = {
            'left': int(cap_region['left']),
            'top': int(cap_region['top']),
            'width': int(cap_region['width']),
            'height': int(cap_region['height'])
        }

        window = 'Setup Wizard'
        cv2.namedWindow(window)

        # Step 1: capture rectangle
        while True:
            frame = np.array(sct.grab(region))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            cv2.putText(frame, "Draw CAPTURE region: drag and press 's' to save, 'q' to skip",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            rect = draw_rect(window, frame)
            if rect:
                x, y, w, h = rect
                config.config.set('CAPTURE', 'game_left', str(x))
                config.config.set('CAPTURE', 'game_top', str(y))
                config.config.set('CAPTURE', 'game_width', str(w))
                config.config.set('CAPTURE', 'game_height', str(h))
                # Mark as confirmed resolution
                config.config.set('CAPTURE', 'last_confirmed_width', str(w))
                config.config.set('CAPTURE', 'last_confirmed_height', str(h))
                region.update({'left': x, 'top': y, 'width': w, 'height': h})
            break

        # Refresh frame within new region
        frame = np.array(sct.grab(region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Step 2: lane polygons
        lane_order = ['S', 'D', 'F', 'J', 'K', 'L']
        for lane in lane_order:
            poly = draw_polygon(window, frame, lane)
            if poly is None:
                continue
            section = f'LANE_POLYGON_{lane}'
            if not config.config.has_section(section):
                config.config.add_section(section)
            config.config.set(section, 'point_count', str(len(poly)))
            for i, (px, py) in enumerate(poly):
                config.config.set(section, f'point_{i}_x', str(px))
                config.config.set(section, f'point_{i}_y', str(py))

        # Step 3: SCORE ROI
        cv2.putText(frame, "Draw SCORE ROI: drag and press 's' to save, 'q' to skip",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        score_rect = draw_rect(window, frame)
        if score_rect:
            x, y, w, h = score_rect
            if not config.config.has_section('SCORE'):
                config.config.add_section('SCORE')
            config.config.set('SCORE', 'score_region_relative', str({'left': x, 'top': y, 'width': w, 'height': h}))

        # Step 4: COMBO ROI
        cv2.putText(frame, "Draw COMBO ROI: drag and press 's' to save, 'q' to skip",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        combo_rect = draw_rect(window, frame)
        if combo_rect:
            x, y, w, h = combo_rect
            if not config.config.has_section('COMBO'):
                config.config.add_section('COMBO')
            config.config.set('COMBO', 'combo_region', str({'left': x, 'top': y, 'width': w, 'height': h}))

        # Step 5: SONG TIME ROI (top-right timer)
        cv2.putText(frame, "Draw SONG TIME ROI: drag and press 's' to save, 'q' to skip",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        time_rect = draw_rect(window, frame)
        if time_rect:
            x, y, w, h = time_rect
            if not config.config.has_section('SONG'):
                config.config.add_section('SONG')
            config.config.set('SONG', 'song_time_region', str({'left': x, 'top': y, 'width': w, 'height': h}))

        config.save_config()
        cv2.destroyAllWindows()
        print('✅ Configuration saved to config.ini')


if __name__ == '__main__':
    main()


