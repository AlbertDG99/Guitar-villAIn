import time
import cv2
import numpy as np

from src.utils.config_manager import ConfigManager
from src.core.screen_capture import ScreenCapture


def compute_lanes_roi(polygons: dict, capture_cfg: dict) -> tuple[int, int, int, int]:
    xs, ys = [], []
    for pts in polygons.values():
        for (x, y) in pts:
            xs.append(x); ys.append(y)
    if not xs or not ys:
        return 0, 0, int(capture_cfg['width']), int(capture_cfg['height'])
    min_x, min_y = max(0, min(xs)), max(0, min(ys))
    max_x = min(int(capture_cfg['width']) - 1, max(xs))
    max_y = min(int(capture_cfg['height']) - 1, max(ys))
    pad = 4
    x0 = max(0, min_x - pad); y0 = max(0, min_y - pad)
    x1 = min(int(capture_cfg['width']), max_x + pad)
    y1 = min(int(capture_cfg['height']), max_y + pad)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def precompute_lane_masks_roi(polygons: dict, lanes_roi: tuple[int, int, int, int]) -> dict:
    x0, y0, w, h = lanes_roi
    masks = {}
    for lane_key, pts in polygons.items():
        if not pts:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = np.array([(x - x0, y - y0) for (x, y) in pts], dtype=np.int32)
        cv2.fillPoly(mask, [shifted], (255,))
        masks[lane_key] = mask
    return masks


def detect_mask(hsv_img, color: str, hsv_ranges: dict, morph: dict):
    lower = np.array([hsv_ranges[color]['h_min'],
                      hsv_ranges[color]['s_min'],
                      hsv_ranges[color]['v_min']])
    upper = np.array([hsv_ranges[color]['h_max'],
                      hsv_ranges[color]['s_max'],
                      hsv_ranges[color]['v_max']])
    mask = cv2.inRange(hsv_img, lower, upper)
    close_size = morph['close_size']
    dilate_size = morph['dilate_size']
    if color == 'yellow':
        close_size = max(3, close_size // 2)
        dilate_size = max(2, dilate_size // 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    mask = cv2.dilate(mask, np.ones((dilate_size, dilate_size), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def main():
    cfg = ConfigManager(config_path='config/config.ini')
    capture_cfg = cfg.get_capture_area_config()
    hsv_ranges = cfg.get_hsv_ranges()
    morph = cfg.get_morphology_params()
    polygons = cfg.get_note_lane_polygons_relative()

    # ROI y máscaras por carril (relativas al ROI)
    lanes_roi = compute_lanes_roi(polygons, capture_cfg)
    lane_masks_roi = precompute_lane_masks_roi(polygons, lanes_roi)

    cap = ScreenCapture(capture_cfg)
    cap.start()
    time.sleep(0.5)

    fps_t0 = time.time()
    fps_count = 0
    fps_val = 0.0

    try:
        while True:
            frame = cap.get_latest_frame()
            if frame is None or frame.size == 0:
                cv2.waitKey(1)
                continue

            vis = frame.copy()

            # Recorta a ROI de carriles para acelerar
            x0, y0, w, h = lanes_roi
            roi = frame[y0:y0 + h, x0:x0 + w]
            if roi.size == 0:
                cv2.imshow('Input Preview', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            gmask_roi = detect_mask(hsv_roi, 'green', hsv_ranges, morph)
            ymask_roi = detect_mask(hsv_roi, 'yellow', hsv_ranges, morph)

            # Por carril: AND binario y cuenta de pixeles
            for lane_name, mask_roi in lane_masks_roi.items():
                g_present = cv2.countNonZero(cv2.bitwise_and(gmask_roi, mask_roi)) > 0
                y_present = cv2.countNonZero(cv2.bitwise_and(ymask_roi, mask_roi)) > 0

                pts = np.array(polygons[lane_name], np.int32)
                color = (0, 255, 0) if g_present else (0, 255, 255) if y_present else (120, 120, 120)
                cv2.polylines(vis, [pts], True, color, 2)

                if g_present or y_present:
                    overlay = vis.copy()
                    fill_color = (0, 120, 0) if g_present else (0, 120, 120)
                    cv2.fillPoly(overlay, [pts], fill_color)
                    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

                center = np.mean(pts, axis=0).astype(int)
                cv2.putText(vis, f"{lane_name} G:{int(g_present)} Y:{int(y_present)}",
                            tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # FPS
            fps_count += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps_val = fps_count / (now - fps_t0)
                fps_t0 = now
                fps_count = 0
            cv2.putText(vis, f"FPS: {fps_val:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Input Preview', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()