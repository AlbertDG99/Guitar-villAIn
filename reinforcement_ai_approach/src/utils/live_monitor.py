import cv2
import numpy as np
from typing import List, Optional


class LiveMonitor:
    """Lightweight OpenCV UI to visualize model inputs and actions."""

    def __init__(self, window_name: str = 'RL Monitor', key_names: Optional[List[str]] = None):
        self.window_name = window_name
        self.key_names = key_names or ['s', 'd', 'f', 'j', 'k', 'l']
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def update(self, state, action_index: int, combo: int, score: int, reward: float, elapsed: float, epsilon: float = None, loss: float = None):
        # State: 12 dims [g1..g6, y1..y6]
        greens = state[:6].astype(int)
        yellows = state[6:12].astype(int)

        # Convert action index to 6-bit vector (pressed per lane)
        action_bits = [(action_index >> b) & 1 for b in range(6)]

        width = 720
        height = 320
        img = np.full((height, width, 3), (18, 18, 18), dtype=np.uint8)

        # Header
        header = f"Combo: {combo}  Score: {score}  Reward: {reward:.2f}  Time: {elapsed:4.1f}s"
        if epsilon is not None:
            header += f"  Eps: {epsilon:.3f}"
        if loss is not None:
            header += f"  Loss: {loss:.3f}"
        cv2.putText(img, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Per-lane bars
        margin_left = 20
        top = 60
        lane_w = (width - margin_left * 2) // 6
        lane_h = 220

        for i in range(6):
            x0 = margin_left + i * lane_w
            y0 = top
            # Background
            cv2.rectangle(img, (x0 + 4, y0), (x0 + lane_w - 6, y0 + lane_h), (45, 45, 45), -1)

            # Green indicator (hold)
            if greens[i] == 1:
                cv2.rectangle(img, (x0 + 8, y0 + 10), (x0 + lane_w - 10, y0 + 100), (0, 255, 0), -1)

            # Yellow indicator (tap)
            if yellows[i] == 1:
                cv2.rectangle(img, (x0 + 8, y0 + 120), (x0 + lane_w - 10, y0 + 210), (0, 255, 255), -1)

            # Action pressed overlay
            if action_bits[i] == 1:
                overlay = img.copy()
                cv2.rectangle(overlay, (x0 + 4, y0), (x0 + lane_w - 6, y0 + lane_h), (120, 180, 255), -1)
                cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

            # Lane label
            lane_label = self.key_names[i].upper() if i < len(self.key_names) else str(i)
            cv2.putText(img, lane_label, (x0 + lane_w // 2 - 8, y0 + lane_h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.window_name)


