import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pydirectinput
import time
import cv2
from concurrent.futures import ThreadPoolExecutor

from src.core.screen_capture import ScreenCapture
from src.utils.config_manager import ConfigManager
from src.core.score_detector import ScoreDetector
from src.core.combo_detector import ComboDetector
from src.utils.logger import setup_logger
from src.utils.logger import performance_logger


class GuitarHeroEnv(gym.Env):
    """
    Gymnasium environment for learning to play Guitar Hero.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, config_path=None, render_mode=None):
        super().__init__()
        self.logger = setup_logger('GuitarHeroEnv')

        self.config_manager = ConfigManager(config_path=config_path)
        ai_config = self.config_manager.get_ai_config()
        capture_config = self.config_manager.get_capture_area_config()
        if capture_config is None:
            raise ValueError(
                "Could not load capture configuration. Check config.ini.")

        self.key_bindings = self.config_manager.get_key_bindings()
        self.num_keys = len(self.key_bindings)
        self.scancode_map = {
            key.lower(): pydirectinput.KEYBOARD_MAPPING[key.lower()]
            for key in self.key_bindings if key.lower() in pydirectinput.KEYBOARD_MAPPING
        }
        if len(self.scancode_map) != len(self.key_bindings):
            self.logger.warning(
                "Some keys from 'key_bindings' were not found in the scancode map and will be ignored.")

        # Use Discrete(2^num_keys) to allow simultaneous key presses (combinations)
        self.action_space = spaces.Discrete(2 ** self.num_keys)
        # Observation: 6 greens (0/1) + 6 yellows (0/1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_keys * 2,), dtype=np.float32
        )

        self.screen_capturer = ScreenCapture(capture_config)
        self.combo_detector = ComboDetector(
            self.config_manager.get_combo_region())
        self.score_detector = ScoreDetector(
            self.config_manager.get_score_region())
        self.note_polygons = self.config_manager.get_note_lane_polygons_relative()
        self.hsv_ranges = self.config_manager.get_hsv_ranges()
        self.morphology_params = self.config_manager.get_morphology_params()
        # Precompute ROI and lane masks (cropped) to reduce per-frame work
        self.lanes_roi = self._compute_lanes_roi(self.note_polygons, capture_config)
        self.lane_masks_roi = self._precompute_lane_masks_roi(self.note_polygons, self.lanes_roi)
        # Throttle OCR updates to reduce latency
        self._score_update_interval = 0.3
        self._combo_update_interval = 0.3
        self._last_score_update_time = 0.0
        self._last_combo_update_time = 0.0
        self._score_cache = 0
        self._combo_cache = 1

        # Key press state (0/1 per key)
        self.key_states = np.zeros(self.num_keys, dtype=int)
        self.prev_combo = 0
        self.prev_score = 0
        self.frame_index = 0
        self.frame = None
        self.episode_start_time = 0
        # Fixed 60s episode duration per requirements
        self.max_episode_duration = 60
        self.episode_count = 0

        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lane_executor = ThreadPoolExecutor(max_workers=min(self.num_keys, 6))
        self.screen_capturer.start()
        self._last_frame_start = None

    def step(self, action):
        import time as _t
        self._last_frame_start = _t.perf_counter()
        action_vector = self._action_index_to_vector(action)
        self._execute_action_vector(action_vector)
        time.sleep(0.01)

        note_states, current_combo, current_score = self._get_all_detections()

        if note_states is None or current_combo is None or current_score is None:
            self.logger.warning("Detection failure, returning neutral state.")
            assert self.observation_space.shape is not None
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            return observation, 0.0, False, False, {}

        # Build observation: 6 greens + 6 yellows
        greens, yellows = note_states
        observation = np.array(list(greens) + list(yellows), dtype=np.float32)

        reward = self._calculate_reward(greens, yellows, action_vector, current_combo, current_score)

        terminated = False
        truncated = (
            time.time() -
            self.episode_start_time) > self.max_episode_duration

        self.prev_combo = current_combo
        self.prev_score = current_score

        info = {'score': current_score, 'combo': current_combo}

        # Performance
        frame_latency_ms = ( _t.perf_counter() - self._last_frame_start) * 1000.0
        performance_logger.log_fps(self._compute_fps())
        performance_logger.log_metric("latency_ms", frame_latency_ms, "ms")
        performance_logger.maybe_console_report(1.0)

        self._log_status(observation)

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.logger.info("Resetting environment...")
        # Episode start sequence per requirements
        if self.episode_count == 0:
            self._start_song()
        else:
            self._restart_song_sequence()

        # Small delay to ensure UI reacts, then start timer from here
        time.sleep(0.5)
        self.episode_start_time = time.time()
        note_states, combo, score = self._get_all_detections()

        if note_states is None or combo is None or score is None:
            self.logger.error(
                "No se pudo obtener un estado inicial válido. Forzando estado a cero.")
            greens = [0] * self.num_keys
            yellows = [0] * self.num_keys
            note_states = (greens, yellows)
            combo = 1
            score = 0

        self.prev_score = score
        self.prev_combo = combo
        self.key_states = np.zeros(self.num_keys, dtype=int)
        self.frame_index = 0
        self.episode_count += 1

        initial_obs = np.array(list(note_states[0]) + list(note_states[1]), dtype=np.float32)
        return initial_obs, {'score': score, 'combo': combo}

    def _log_status(self, state: np.ndarray):
        """Prints current state and FPS to console at regular intervals."""
        self.fps_counter = getattr(self, 'fps_counter', 0) + 1
        self.last_log_time = getattr(self, 'last_log_time', time.time())
        self.last_fps = getattr(self, 'last_fps', 0)

        current_time = time.time()

        if current_time - self.last_log_time > 1.0:
            self.last_fps = self.fps_counter / \
                (current_time - self.last_log_time)
            self.fps_counter = 0
            self.last_log_time = current_time

            note_states_str = ' '.join(map(str, state.astype(int)))
            combo = int(self.prev_combo)

            print(f"FPS: {self.last_fps:<5.2f} | "
                  f"State: [Notes: {note_states_str}] [Combo: {combo:<3}] | "
                  f"Score Prev: {self.prev_score:<6} | "
                  f"Combo Prev: {self.prev_combo:<3} \r", end="")

    def _compute_fps(self) -> float:
        self.fps_counter = getattr(self, 'fps_counter', 0) + 1
        self.last_log_time = getattr(self, 'last_log_time', time.time())
        current_time = time.time()
        if current_time - self.last_log_time > 1.0:
            self.last_fps = self.fps_counter / (current_time - self.last_log_time)
            self.fps_counter = 0
            self.last_log_time = current_time
        return getattr(self, 'last_fps', 0.0)

    def close(self):
        self.logger.info("Closing Guitar Hero environment.")
        print("\n")
        try:
            for i, key_state in enumerate(self.key_states):
                if key_state == 1:
                    key_name = self.key_bindings[i]
                    scancode = self.scancode_map.get(key_name.lower())
                    if scancode:
                        pydirectinput.keyUp(key_name)
            self.logger.info("Keys released successfully.")
        except Exception as e:
            self.logger.error(f"Error releasing keys: {e}")

        self.executor.shutdown(wait=True)
        try:
            self.lane_executor.shutdown(wait=True)
        except Exception:
            pass
        if self.screen_capturer:
            self.screen_capturer.stop()
        cv2.destroyAllWindows()
        self.logger.info("Environment resources released.")

    def _start_song(self):
        """Press ENTER to start the song/loop."""
        try:
            pydirectinput.press('enter')
        except Exception:
            pass

    def _restart_song_sequence(self):
        """Restart sequence: F5 then ENTER after 2s (per requirements)."""
        try:
            pydirectinput.press('f5')
            time.sleep(2.0)
            pydirectinput.press('enter')
        except Exception:
            pass

    def _execute_action_vector(self, action_vector: np.ndarray):
        """Presses/releases keys based on 6-bit action vector (0/1 per key)."""
        for i, key_action in enumerate(action_vector):
            key_name = self.key_bindings[i]
            scancode = self.scancode_map.get(key_name.lower())
            if not scancode:
                continue
            is_pressed = self.key_states[i] == 1
            if key_action == 1 and not is_pressed:
                pydirectinput.keyDown(key_name)
                self.key_states[i] = 1
            elif key_action == 0 and is_pressed:
                pydirectinput.keyUp(key_name)
                self.key_states[i] = 0

    def _action_index_to_vector(self, idx: int) -> np.ndarray:
        vec = [(idx >> b) & 1 for b in range(self.num_keys)]
        return np.array(vec, dtype=int)

    def _vector_to_action_index(self, vec: np.ndarray) -> int:
        idx = 0
        for b, v in enumerate(vec):
            if v:
                idx |= (1 << b)
        return idx

    def _get_all_detections(self):
        """Gets the latest frame and executes all detections in parallel."""
        self.frame = self.screen_capturer.get_latest_frame()
        if self.frame is None or self.frame.size == 0:
            self.logger.warning("Could not get a valid frame from capturer.")
            return None, None, None

        future_notes = self.executor.submit(self._detect_notes_by_color)

        # Throttled OCR for combo and score
        now = time.time()
        if self.combo_detector and (now - self._last_combo_update_time > self._combo_update_interval):
            try:
                self._combo_cache = int(self.combo_detector.detect(self.frame))
            except Exception:
                pass
            self._last_combo_update_time = now

        if self.score_detector and (now - self._last_score_update_time > self._score_update_interval):
            try:
                self._score_cache = int(self.score_detector.update_score(self.frame))
            except Exception:
                pass
            self._last_score_update_time = now

        return future_notes.result(), self._combo_cache, self._score_cache

    def _detect_notes_by_color(self) -> tuple[list[int], list[int]]:
        """Lane-wise color detection using HSV + morphology with per-lane threading."""
        greens = [0] * self.num_keys
        yellows = [0] * self.num_keys
        if not self.hsv_ranges or not self.note_polygons or self.frame is None:
            return greens, yellows

        # Crop to lanes ROI for faster processing
        x0, y0, w, h = self.lanes_roi
        roi = self.frame[y0:y0 + h, x0:x0 + w]
        if roi.size == 0:
            return greens, yellows

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        def _detect_mask_roi(color: str) -> np.ndarray:
            lower = np.array([
                self.hsv_ranges[color]['h_min'],
                self.hsv_ranges[color]['s_min'],
                self.hsv_ranges[color]['v_min']
            ])
            upper = np.array([
                self.hsv_ranges[color]['h_max'],
                self.hsv_ranges[color]['s_max'],
                self.hsv_ranges[color]['v_max']
            ])
            mask = cv2.inRange(hsv_roi, lower, upper)
            close_size = self.morphology_params['close_size']
            dilate_size = self.morphology_params['dilate_size']
            if color == 'yellow':
                close_size = max(3, close_size // 2)
                dilate_size = max(2, dilate_size // 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
            mask = cv2.dilate(mask, np.ones((dilate_size, dilate_size), np.uint8), iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            return mask

        green_mask_roi = _detect_mask_roi('green')
        yellow_mask_roi = _detect_mask_roi('yellow')

        def process_lane(idx_lane: int, lane_key: str):
            lane_mask = self.lane_masks_roi.get(lane_key)
            if lane_mask is None:
                return idx_lane, 0, 0
            g_present = cv2.countNonZero(cv2.bitwise_and(green_mask_roi, lane_mask)) > 0
            if g_present:
                return idx_lane, 1, 0
            y_present = cv2.countNonZero(cv2.bitwise_and(yellow_mask_roi, lane_mask)) > 0
            return idx_lane, 0, 1 if y_present else 0

        tasks = []
        for i, key in enumerate(self.key_bindings):
            lane_key = key.upper()
            if lane_key in self.lane_masks_roi:
                tasks.append(self.lane_executor.submit(process_lane, i, lane_key))

        for future in tasks:
            idx, g, y = future.result()
            greens[idx] = g
            yellows[idx] = y

        return greens, yellows

    def _get_color_mask(self, hsv_frame: np.ndarray, color: str) -> np.ndarray:
        """Deprecated: kept for compatibility."""
        ranges = self.hsv_ranges.get(color)
        if not ranges:
            return np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        lower = np.array([ranges['h_min'], ranges['s_min'], ranges['v_min']])
        upper = np.array([ranges['h_max'], ranges['s_max'], ranges['v_max']])
        return cv2.inRange(hsv_frame, lower, upper)

    def _compute_lanes_roi(self, polygons: dict, capture_cfg: dict) -> tuple[int, int, int, int]:
        xs = []
        ys = []
        for pts in polygons.values():
            for (x, y) in pts:
                xs.append(x)
                ys.append(y)
        if not xs or not ys:
            return 0, 0, int(capture_cfg['width']), int(capture_cfg['height'])
        min_x = max(0, min(xs))
        min_y = max(0, min(ys))
        max_x = min(int(capture_cfg['width']) - 1, max(xs))
        max_y = min(int(capture_cfg['height']) - 1, max(ys))
        pad = 4
        x0 = max(0, min_x - pad)
        y0 = max(0, min_y - pad)
        x1 = min(int(capture_cfg['width']), max_x + pad)
        y1 = min(int(capture_cfg['height']), max_y + pad)
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def _precompute_lane_masks_roi(self, polygons: dict, lanes_roi: tuple[int, int, int, int]) -> dict:
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

    def _calculate_reward(self, greens: list[int], yellows: list[int], action_vector: np.ndarray, current_combo: int, current_score: int) -> float:
        """Reward prioritizing maintaining large combos, then score gains, with mild action shaping."""
        reward = 0.0

        # Prioritize combo maintenance
        if self.prev_combo >= 3 and current_combo == 1:
            reward -= 5.0  # strong penalty for breaking a built combo
        elif current_combo > self.prev_combo:
            # reward increases as combo grows
            reward += 0.2 * min(current_combo - self.prev_combo, 5)
        else:
            # small living bonus proportional to sustained combo
            reward += 0.02 * min(current_combo, 50) / 50.0

        # Score shaping (small weight to avoid overshadowing combo)
        if current_score is not None and self.prev_score is not None:
            delta_score = max(0, current_score - self.prev_score)
            reward += 0.001 * delta_score

        # Action shaping: encourage pressing when yellow note is present and holding on green
        for i in range(self.num_keys):
            is_green = greens[i] == 1
            is_yellow = yellows[i] == 1
            is_pressed = action_vector[i] == 1

            if is_yellow and is_pressed:
                reward += 0.3
            if is_green and is_pressed:
                reward += 0.05
            if not is_green and not is_yellow and is_pressed:
                reward -= 0.02  # light anti-spam

        self.frame_index += 1
        return float(reward)

    def _hard_reset_song(self):
        """Deprecated: Not used. Kept for compatibility."""
        self.logger.info("Hard reset is deprecated; using F5+Enter restart sequence.")
