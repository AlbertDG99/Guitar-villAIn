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

    def __init__(self, config_path='config/config.ini', render_mode=None):
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
        # Observation: 6 greens (0/1) + 6 yellows (0/1) + normalized internal combo
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_keys * 2 + 1,), dtype=np.float32
        )

        self.screen_capturer = ScreenCapture(capture_config)
        self.combo_detector = ComboDetector(
            self.config_manager.get_combo_region())
        self.score_detector = ScoreDetector(
            self.config_manager.get_score_region())
        self.song_time_region = self.config_manager.get_song_time_region()
        self.note_polygons = self.config_manager.get_note_lane_polygons_relative()
        self.hsv_ranges = self.config_manager.get_hsv_ranges()

        # Key press state (0/1 per key)
        self.key_states = np.zeros(self.num_keys, dtype=int)
        self.prev_combo = 0
        self.prev_score = 0
        # Internal combo and per-lane yellow hit cooldown frames to avoid duplicate counts
        self.internal_combo = 0
        self.yellow_hit_cooldown_frames = np.zeros(self.num_keys, dtype=int)
        self.green_hold_required = np.zeros(self.num_keys, dtype=int)  # 1 while green should be held
        self.frame_index = 0
        self.frame = None
        self.episode_start_time = 0
        self.max_episode_duration = ai_config.get(
            'max_episode_duration_secs', 90)

        self.executor = ThreadPoolExecutor(max_workers=3)
        self.screen_capturer.start()
        self._last_frame_start = None

    def step(self, action):
        import time as _t
        self._last_frame_start = _t.perf_counter()
        action_vector = self._action_index_to_vector(action)
        self._execute_action_vector(action_vector)
        time.sleep(0.01)

        note_states, current_combo, current_score = self._get_all_detections()
        # End-of-song detection via OCR on configured time region (optional)
        if self.song_time_region is not None:
            if self._is_song_finished(self.frame, self.song_time_region):
                truncated = True
                info = {'score': current_score, 'combo': current_combo, 'song_end': True}
                observation = np.array(list(note_states[0]) + list(note_states[1]) + [min(self.internal_combo, 50)/50.0], dtype=np.float32)
                frame_latency_ms = ( _t.perf_counter() - self._last_frame_start) * 1000.0
                performance_logger.log_fps(self._compute_fps())
                performance_logger.log_metric("latency_ms", frame_latency_ms, "ms")
                performance_logger.maybe_console_report(1.0)
                self._log_status(observation)
                # TODO: Call self._restart_song_sequence() here when the sequence is defined
                return observation, 0.0, False, truncated, info

        if note_states is None or current_combo is None or current_score is None:
            self.logger.warning("Detection failure, returning neutral state.")
            assert self.observation_space.shape is not None
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            return observation, 0.0, False, False, {}

        # Build observation: 6 greens + 6 yellows + internal combo (normalized)
        greens, yellows = note_states
        combo_norm = min(self.internal_combo, 50) / 50.0
        observation = np.array(list(greens) + list(yellows) + [combo_norm], dtype=np.float32)

        reward = self._calculate_reward_shaped(greens, yellows, action_vector)

        terminated = self.prev_combo > 2 and current_combo == 1
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
        self.episode_start_time = time.time()

        self._hard_reset_song()

        time.sleep(0.5)
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
        self.internal_combo = 0
        self.yellow_hit_cooldown_frames[:] = 0
        self.green_hold_required[:] = 0
        self.frame_index = 0

        initial_obs = np.array(list(note_states[0]) + list(note_states[1]) + [0.0], dtype=np.float32)
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

            note_states_str = ' '.join(map(str, state[:-1].astype(int)))
            combo = int(state[-1])

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
        if self.screen_capturer:
            self.screen_capturer.stop()
        cv2.destroyAllWindows()
        self.logger.info("Environment resources released.")

    def _is_song_finished(self, frame: np.ndarray, roi: dict) -> bool:
        """Use OCR on song time ROI to detect end of song. If it returns 00:00 or disappears for a while, consider finished."""
        try:
            import pytesseract
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
            img = frame[y:y+h, x:x+w]
            if img.size == 0:
                return False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            text = pytesseract.image_to_string(bw, config='--psm 7 -c tessedit_char_whitelist=0123456789:').strip()
            if not text:
                return False
            digits = ''.join(ch for ch in text if ch.isdigit() or ch == ':')
            if digits.count(':') >= 1 and (digits.endswith('0') or digits.endswith('00')):
                # rough check for 00:00 or 0:0
                parts = [p for p in digits.split(':') if p.isdigit()]
                if len(parts) >= 2 and int(parts[-1]) == 0 and int(parts[-2]) == 0:
                    return True
            return False
        except Exception:
            return False

    def _restart_song_sequence(self):
        """TODO: Implement automatic restart of the song via keyboard/mouse sequence.
        This should simulate the exact key/click flow required by the game UI.
        """
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
        future_combo = self.executor.submit(
            self.combo_detector.detect, self.frame)
        future_score = self.executor.submit(
            self.score_detector.update_score, self.frame)

        return future_notes.result(), future_combo.result(), future_score.result()

    def _detect_notes_by_color(self) -> tuple[list[int], list[int]]:
        greens = [0] * self.num_keys
        yellows = [0] * self.num_keys
        if not self.hsv_ranges or not self.note_polygons or self.frame is None:
            return greens, yellows

        hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        green_mask = self._get_color_mask(hsv_frame, 'green')
        yellow_mask = self._get_color_mask(hsv_frame, 'yellow')

        for i, key in enumerate(self.key_bindings):
            lane_key = key.upper()
            if lane_key in self.note_polygons:
                poly = np.array(self.note_polygons[lane_key], dtype=np.int32)
                lane_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(lane_mask, [poly], (255,))

                masked_green = cv2.bitwise_and(green_mask, green_mask, mask=lane_mask)
                masked_yellow = cv2.bitwise_and(yellow_mask, yellow_mask, mask=lane_mask)

                greens[i] = 1 if np.any(masked_green) else 0
                yellows[i] = 1 if np.any(masked_yellow) else 0
        return greens, yellows

    def _get_color_mask(self, hsv_frame: np.ndarray, color: str) -> np.ndarray:
        """Applies HSV filter for a color."""
        ranges = self.hsv_ranges.get(color)
        if not ranges:
            return np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

        lower = np.array([ranges['h_min'], ranges['s_min'], ranges['v_min']])
        upper = np.array([ranges['h_max'], ranges['s_max'], ranges['v_max']])
        mask = cv2.inRange(hsv_frame, lower, upper)
        return mask

    def _calculate_reward_shaped(self, greens: list[int], yellows: list[int], action_vector: np.ndarray) -> float:
        """Shaped reward based on detection and action alignment.
        - Reward pressing when yellow present (one-time per note, with cooldown)
        - Reward holding when green present (penalize not holding)
        - Strong penalty when expected press/hold is missed (combo break)
        - Penalize spamming keys when no note is present
        """
        hit_reward = 1.0
        hold_reward = 0.1
        miss_penalty = -2.0
        spam_penalty = -0.05
        strong_combo_break_penalty = -5.0

        reward = 0.0
        combo_broken = False

        # Update per-lane logic
        for i in range(self.num_keys):
            is_green = greens[i] == 1
            is_yellow = yellows[i] == 1
            is_pressed = action_vector[i] == 1

            # Green logic: must hold while present
            if is_green:
                self.green_hold_required[i] = 1
                if is_pressed:
                    reward += hold_reward
                else:
                    reward += miss_penalty
                    combo_broken = True
            else:
                # If green no longer present, holding not required
                self.green_hold_required[i] = 0

            # Yellow logic: one hit per note with simple cooldown window
            if is_yellow:
                if self.yellow_hit_cooldown_frames[i] == 0 and is_pressed:
                    reward += hit_reward
                    self.internal_combo += 1
                    # Start cooldown to avoid counting same note in consecutive frames
                    self.yellow_hit_cooldown_frames[i] = 3
            else:
                # When yellow not visible, let cooldown decay
                if self.yellow_hit_cooldown_frames[i] > 0:
                    self.yellow_hit_cooldown_frames[i] -= 1

            # Spam penalty: pressing when neither green nor yellow present
            if not is_green and not is_yellow and is_pressed:
                reward += spam_penalty

        # Strong penalty on combo break
        if combo_broken:
            reward += strong_combo_break_penalty
            self.internal_combo = 0

        self.frame_index += 1
        return reward

    def _hard_reset_song(self):
        """Executes the song reset sequence in the game."""
        self.logger.info("Executing song reset...")
        pydirectinput.press('esc')
        time.sleep(1.0)

        capture_area = self.config_manager.get_capture_area_config()
        if capture_area:
            center_x = capture_area['left'] + capture_area['width'] // 2
            center_y = capture_area['top'] + capture_area['height'] // 2
            pydirectinput.moveTo(center_x, center_y)
            time.sleep(0.1)
            pydirectinput.click()

        time.sleep(2.0)
        self.logger.info("Song reset completed.")
