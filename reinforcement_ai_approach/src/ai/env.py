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

        self.action_space = spaces.MultiBinary(self.num_keys)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_keys + 1,), dtype=np.float32
        )

        self.screen_capturer = ScreenCapture(capture_config)
        self.combo_detector = ComboDetector(
            self.config_manager.get_combo_region())
        self.score_detector = ScoreDetector(
            self.config_manager.get_score_region())
        self.note_polygons = self.config_manager.get_note_lane_polygons_relative()
        self.hsv_ranges = self.config_manager.get_hsv_ranges()

        self.key_states = np.zeros(self.num_keys, dtype=int)
        self.prev_combo = 0
        self.prev_score = 0
        self.frame = None
        self.episode_start_time = 0
        self.max_episode_duration = ai_config.get(
            'max_episode_duration_secs', 90)

        self.executor = ThreadPoolExecutor(max_workers=3)
        self.screen_capturer.start()

    def step(self, action):
        self._execute_action(action)
        time.sleep(0.01)

        note_states, current_combo, current_score = self._get_all_detections()

        if note_states is None or current_combo is None or current_score is None:
            self.logger.warning("Detection failure, returning neutral state.")
            assert self.observation_space.shape is not None
            observation = np.zeros(
                self.observation_space.shape,
                dtype=np.float32)
            return observation, 0.0, False, False, {}

        reward = self._calculate_reward(current_combo, current_score)
        observation = np.array(note_states + [current_combo], dtype=np.float32)

        terminated = self.prev_combo > 2 and current_combo == 1
        truncated = (
            time.time() -
            self.episode_start_time) > self.max_episode_duration

        self.prev_combo = current_combo
        self.prev_score = current_score

        info = {'score': current_score, 'combo': current_combo}

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
            note_states = [0] * self.num_keys
            combo = 1
            score = 0

        self.prev_score = score
        self.prev_combo = combo
        self.key_states = np.zeros(self.num_keys, dtype=int)

        initial_obs = np.array(note_states + [combo], dtype=np.float32)
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

    def _execute_action(self, action_vector: np.ndarray):
        """Compares action vector with key states and presses/releases."""
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

    def _detect_notes_by_color(self) -> list[int]:
        note_states = [0] * self.num_keys
        if not self.hsv_ranges or not self.note_polygons or self.frame is None:
            return note_states

        hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        green_mask = self._get_color_mask(hsv_frame, 'green')

        for i, key in enumerate(self.key_bindings):
            lane_key = key.upper()
            if lane_key in self.note_polygons:
                poly = np.array(self.note_polygons[lane_key], dtype=np.int32)
                lane_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(lane_mask, [poly], (255,))

                masked_lane = cv2.bitwise_and(
                    green_mask, green_mask, mask=lane_mask)

                if np.any(masked_lane):
                    note_states[i] = 1
        return note_states

    def _get_color_mask(self, hsv_frame: np.ndarray, color: str) -> np.ndarray:
        """Applies HSV filter for a color."""
        ranges = self.hsv_ranges.get(color)
        if not ranges:
            return np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

        lower = np.array([ranges['h_min'], ranges['s_min'], ranges['v_min']])
        upper = np.array([ranges['h_max'], ranges['s_max'], ranges['v_max']])
        mask = cv2.inRange(hsv_frame, lower, upper)
        return mask

    def _calculate_reward(
            self,
            current_combo: int,
            current_score: int) -> float:
        combo_weight = 0.7
        score_weight = 0.3

        combo_reward = 0.0
        if current_combo > self.prev_combo:
            combo_reward = 1.0
        elif current_combo > 1 and current_combo == self.prev_combo:
            combo_reward = 0.1

        if self.prev_combo > 2 and current_combo == 1:
            combo_reward = -5.0

        score_reward = 0.0
        if current_score > self.prev_score:
            score_gain = current_score - self.prev_score
            score_reward = np.log1p(score_gain) / 10.0

        return (combo_reward * combo_weight) + (score_reward * score_weight)

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
