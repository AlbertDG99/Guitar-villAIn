"""Configuration manager for the Guitar Hero AI system."""

from typing import Dict, Any, Tuple, List, Optional
import configparser
from pathlib import Path
import logging
import json
import ast


class ConfigManager:
    """System configuration manager. Fails fast if there are errors."""

    def __init__(self, config_path=None):
        """
        Initializes the ConfigManager and loads the configuration.

        Args:
            config_path (str): Path to the configuration file.
        """
        if config_path is None:
            # Default path relative to the reinforcement_ai_approach directory
            config_path = Path(__file__).parents[2] / 'config' / 'config.ini'

        self.config_file = Path(config_path)
        self.config = configparser.ConfigParser(
            interpolation=None,
            inline_comment_prefixes=(';', '#')
        )
        self.logger = logging.getLogger(__name__)
        self._load_config()

    def _load_config(self):
        """Load configuration from file. Fails if the file doesn't exist."""
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"The configuration file '{self.config_file}' was not found.")
        self.config.read(self.config_file)

    def save_config(self):
        """Save configuration to file"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            self.config.write(f)

    def get(self, section: str, key: str) -> str:
        """Get configuration value. Fails if not found."""
        return self.config.get(section, key)

    def getint(self, section: str, key: str) -> int:
        """Get integer configuration value. Fails if not found."""
        return self.config.getint(section, key)

    def getfloat(self, section: str, key: str) -> float:
        """Get float configuration value. Fails if not found."""
        return self.config.getfloat(section, key)

    def getboolean(self, section: str, key: str) -> bool:
        """Get boolean configuration value. Fails if not found."""
        return self.config.getboolean(section, key)

    def get_hsv_ranges(self) -> Dict[str, Dict[str, int]]:
        """Gets HSV ranges for each color. Fails if the section doesn't exist."""
        ranges = {}
        if not self.config.has_section('HSV_RANGES'):
            raise configparser.NoSectionError('HSV_RANGES')

        color_keys = ['green', 'yellow']

        for color in color_keys:
            try:
                ranges[color] = {
                    'h_min': self.getint('HSV_RANGES', f'{color}_h_min'),
                    's_min': self.getint('HSV_RANGES', f'{color}_s_min'),
                    'v_min': self.getint('HSV_RANGES', f'{color}_v_min'),
                    'h_max': self.getint('HSV_RANGES', f'{color}_h_max'),
                    's_max': self.getint('HSV_RANGES', f'{color}_s_max'),
                    'v_max': self.getint('HSV_RANGES', f'{color}_v_max'),
                }
            except configparser.NoOptionError as e:
                raise ValueError(
                    f"Missing color key for '{color}' in [HSV_RANGES]") from e
        return ranges

    def get_morphology_params(self) -> Dict[str, int]:
        """Loads morphology parameters. Fails if the section doesn't exist."""
        if not self.config.has_section('MORPHOLOGY'):
            raise configparser.NoSectionError('MORPHOLOGY')

        return {
            'close_size': self.getint('MORPHOLOGY', 'close_size'),
            'dilate_size': self.getint('MORPHOLOGY', 'dilate_size'),
            'min_area': self.getint('MORPHOLOGY', 'min_area'),
            'max_area': self.getint('MORPHOLOGY', 'max_area')
        }

    def get_note_lane_polygons(self) -> Dict[str, List[Tuple[int, int]]]:
        """Gets the polygons of the note lanes."""
        lane_names = ['S', 'D', 'F', 'J', 'K', 'L']
        polygons = {}

        for lane_name in lane_names:
            section_name = f'LANE_POLYGON_{lane_name}'
            if self.config.has_section(section_name):
                points = []
                point_count = self.getint(section_name, 'point_count')
                for i in range(point_count):
                    x_key = f'point_{i}_x'
                    y_key = f'point_{i}_y'
                    x = self.getint(section_name, x_key)
                    y = self.getint(section_name, y_key)
                    points.append((x, y))

                if len(points) >= 3:
                    polygons[lane_name] = points

        if not polygons:
            print(
                "Warning: No lane polygons found in config.ini. Note detection will not work.")

        return polygons

    def get_note_lane_polygons_relative(
            self) -> Dict[str, List[Tuple[int, int]]]:
        """Gets the lane polygons."""
        return self.get_note_lane_polygons()

    def has_note_lane_polygons(self) -> bool:
        """Checks if there are lane polygons defined in the configuration."""
        polygons = self.get_note_lane_polygons()
        return len(polygons) > 0

    def get_score_region(self) -> Optional[Dict[str, int]]:
        """Gets the score region (relative to the game window)."""
        if not self.config.has_section('SCORE'):
            self.logger.warning(
                "Section [SCORE] not found in config.ini. Score detection will not work.")
            return None
        try:
            region_str = self.get('SCORE', 'score_region_relative')
            # Safer parsing than eval
            region_dict = ast.literal_eval(region_str)
            return {
                'x': int(region_dict['left']),
                'y': int(region_dict['top']),
                'width': int(region_dict['width']),
                'height': int(region_dict['height'])
            }
        except (configparser.NoOptionError, SyntaxError, NameError, KeyError) as e:
            self.logger.error(
                f"Error reading 'score_region_relative' from config.ini: {e}")
            return None

    def get_combo_region(self) -> Optional[Dict[str, int]]:
        """Gets the combo region (relative to the game window)."""
        if not self.config.has_section('COMBO'):
            self.logger.warning(
                "Section [COMBO] not found in config.ini. Combo detection will not work.")
            return None
        try:
            region_str = self.get('COMBO', 'combo_region')
            region_dict = ast.literal_eval(region_str)
            return {
                'x': int(region_dict['left']),
                'y': int(region_dict['top']),
                'width': int(region_dict['width']),
                'height': int(region_dict['height'])
            }
        except (configparser.NoOptionError, SyntaxError, NameError, KeyError) as e:
            self.logger.error(
                f"Error reading 'combo_region' from config.ini: {e}")
            return None

    def get_song_time_region(self) -> Optional[Dict[str, int]]:
        """Deprecated: song time region not used anymore."""
        return None
        

    def get_capture_area_config(self):
        try:
            if 'calibration' in self.config and 'region' in self.config['calibration']:
                region_str = self.config['calibration']['region']
                region_data = json.loads(region_str.replace("'", "\""))

                if 'raw_coords' in region_data and 'real' in region_data['raw_coords']:
                    real_coords = region_data['raw_coords']['real']
                    return {
                        'left': real_coords[0],
                        'top': real_coords[1],
                        'width': real_coords[2] - real_coords[0],
                        'height': real_coords[3] - real_coords[1]
                    }

            return {
                'left': self.config.getint('CAPTURE', 'game_left'),
                'top': self.config.getint('CAPTURE', 'game_top'),
                'width': self.config.getint('CAPTURE', 'game_width'),
                'height': self.config.getint('CAPTURE', 'game_height'),
            }
        except (configparser.NoSectionError, configparser.NoOptionError, KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Error getting capture area configuration: {e}")
            return None

    def get_ai_config(self) -> Dict[str, Any]:
        """Loads configuration for the AI agent. Fails if the section doesn't exist."""
        if not self.config.has_section('AI'):
            raise configparser.NoSectionError('AI')

        ai_config = {
            'learning_rate': self.getfloat('AI', 'learning_rate'),
            'epsilon': self.getfloat('AI', 'epsilon'),
            'epsilon_decay': self.getfloat('AI', 'epsilon_decay'),
            'epsilon_min': self.getfloat('AI', 'epsilon_min'),
            'batch_size': self.getint('AI', 'batch_size'),
            'memory_size': self.getint('AI', 'memory_size'),
            'target_update_frequency': self.getint('AI', 'target_update_frequency'),
            'double_dqn': self.getboolean('AI', 'double_dqn'),
            'dueling_dqn': self.getboolean('AI', 'dueling_dqn'),
            'prioritized_replay': self.getboolean('AI', 'prioritized_replay'),
            'use_mixed_precision': self.getboolean('AI', 'use_mixed_precision'),
            'model_save_path': self.get('AI', 'model_save_path'),
            'num_episodes': self.getint('AI', 'num_episodes'),
            'max_steps_per_episode': self.getint('AI', 'max_steps_per_episode'),
            'save_frequency': self.getint('AI', 'save_frequency')
        }
        return ai_config

    def get_key_bindings(self):
        """Returns the list of action keys from the [INPUT] section."""
        if 'INPUT' in self.config and 'key_bindings' in self.config['INPUT']:
            keys_str = self.config['INPUT']['key_bindings']
            return [key.strip() for key in keys_str.split(',')]
        print(
            "Warning: 'key_bindings' not found in [INPUT] section of config.ini. Using default value.")
        return ['s', 'd', 'f', 'j', 'k', 'l']

    def get_config(self, section, option, fallback=None):
        return self.config.get(section, option, fallback=fallback)
