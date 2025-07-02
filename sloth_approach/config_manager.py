"""
Config Manager - Gestor de Configuración
=========================================

Maneja todas las configuraciones del sistema Guitar Hero IA.
"""

from typing import Dict, Any, Tuple, List, Optional
import configparser
from pathlib import Path
import logging
import json


class ConfigManager:
    """Gestor de configuración del sistema. Falla rápido si hay errores."""

    def __init__(self, config_path='config/config.ini'):
        """
        Inicializa el ConfigManager y carga la configuración.

        Args:
            config_path (str): Ruta al fichero de configuración.
        """
        self.config_file = Path(config_path)
        self.config = configparser.ConfigParser(
            interpolation=None,
            inline_comment_prefixes=(';', '#')
        )
        self.logger = logging.getLogger(__name__)
        self._load_config()

    def _load_config(self):
        """Cargar configuración desde archivo. Falla si el archivo no existe."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"El archivo de configuración '{self.config_file}' no se encontró.")
        self.config.read(self.config_file)

    def save_config(self):
        """Guardar configuración a archivo"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            self.config.write(f)

    def get(self, section: str, key: str) -> str:
        """Obtener valor de configuración. Falla si no se encuentra."""
        return self.config.get(section, key)

    def getint(self, section: str, key: str) -> int:
        """Obtener valor entero de configuración. Falla si no se encuentra."""
        return self.config.getint(section, key)

    def getfloat(self, section: str, key: str) -> float:
        """Obtener valor flotante de configuración. Falla si no se encuentra."""
        return self.config.getfloat(section, key)

    def getboolean(self, section: str, key: str) -> bool:
        """Obtener valor booleano de configuración. Falla si no se encuentra."""
        return self.config.getboolean(section, key)

    def get_hsv_ranges(self) -> Dict[str, Dict[str, int]]:
        """Obtiene los rangos HSV para cada color. Falla si la seccion no existe."""
        ranges = {}
        if not self.config.has_section('HSV_RANGES'):
            raise configparser.NoSectionError('HSV_RANGES')

        # Asumimos que los colores a buscar son 'green' y 'yellow' por ahora
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
                raise ValueError(f"Falta una clave de color para '{color}' en [HSV_RANGES]") from e
        return ranges

    def get_morphology_params(self) -> Dict[str, int]:
        """Carga los parámetros de morfología. Falla si la seccion no existe."""
        if not self.config.has_section('MORPHOLOGY'):
            raise configparser.NoSectionError('MORPHOLOGY')
        
        return {
            'close_size': self.getint('MORPHOLOGY', 'close_size'),
            'dilate_size': self.getint('MORPHOLOGY', 'dilate_size'),
            'min_area': self.getint('MORPHOLOGY', 'min_area'),
            'max_area': self.getint('MORPHOLOGY', 'max_area')
        }

    def get_note_lane_polygons(self) -> Dict[str, List[Tuple[int, int]]]:
        """Obtiene los polígonos de las líneas de notas."""
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
            print("Advertencia: No se encontraron poligonos de carril en config.ini. La deteccion de notas no funcionara.")

        return polygons

    def get_note_lane_polygons_relative(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Obtiene los poligonos de los carriles.
        NOTA: Los poligonos en formato LANE_POLYGON_* ya son relativos,
        por lo que este metodo simplemente los devuelve directamente.
        """
        return self.get_note_lane_polygons()

    def has_note_lane_polygons(self) -> bool:
        """Comprueba si hay polígonos de carril definidos en la configuración."""
        polygons = self.get_note_lane_polygons()
        return len(polygons) > 0

    def get_combo_region(self) -> Optional[Dict[str, int]]:
        """Obtiene la región del combo (relativa a la ventana de juego)."""
        if not self.config.has_section('COMBO'):
            self.logger.warning("No se encontró la sección [COMBO] en config.ini. La detección de combo no funcionará.")
            return None
        try:
            region_str = self.get('COMBO', 'combo_region')
            region_dict = eval(region_str) 
            return {
                'x': int(region_dict['left']),
                'y': int(region_dict['top']),
                'width': int(region_dict['width']),
                'height': int(region_dict['height'])
            }
        except (configparser.NoOptionError, SyntaxError, NameError, KeyError) as e:
            self.logger.error(f"Error al leer 'combo_region' de config.ini: {e}")
            return None

    def get_capture_area_config(self):
        """
        Devuelve la configuración del área de captura, priorizando la región calibrada.
        """
        try:
            # Prioridad 1: Usar la región de calibración si existe
            if 'calibration' in self.config and 'region' in self.config['calibration']:
                region_str = self.config['calibration']['region']
                # El string es un diccionario, hay que evaluarlo de forma segura
                region_data = json.loads(region_str.replace("'", "\""))
                
                # Extraer las coordenadas del diccionario anidado 'real'
                if 'raw_coords' in region_data and 'real' in region_data['raw_coords']:
                    real_coords = region_data['raw_coords']['real']
                    return {
                        'left': real_coords[0],
                        'top': real_coords[1],
                        'width': real_coords[2] - real_coords[0],
                        'height': real_coords[3] - real_coords[1]
                    }

            # Prioridad 2: Usar los valores individuales de la sección [CAPTURE]
            return {
                'left': self.config.getint('CAPTURE', 'game_left'),
                'top': self.config.getint('CAPTURE', 'game_top'),
                'width': self.config.getint('CAPTURE', 'game_width'),
                'height': self.config.getint('CAPTURE', 'game_height'),
            }
        except (configparser.NoSectionError, configparser.NoOptionError, KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Error al obtener la configuración del área de captura: {e}")
            return None

    def get_ai_config(self) -> Dict[str, Any]:
        """Carga la configuración para el agente de IA. Falla si la sección no existe."""
        if not self.config.has_section('AI'):
            raise configparser.NoSectionError('AI')

        # Usamos un diccionario para poder devolver diferentes tipos de datos
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
        """Devuelve la lista de teclas de acción desde la sección [INPUT]."""
        if 'INPUT' in self.config and 'key_bindings' in self.config['INPUT']:
            keys_str = self.config['INPUT']['key_bindings']
            return [key.strip() for key in keys_str.split(',')]
        # Devuelve un valor por defecto o lanza un error si no se encuentra
        print("Advertencia: No se encontró 'key_bindings' en la sección [INPUT] de config.ini. Usando valor por defecto.")
        return ['s', 'd', 'f', 'j', 'k', 'l']

    def get_config(self, section, option, fallback=None):
        return self.config.get(section, option, fallback=fallback) 