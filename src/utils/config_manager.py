"""
Config Manager - Gestor de Configuración
=========================================

Maneja todas las configuraciones del sistema Guitar Hero IA.
"""

from typing import Dict, Any, Tuple, List, Optional
import configparser
from pathlib import Path
import logging


class ConfigManager:
    """Gestor de configuración del sistema. Falla rápido si hay errores."""

    def __init__(self, config_file: str = "config/config.ini"):
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser(interpolation=None)
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

    def get_score_region(self) -> Optional[Dict[str, int]]:
        """Obtiene la región de la puntuación (relativa a la ventana de juego)."""
        if not self.config.has_section('SCORE'):
            self.logger.warning("No se encontró la sección [SCORE] en config.ini. La detección de score no funcionará.")
            return None
        try:
            # El string se guarda como un diccionario, así que usamos eval de forma segura
            region_str = self.get('SCORE', 'score_region_relative')
            region_dict = eval(region_str)
            # Validar y devolver en formato x, y, width, height
            return {
                'x': int(region_dict['left']),
                'y': int(region_dict['top']),
                'width': int(region_dict['width']),
                'height': int(region_dict['height'])
            }
        except (configparser.NoOptionError, SyntaxError, NameError, KeyError) as e:
            self.logger.error(f"Error al leer 'score_region_relative' de config.ini: {e}")
            return None
