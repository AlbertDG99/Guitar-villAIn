"""
Config Manager - Gestor de Configuración
=========================================

Maneja todas las configuraciones del sistema Guitar Hero IA.
"""

from typing import Dict, Any, Tuple, List, Optional
import configparser
from pathlib import Path


class ConfigManager:  # pylint: disable=too-many-instance-attributes
    """Gestor de configuración del sistema"""

    def __init__(self, config_file: str = "config/config.ini"):
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser(interpolation=None)
        self._load_config()

    def _load_config(self):
        """Cargar configuración desde archivo"""
        if self.config_file.exists():
            self.config.read(self.config_file)
        else:
            self._create_default_config()

    def create_default_config(self):
        """Crear configuración por defecto"""
        self._create_default_config()
        self._save_config()

    def _create_default_config(self):
        """Crear configuración por defecto"""
        # Configuración de captura de pantalla
        self.config['CAPTURE'] = {
            'method': 'mss',
            'screen_width': '2560',
            'screen_height': '1440',
            'game_left': '100',
            'game_top': '100',
            'game_width': '800',
            'game_height': '600',
            'downscale_factor': '1.0',
            'fps_limit': '60'
        }

        # Configuración de detección
        self.config['DETECTION'] = {
            'capture_fps': '30',
            'capture_region_x': '0',
            'capture_region_y': '0',
            'capture_region_width': '1920',
            'capture_region_height': '1080',
            'lane_positions': '150,250,350,450,550,650',
            'note_detection_threshold': '0.8',
            'note_colors_red': '255,128,64,255,128,0',
            'note_colors_green': '0,255,128,255,255,0',
            'note_colors_blue': '0,0,255,0,255,255',
            'sustain_note_color': '0,255,0'
        }

        self.config['HSV_RANGES'] = {
            'green_h_min': '40', 'green_s_min': '100', 'green_v_min': '100',
            'green_h_max': '80', 'green_s_max': '255', 'green_v_max': '255',
            'red_h_min': '0', 'red_s_min': '100', 'red_v_min': '100',
            'red_h_max': '10', 'red_s_max': '255', 'red_v_max': '255',
            # Añadir más colores si es necesario
        }

        # Configuración de timing
        self.config['TIMING'] = {
            'note_speed_pixels_per_second': '400',
            'perfect_timing_window_ms': '50',
            'good_timing_window_ms': '100',
            'ok_timing_window_ms': '150',
            'input_latency_compensation_ms': '10',
            'audio_latency_compensation_ms': '20',
            'show_timing_info': 'True',
            'show_ai_decisions': 'True'
        }

        # Configuración de input
        self.config['INPUT'] = {
            'key_bindings': 's,d,f,j,k,l',
            'key_press_duration_ms': '50',
            'simultaneous_keys_max': '6'
        }

        # Configuración de IA
        self.config['AI'] = {
            'learning_rate': '0.001',
            'epsilon': '0.1',
            'epsilon_decay': '0.995',
            'epsilon_min': '0.01',
            'batch_size': '32',
            'memory_size': '10000',
            'target_update_frequency': '100',
            'reward_perfect': '100',
            'reward_good': '50',
            'reward_ok': '25',
            'reward_miss': '-10'
        }

        # Configuración de logging
        self.config['LOGGING'] = {
            'log_level': 'INFO',
            'log_file': 'logs/guitar_hero_ia.log',
            'max_log_size_mb': '10',
            'backup_count': '5'
        }

        # Configuración de visualización
        self.config['VISUALIZATION'] = {
            'show_debug_window': 'True',
            'show_detection_overlay': 'True',
        }

    def _save_config(self):
        """Guardar configuración a archivo"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            self.config.write(f)

    def get(self, section: str, key: str, fallback: Any = "") -> str:
        """Obtener valor de configuración"""
        return self.config.get(section, key, fallback=fallback)

    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Obtener valor entero de configuración"""
        return self.config.getint(section, key, fallback=fallback)

    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Obtener valor flotante de configuración"""
        return self.config.getfloat(section, key, fallback=fallback)

    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Obtener valor booleano de configuración"""
        return self.config.getboolean(section, key, fallback=fallback)

    def set(self, section: str, key: str, value: Any):
        """Establecer valor de configuración"""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))
        self.save_config()

    def get_capture_region(self) -> Tuple[int, int, int, int]:
        """Obtener región de captura (x, y, width, height)"""
        return (
            self.getint('DETECTION', 'capture_region_x'),
            self.getint('DETECTION', 'capture_region_y'),
            self.getint('DETECTION', 'capture_region_width'),
            self.getint('DETECTION', 'capture_region_height')
        )

    def get_lane_positions(self) -> List[int]:
        """Obtener posiciones de los carriles"""
        positions_str = self.get('DETECTION', 'lane_positions')
        try:
            return [int(x.strip()) for x in positions_str.split(',') if x.strip()]
        except (ValueError, TypeError) as e:
            print(f"❌ Error en configuración lane_positions: {e}. Usando valores por defecto.")
            return [200, 333, 466, 600, 733, 866]

    def get_hsv_ranges(self) -> Dict[str, Dict[str, int]]:
        """Obtiene los rangos HSV para cada color desde la configuración."""
        ranges = {}
        
        # Intentar cargar desde archivo optimizado primero
        optimized_ranges = self._load_optimized_hsv_ranges()
        if optimized_ranges:
            return optimized_ranges
        
        # Fallback a configuración por defecto
        if not self.config.has_section('HSV_RANGES'):
            return ranges

        color_keys = set(opt.split('_')[0] for opt in self.config.options('HSV_RANGES'))

        for color in color_keys:
            ranges[color] = {
                'h_min': self.getint('HSV_RANGES', f'{color}_h_min', 0),
                's_min': self.getint('HSV_RANGES', f'{color}_s_min', 0),
                'v_min': self.getint('HSV_RANGES', f'{color}_v_min', 0),
                'h_max': self.getint('HSV_RANGES', f'{color}_h_max', 179),
                's_max': self.getint('HSV_RANGES', f'{color}_s_max', 255),
                'v_max': self.getint('HSV_RANGES', f'{color}_v_max', 255),
            }
        return ranges

    def _load_optimized_hsv_ranges(self) -> Optional[Dict[str, Dict[str, int]]]:
        """Cargar rangos HSV optimizados desde archivo"""
        import re
        
        hsv_file = Path("hsv_ranges_optimized.txt")
        if not hsv_file.exists():
            return None
            
        try:
            ranges = {}
            with open(hsv_file, 'r') as f:
                content = f.read()
                
            # Parsear usando regex para extraer valores de np.array
            patterns = {
                'yellow_lower': r'yellow_lower\s*=\s*np\.array\(\[([^\]]+)\]\)',
                'yellow_upper': r'yellow_upper\s*=\s*np\.array\(\[([^\]]+)\]\)',
                'green_lower': r'green_lower\s*=\s*np\.array\(\[([^\]]+)\]\)',
                'green_upper': r'green_upper\s*=\s*np\.array\(\[([^\]]+)\]\)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    values = [int(x.strip()) for x in match.group(1).split(',')]
                    if len(values) >= 3:
                        color = 'yellow' if 'yellow' in key else 'green'
                        range_type = 'lower' if 'lower' in key else 'upper'
                        
                        if color not in ranges:
                            ranges[color] = {}
                            
                        if range_type == 'lower':
                            ranges[color].update({
                                'h_min': values[0], 
                                's_min': values[1], 
                                'v_min': values[2]
                            })
                        else:
                            ranges[color].update({
                                'h_max': values[0], 
                                's_max': values[1], 
                                'v_max': values[2]
                            })
            
            if ranges:
                print(f"✅ Rangos HSV optimizados cargados: {ranges}")
                return ranges
                
        except Exception as e:
            print(f"⚠️ Error cargando rangos HSV optimizados: {e}")
            
        return None

    def get_yellow_hsv_range(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Obtener rango HSV específico para notas amarillas"""
        ranges = self.get_hsv_ranges()
        if 'yellow' in ranges:
            yellow = ranges['yellow']
            lower = (yellow['h_min'], yellow['s_min'], yellow['v_min'])
            upper = (yellow['h_max'], yellow['s_max'], yellow['v_max'])
            return lower, upper
        # Valores por defecto para amarillo
        return (15, 100, 100), (40, 255, 255)

    def get_green_hsv_range(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Obtener rango HSV específico para notas verdes"""
        ranges = self.get_hsv_ranges()
        if 'green' in ranges:
            green = ranges['green']
            lower = (green['h_min'], green['s_min'], green['v_min'])
            upper = (green['h_max'], green['s_max'], green['v_max'])
            return lower, upper
        # Valores por defecto para verde
        return (25, 40, 40), (95, 255, 255)

    def set_hsv_ranges(self, hsv_ranges: Dict[str, Dict[str, int]]):
        """Establece los rangos HSV en la configuración."""
        if not self.config.has_section('HSV_RANGES'):
            self.config.add_section('HSV_RANGES')

        for color, params in hsv_ranges.items():
            for key, value in params.items():
                self.config.set('HSV_RANGES', f"{color}_{key}", str(value))
        self.save_config()

    def get_note_colors(self) -> List[Tuple[int, int, int]]:
        """Obtener colores de las notas (RGB)"""
        default_colors = [
            (255, 128, 64), (255, 128, 0), (0, 255, 128),
            (255, 255, 0), (0, 0, 255), (0, 255, 255)
        ]
        try:
            color_names = ['red', 'green', 'blue']
            parsed_colors = []
            for name in color_names:
                values_str = self.get('DETECTION', f'note_colors_{name}')
                values = [int(v.strip()) for v in values_str.split(',') if v.strip()]
                parsed_colors.append(tuple(values))

            return list(zip(*parsed_colors))
        except (ValueError, TypeError, configparser.NoOptionError) as e:
            print(f"❌ Error en configuración note_colors: {e}. Usando valores por defecto.")
            return default_colors

    def get_key_bindings(self) -> List[str]:
        """Obtener las teclas para cada carril."""
        try:
            keys_str = self.get('INPUT', 'key_bindings', 's,d,f,j,k,l')
            return [k.strip() for k in keys_str.split(',') if k.strip()]
        except (configparser.NoSectionError, configparser.NoOptionError):
            return ['s', 'd', 'f', 'j', 'k', 'l']

    def get_ai_config(self) -> Dict[str, Any]:
        """Obtener configuración de la IA."""
        if not self.config.has_section('AI'):
            return {}
        return dict(self.config.items('AI'))

    def save_config(self):
        """Guardar configuración a archivo"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            self.config.write(f)

    def set_capture_region(self, region: Dict[str, int]):
        """Guarda la región de captura en la configuración."""
        if 'CAPTURE' not in self.config:
            self.config.add_section('CAPTURE')
        self.config.set('CAPTURE', 'capture_region', str(region))

    def get_score_region(self) -> Optional[Dict[str, int]]:
        """Obtiene la región de la puntuación como un diccionario."""
        region_str = self.get('SCORE', 'score_region')
        if region_str:
            try:
                # El string se guarda como un diccionario, así que usamos eval
                return eval(region_str)
            except (SyntaxError, NameError):
                return None
        return None

    def set_score_region(self, region: Dict[str, int]):
        """Guarda la región de la puntuación en la configuración."""
        if 'SCORE' not in self.config:
            self.config.add_section('SCORE')
        self.config.set('SCORE', 'score_region', str(region))

    def get_hsv_color_ranges(self) -> Dict[str, Dict[str, int]]:
        """Obtiene los rangos de color HSV para la detección de notas."""
        colors = ['green', 'red', 'yellow', 'blue', 'orange']
        ranges = {}
        for color in colors:
            ranges[color] = {
                'h_min': self.getint('HSV_RANGES', f'{color}_h_min', 0),
                's_min': self.getint('HSV_RANGES', f'{color}_s_min', 0),
                'v_min': self.getint('HSV_RANGES', f'{color}_v_min', 0),
                'h_max': self.getint('HSV_RANGES', f'{color}_h_max', 179),
                's_max': self.getint('HSV_RANGES', f'{color}_s_max', 255),
                'v_max': self.getint('HSV_RANGES', f'{color}_v_max', 255),
            }
        return ranges

    def get_note_lane_polygons(self) -> Dict[str, List[Tuple[int, int]]]:
        """Obtiene los polígonos de las líneas de notas."""
        lane_names = ['S', 'D', 'F', 'J', 'K', 'L']
        polygons = {}
        
        for lane_name in lane_names:
            # Intentar primero las nuevas secciones LANE_POLYGON_*
            section_name = f'LANE_POLYGON_{lane_name}'
            if self.config.has_section(section_name):
                points = []
                point_count = self.getint(section_name, 'point_count', 0)
                for i in range(point_count):  # Número variable de puntos
                    x_key = f'point_{i}_x'
                    y_key = f'point_{i}_y'
                    if (self.config.has_option(section_name, x_key) and 
                        self.config.has_option(section_name, y_key)):
                        x = self.getint(section_name, x_key)
                        y = self.getint(section_name, y_key)
                        points.append((x, y))
                
                if len(points) >= 3:  # Al menos 3 puntos para un polígono
                    polygons[lane_name] = points
            else:
                # Fallback a las secciones antiguas NOTE_LANE_*
                section_name = f'NOTE_LANE_{lane_name}'
                if self.config.has_section(section_name):
                    points = []
                    for i in range(1, 5):  # 4 puntos por polígono (formato antiguo)
                        x_key = f'point_{i}_x'
                        y_key = f'point_{i}_y'
                        if (self.config.has_option(section_name, x_key) and 
                            self.config.has_option(section_name, y_key)):
                            x = self.getint(section_name, x_key)
                            y = self.getint(section_name, y_key)
                            points.append((x, y))
                    
                    if len(points) == 4:
                        polygons[lane_name] = points
                    
        return polygons

    def get_note_lane_polygons_relative(self) -> Dict[str, List[Tuple[int, int]]]:
        """Obtiene los polígonos de las líneas de notas en coordenadas relativas al área de juego."""
        lane_names = ['S', 'D', 'F', 'J', 'K', 'L']
        polygons = {}
        
        # Obtener offset del área de juego para translación si es necesario
        game_left = int(self.get('CAPTURE', 'game_left', '0'))
        game_top = int(self.get('CAPTURE', 'game_top', '0'))
        
        for lane_name in lane_names:
            # Intentar primero las nuevas secciones LANE_POLYGON_* (ya en coordenadas relativas)
            section_name = f'LANE_POLYGON_{lane_name}'
            if self.config.has_section(section_name):
                points = []
                point_count = self.getint(section_name, 'point_count', 0)
                for i in range(point_count):  # Número variable de puntos
                    x_key = f'point_{i}_x'
                    y_key = f'point_{i}_y'
                    if (self.config.has_option(section_name, x_key) and 
                        self.config.has_option(section_name, y_key)):
                        x = self.getint(section_name, x_key)
                        y = self.getint(section_name, y_key)
                        # LANE_POLYGON_* ya están en coordenadas relativas - no aplicar translación
                        points.append((x, y))
                
                if len(points) >= 3:  # Al menos 3 puntos para un polígono
                    polygons[lane_name] = points
            else:
                # Fallback a las secciones antiguas NOTE_LANE_* (necesitan translación)
                section_name = f'NOTE_LANE_{lane_name}'
                if self.config.has_section(section_name):
                    points = []
                    for i in range(1, 5):  # 4 puntos por polígono (formato antiguo)
                        x_key = f'point_{i}_x'
                        y_key = f'point_{i}_y'
                        if (self.config.has_option(section_name, x_key) and 
                            self.config.has_option(section_name, y_key)):
                            x = self.getint(section_name, x_key)
                            y = self.getint(section_name, y_key)
                            # NOTE_LANE_* están en coordenadas absolutas - aplicar translación
                            relative_x = x - game_left
                            relative_y = y - game_top
                            points.append((relative_x, relative_y))
                    
                    if len(points) == 4:
                        polygons[lane_name] = points
                    
        return polygons

    def has_note_lane_polygons(self) -> bool:
        """Verifica si hay polígonos de líneas de notas configurados."""
        polygons = self.get_note_lane_polygons()
        return len(polygons) > 0
