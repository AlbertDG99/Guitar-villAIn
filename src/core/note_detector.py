"""
Note Detector - Detector de Notas
=================================

Detector de notas usando Computer Vision para identificar notas en la pantalla.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
import os

import cv2
import numpy as np

from src.utils.logger import setup_logger, performance_logger


@dataclass
class Note:  # pylint: disable=too-many-instance-attributes
    """Clase para representar una nota detectada"""
    lane: int  # Carril (0-5 para S,D,F,J,K,L)
    x: int     # Posición X
    y: int     # Posición Y
    width: int
    height: int
    confidence: float  # Confianza de detección (0-1)
    note_type: str    # 'normal', 'sustain', 'star'
    color: Tuple[int, int, int]  # Color BGR para visualización
    timestamp: float  # Tiempo de detección


@dataclass
class TemplateData:
    """Clase para almacenar los datos de una plantilla pre-cargada."""
    name: str
    color_name: str
    note_type: str
    bgr_data: np.ndarray
    mask: Optional[np.ndarray]
    width: int
    height: int
    color_bgr_vis: Tuple[int, int, int]


class NoteDetector:  # pylint: disable=too-many-instance-attributes
    """Detector de notas en tiempo real usando un enfoque híbrido."""

    def __init__(self, config_manager, templates_path: str = "data/templates"):
        self.config_manager = config_manager
        self.logger = setup_logger('NoteDetector')

        # Configuración de detección
        self.lane_positions = self.config_manager.get_lane_positions()
        self.detection_threshold = self.config_manager.getfloat('DETECTION', 'note_detection_threshold', 0.8)
        
        # --- NUEVO: Carga de rangos HSV para el pre-filtrado ---
        self.hsv_color_ranges = self.config_manager.get_hsv_ranges()
        self.templates = self._load_templates(templates_path)
        
        # Configuración de áreas de detección (obsoleto con template matching pero se mantiene por ahora)
        self.min_note_area = self.config_manager.getint('DETECTION', 'min_note_area', 100)
        self.min_sustain_area = self.config_manager.getint('DETECTION', 'min_sustain_area', 200)

        # Estadísticas
        self.notes_detected = 0
        self.detection_times: List[float] = []

        self.logger.info("NoteDetector (Híbrido) inicializado. %d plantillas y %d rangos HSV cargados.",
                         len(self.templates), len(self.hsv_color_ranges))
        self.logger.debug("Posiciones de carriles: %s", self.lane_positions)

    def _load_templates(self, templates_path: str) -> Dict[str, TemplateData]:
        """Carga las plantillas de imágenes desde la ruta especificada."""
        templates = {}
        if not os.path.exists(templates_path):
            self.logger.error("La ruta de plantillas no existe: %s", templates_path)
            return templates

        for filename in os.listdir(templates_path):
            if filename.endswith('.png'):
                try:
                    name = os.path.splitext(filename)[0]
                    path = os.path.join(templates_path, filename)
                    template_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if template_img is None:
                        continue

                    template_bgr = template_img[:, :, :3]
                    mask = template_img[:, :, 3] if template_img.shape[2] == 4 else None
                    height, width, _ = template_bgr.shape

                    parts = name.split('_')
                    color_str = parts[0]
                    note_type = parts[1] if len(parts) > 1 else 'normal'
                    
                    color_bgr_vis = self._get_color_bgr(color_str)

                    templates[name] = TemplateData(
                        name=name,
                        color_name=color_str,
                        note_type=note_type,
                        bgr_data=template_bgr,
                        mask=mask,
                        width=width,
                        height=height,
                        color_bgr_vis=color_bgr_vis
                    )
                    self.logger.info("Plantilla cargada: %s", name)
                except Exception as e:
                    self.logger.error("Error al cargar la plantilla %s: %s", filename, e)
        return templates
    
    def _get_color_bgr(self, color_name: str) -> Tuple[int, int, int]:
        """Devuelve un color BGR para visualización basado en el nombre del color."""
        colors = {
            'green': (0, 255, 0), 'yellow': (0, 255, 255), 'red': (0, 0, 255),
            'blue': (255, 0, 0), 'orange': (0, 165, 255)
        }
        return colors.get(color_name.lower(), (255, 255, 255))
    
    def _non_max_suppression(self, notes: List[Note], overlap_thresh: float = 0.3) -> List[Note]:
        """Elimina las detecciones de notas que se superponen."""
        if not notes:
            return []

        # Convertir objetos Note a cajas (x1, y1, x2, y2) y confianzas
        boxes = np.array([[n.x, n.y, n.x + n.width, n.y + n.height] for n in notes])
        confidences = np.array([n.confidence for n in notes])

        pick = []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Ordenar por confianza
        idxs = np.argsort(confidences)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # Encontrar las coordenadas de la caja más grande
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Calcular el ancho y alto de la caja de superposición
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # Calcular el ratio de superposición
            overlap = (w * h) / area[idxs[:last]]
            
            # Eliminar los índices de las cajas que se superponen
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        # Devolver solo las notas seleccionadas
        return [notes[i] for i in pick]

    def get_hsv_color_ranges(self) -> Dict:
        """Devuelve la configuración HSV actual."""
        return self.hsv_color_ranges.copy()

    def set_hsv_color_ranges(self, color_name: str, ranges: Dict[str, int]):
        """
        Actualiza los rangos HSV para un color específico.
        """
        if color_name in self.hsv_color_ranges:
            self.hsv_color_ranges[color_name] = ranges
            self.logger.info("Configuración HSV actualizada para %s en el detector.", color_name)
        else:
            self.logger.warning("Intento de actualizar rangos para un color no existente: %s", color_name)

    def detect_notes(self, frame: np.ndarray) -> List[Note]:
        """Detecta notas en el frame actual usando el método híbrido."""
        start_time = time.time()
        if frame is None:
            return []

        all_notes = []
        # Para cada color que podemos detectar (verde, amarillo), ejecutamos la detección
        for color_name in self.hsv_color_ranges:
            notes, _ = self._detect_notes_by_template(frame, color_name)
            if notes:
                all_notes.extend(notes)

        # Non-max suppression para eliminar detecciones duplicadas entre todos los colores
        final_notes = self._non_max_suppression(all_notes)
        
        detection_time = (time.time() - start_time) * 1000
        self.detection_times.append(detection_time)
        performance_logger.log_timing('note_detection', detection_time)

        if final_notes:
            self.notes_detected += len(final_notes)
            self.logger.debug("%d notas detectadas en %.2fms", len(final_notes), detection_time)
        
        return final_notes

    def debug_and_detect_one_color(self, frame: np.ndarray, color_name: str) -> Tuple[List[Note], List[np.ndarray]]:
        """
        Ejecuta el pipeline de detección para un solo color y devuelve las notas
        y las ROIs procesadas para depuración.
        """
        if frame is None:
            return [], []

        notes, rois = self._detect_notes_by_template(frame, color_name)
        final_notes = self._non_max_suppression(notes)
        
        return final_notes, rois

    def _merge_rects(self, rects: list, padding: int = 25) -> list:
        """Fusiona rectángulos superpuestos o adyacentes en una lista."""
        if not rects:
            return []

        merged = True
        while merged:
            merged = False
            for i in range(len(rects) - 1, -1, -1):
                for j in range(i - 1, -1, -1):
                    r1 = rects[i]
                    r2 = rects[j]
                    
                    # Comprobar si los rectángulos se solapan (con un padding extra)
                    if (r1[0] < r2[0] + r2[2] + padding and
                        r1[0] + r1[2] > r2[0] - padding and
                        r1[1] < r2[1] + r2[3] + padding and
                        r1[1] + r1[3] > r2[1] - padding):
                        
                        # Fusionar r1 y r2
                        x1 = min(r1[0], r2[0])
                        y1 = min(r1[1], r2[1])
                        x2 = max(r1[0] + r1[2], r2[0] + r2[2])
                        y2 = max(r1[1] + r1[3], r2[1] + r2[3])
                        
                        rects[j] = (x1, y1, x2 - x1, y2 - y1)
                        rects.pop(i)
                        merged = True
                        break
                if merged:
                    break
        return rects

    def _detect_notes_by_template(self, frame: np.ndarray, color_name: str) -> Tuple[List[Note], List[np.ndarray]]:
        """Realiza la detección híbrida: filtro de color para ROIs + template matching."""
        all_detections: List[Note] = []
        processed_rois: List[np.ndarray] = []

        # 1. Filtrado rápido por color para encontrar contornos
        mask = self._create_hsv_mask(frame, color_name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return [], []

        # 2. Fusionar los rectángulos de los contornos para crear ROIs más grandes
        initial_rects = [cv2.boundingRect(c) for c in contours]
        merged_rois = self._merge_rects(initial_rects)

        templates_for_color = [t for t in self.templates.values() if t.color_name == color_name]
        if not templates_for_color:
            return [], []

        # 3. Para cada ROI fusionada, hacer template matching
        for x, y, w, h in merged_rois:
            # Añadir una holgura generosa para asegurar la captura completa de la nota
            padding = 35
            roi_x = max(0, x - padding)
            roi_y = max(0, y - padding)
            roi_w = w + (padding * 2)
            roi_h = h + (padding * 2)
            
            # Asegurarse de que el recorte no se salga de los límites del frame
            roi_frame = frame[roi_y:min(roi_y + roi_h, frame.shape[0]), 
                              roi_x:min(roi_x + roi_w, frame.shape[1])]

            if roi_frame.size == 0:
                continue

            processed_rois.append(roi_frame)

            # Probar todas las plantillas de este color en la ROI
            for template in templates_for_color:
                if roi_frame.shape[0] < template.height or roi_frame.shape[1] < template.width:
                    continue

                result = cv2.matchTemplate(roi_frame, template.bgr_data, cv2.TM_CCOEFF_NORMED, mask=template.mask)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if np.isfinite(max_val) and max_val >= self.detection_threshold:
                    abs_x = roi_x + max_loc[0]
                    abs_y = roi_y + max_loc[1]

                    note = Note(
                        lane=-1,
                        x=abs_x,
                        y=abs_y,
                        width=template.width,
                        height=template.height,
                        confidence=float(max_val),
                        note_type=template.note_type,
                        color=template.color_bgr_vis,
                        timestamp=time.time()
                    )
                    lane = self._get_lane_from_position(note.x)
                    if lane is not None:
                        note.lane = lane
                        all_detections.append(note)

        return all_detections, processed_rois

    def _create_hsv_mask(self, frame: np.ndarray, color_name: str) -> np.ndarray:
        """Crea una máscara de color HSV a partir de un frame BGR."""
        if color_name not in self.hsv_color_ranges:
            self.logger.warning("Color '%s' no encontrado en la configuración HSV.", color_name)
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # CORRECCIÓN: Convertir el frame a HSV antes de aplicar el rango
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hsv_config = self.hsv_color_ranges[color_name]
        lower = np.array([hsv_config['h_min'], hsv_config['s_min'], hsv_config['v_min']])
        upper = np.array([hsv_config['h_max'], hsv_config['s_max'], hsv_config['v_max']])

        mask = cv2.inRange(hsv_frame, lower, upper)
        return mask

    def _get_lane_from_position(self, x_pos: float) -> Optional[int]:
        """Determina a qué carril pertenece una posición x encontrando el más cercano."""
        if not self.lane_positions:
            return None
            
        # Calcular la distancia de la nota a cada carril
        distances = [abs(x_pos - lane_x) for lane_x in self.lane_positions]
        
        # Encontrar el índice del carril con la mínima distancia
        min_distance_index = np.argmin(distances)
        
        # Opcional: añadir un umbral de distancia máxima para evitar asignaciones incorrectas
        # max_allowed_distance = (self.lane_positions[1] - self.lane_positions[0]) / 2
        # if distances[min_distance_index] > max_allowed_distance:
        #     return None
            
        return int(min_distance_index)

    def visualize_detection(self, frame: np.ndarray, notes: List[Note]) -> np.ndarray:
        """
        Visualizar las notas detectadas en un frame

        Args:
            frame: Frame original
            notes: Lista de notas detectadas

        Returns:
            Frame con las detecciones visualizadas
        """
        output_frame = self.draw_lane_lines(frame.copy())
        output_frame = self.draw_detected_notes(output_frame, notes)
        
        text = f"Notas detectadas: {len(notes)}"
        cv2.putText(
            output_frame, text, (10, output_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        return output_frame

    def draw_lane_lines(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja las líneas de los carriles en el frame."""
        for i, lane_x in enumerate(self.lane_positions):
            cv2.line(frame, (lane_x, 0), (lane_x, frame.shape[0]), (100, 100, 100), 1)
            cv2.putText(frame, f"L{i}", (lane_x - 10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def draw_detected_notes(self, frame: np.ndarray, notes: List[Note]) -> np.ndarray:
        """Dibuja las notas detectadas en el frame."""
        for note in notes:
            cv2.rectangle(
                frame, (note.x, note.y),
                (note.x + note.width, note.y + note.height), note.color, 2
            )
            info = f"L{note.lane} {note.confidence:.2f}"
            cv2.putText(
                frame, info, (note.x, note.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, note.color, 2
            )
        return frame

    def get_detection_stats(self) -> Dict:
        """Devuelve estadísticas de detección."""
        if not self.detection_times:
            avg_time = 0
        else:
            avg_time = sum(self.detection_times) / len(self.detection_times)

        return {
            "total_notes_detected": self.notes_detected,
            "average_detection_time_ms": avg_time,
        }
