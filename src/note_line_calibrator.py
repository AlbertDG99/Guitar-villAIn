#!/usr/bin/env python3
"""
Note Line Calibrator - Calibrador de Líneas de Notas
====================================================

Herramienta interactiva para definir polígonos de detección para cada carril de notas.
Permite seleccionar 4 puntos por carril para crear áreas de detección diagonales.
"""

import cv2
import mss
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

class NoteLineCalibrator:
    """Calibrador de líneas de notas usando polígonos de 4 vértices."""
    
    def __init__(self):
        self.logger = setup_logger("NoteLineCalibrator")
        self.config_manager = ConfigManager()
        
        # Estado del calibrador
        self.current_lane = 0
        self.lane_polygons: Dict[int, List[Tuple[int, int]]] = {}
        self.current_points: List[Tuple[int, int]] = []
        self.total_lanes = 6  # S, D, F, J, K, L
        self.lane_names = ['S', 'D', 'F', 'J', 'K', 'L']
        
        # Imagen de trabajo
        self.base_image: Optional[np.ndarray] = None
        self.working_image: Optional[np.ndarray] = None
        
    def calibrate(self) -> bool:
        """Ejecuta el proceso de calibración completo."""
        self.logger.info("Iniciando calibración de líneas de notas...")
        
        try:
            # 1. Capturar pantalla
            if not self._capture_screen():
                return False
                
            # 2. Calibrar cada carril
            for lane_id in range(self.total_lanes):
                self.current_lane = lane_id
                lane_name = self.lane_names[lane_id]
                
                self.logger.info(f"Calibrando carril {lane_name} ({lane_id + 1}/{self.total_lanes})")
                
                if not self._calibrate_lane(lane_name):
                    self.logger.warning(f"Calibración del carril {lane_name} cancelada")
                    return False
                    
            # 3. Guardar configuración
            self._save_configuration()
            
            # 4. Generar imágenes de verificación
            self._generate_verification_images()
            
            self.logger.info("✅ Calibración completada exitosamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error durante la calibración: {e}", exc_info=True)
            return False
            
    def _capture_screen(self) -> bool:
        """Captura la pantalla para calibración."""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = np.array(sct.grab(monitor))
                self.base_image = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                self.working_image = self.base_image.copy()
                return True
        except Exception as e:
            self.logger.error(f"Error capturando pantalla: {e}")
            return False
            
    def _calibrate_lane(self, lane_name: str) -> bool:
        """Calibra un carril específico."""
        self.current_points = []
        window_name = f"Calibrar Carril {lane_name}"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        instructions = [
            f"Calibrando carril {lane_name}",
            "Haz clic en 4 puntos para formar el polígono:",
            "1. Esquina superior izquierda",
            "2. Esquina superior derecha", 
            "3. Esquina inferior derecha",
            "4. Esquina inferior izquierda",
            "",
            "Controles:",
            "- Clic izquierdo: Añadir punto",
            "- 'r': Reiniciar puntos del carril actual",
            "- 'c': Confirmar carril (cuando tengas 4 puntos)",
            "- 'q': Cancelar calibración"
        ]
        
        while True:
            # Crear imagen de trabajo
            if self.working_image is None:
                continue
            display_image = self.working_image.copy()
            
            # Dibujar polígonos de carriles ya calibrados
            self._draw_existing_polygons(display_image)
            
            # Dibujar puntos actuales
            self._draw_current_points(display_image)
            
            # Dibujar instrucciones
            self._draw_instructions(display_image, instructions)
            
            cv2.imshow(window_name, display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
                
            elif key == ord('r'):
                self.current_points = []
                self.logger.info(f"Reiniciando puntos del carril {lane_name}")
                
            elif key == ord('c') and len(self.current_points) == 4:
                self.lane_polygons[self.current_lane] = self.current_points.copy()
                self.logger.info(f"Carril {lane_name} confirmado con 4 puntos")
                cv2.destroyAllWindows()
                return True
                
        return False
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback para manejar clics del mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_points) < 4:
                self.current_points.append((x, y))
                lane_name = self.lane_names[self.current_lane]
                self.logger.info(f"Punto {len(self.current_points)}/4 añadido al carril {lane_name}: ({x}, {y})")
                
    def _draw_existing_polygons(self, image: np.ndarray):
        """Dibuja los polígonos de carriles ya calibrados."""
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (0, 128, 255), (128, 255, 0)]
        
        for lane_id, points in self.lane_polygons.items():
            if len(points) == 4:
                color = colors[lane_id % len(colors)]
                pts = np.array(points, np.int32)
                cv2.polylines(image, [pts], True, color, 2)
                
                # Etiqueta del carril
                center_x = sum(p[0] for p in points) // 4
                center_y = sum(p[1] for p in points) // 4
                cv2.putText(image, self.lane_names[lane_id], (center_x-10, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
    def _draw_current_points(self, image: np.ndarray):
        """Dibuja los puntos del carril actual en calibración."""
        color = (0, 0, 255)  # Rojo para puntos actuales
        
        # Dibujar puntos
        for i, point in enumerate(self.current_points):
            cv2.circle(image, point, 5, color, -1)
            cv2.putText(image, str(i+1), (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Dibujar líneas conectando los puntos
        if len(self.current_points) > 1:
            for i in range(len(self.current_points) - 1):
                cv2.line(image, self.current_points[i], self.current_points[i+1], color, 2)
                
        # Si tenemos 4 puntos, cerrar el polígono
        if len(self.current_points) == 4:
            cv2.line(image, self.current_points[3], self.current_points[0], color, 2)
            
    def _draw_instructions(self, image: np.ndarray, instructions: List[str]):
        """Dibuja las instrucciones en la imagen."""
        y_offset = 30
        for i, instruction in enumerate(instructions):
            y = y_offset + (i * 25)
            color = (255, 255, 255) if not instruction.startswith("Calibrando") else (0, 255, 0)
            cv2.putText(image, instruction, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    def _save_configuration(self):
        """Guarda la configuración de polígonos en config.ini."""
        # Guardar cada polígono como una sección separada
        for lane_id, points in self.lane_polygons.items():
            lane_name = self.lane_names[lane_id]
            section_name = f'NOTE_LANE_{lane_name}'
            
            if not self.config_manager.config.has_section(section_name):
                self.config_manager.config.add_section(section_name)
                
            # Guardar los 4 puntos
            for i, (x, y) in enumerate(points):
                self.config_manager.config.set(section_name, f'point_{i+1}_x', str(x))
                self.config_manager.config.set(section_name, f'point_{i+1}_y', str(y))
                
        self.config_manager.save_config()
        self.logger.info("Configuración de polígonos guardada en config.ini")
        
    def _generate_verification_images(self):
        """Genera imágenes de verificación mostrando los polígonos calibrados."""
        if self.base_image is None:
            self.logger.error("No hay imagen base para generar verificaciones")
            return
            
        verification_image = self.base_image.copy()
        
        # Dibujar todos los polígonos
        self._draw_existing_polygons(verification_image)
        
        # Guardar imagen de verificación
        cv2.imwrite("calibrated_note_lanes.png", verification_image)
        
        # Generar imágenes individuales de cada carril
        for lane_id, points in self.lane_polygons.items():
            lane_name = self.lane_names[lane_id]
            
            # Crear máscara del polígono
            mask = np.zeros(self.base_image.shape[:2], dtype=np.uint8)
            pts = np.array(points, np.int32)
            cv2.fillPoly(mask, [pts], (255,))
            
            # Aplicar máscara
            lane_image = cv2.bitwise_and(self.base_image, self.base_image, mask=mask)
            
            # Guardar imagen del carril
            cv2.imwrite(f"calibrated_lane_{lane_name}.png", lane_image)
            
        self.logger.info("Imágenes de verificación generadas")
        
    def get_lane_polygons(self) -> Dict[int, List[Tuple[int, int]]]:
        """Retorna los polígonos de carriles configurados."""
        return dict(self.lane_polygons)


def main():
    """Función principal para ejecutar el calibrador."""
    print("=== Calibrador de Líneas de Notas ===")
    print("Este calibrador te permite definir áreas poligonales para cada carril de notas.")
    print()
    
    calibrator = NoteLineCalibrator()
    
    if calibrator.calibrate():
        print("\n✅ Calibración completada exitosamente!")
        print("Archivos generados:")
        print("- calibrated_note_lanes.png (vista general)")
        print("- calibrated_lane_[S,D,F,J,K,L].png (carriles individuales)")
        print("- Configuración guardada en config/config.ini")
    else:
        print("\n❌ Calibración fallida o cancelada")


if __name__ == "__main__":
    main() 