#!/usr/bin/env python3
"""
Window Calibrator - Calibrador de Ventana del Juego
===================================================

Herramienta interactiva para seleccionar y guardar las coordenadas
del área de juego y del área de puntuación.
"""

from typing import Dict, Optional
import cv2
import mss
import numpy as np
import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto a sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

# --- Variables globales para el callback del ratón ---
# No es ideal, pero es la forma más simple de interactuar con el callback de OpenCV
roi_points = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    """Función de callback para capturar los clics del ratón."""
    global roi_points, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        drawing = False

class WindowCalibrator:
    """Clase para calibrar la región de captura de la ventana del juego."""

    def __init__(self):
        self.logger = setup_logger("WindowCalibrator")
        self.config_manager = ConfigManager()

    def calibrate(self) -> bool:
        self.logger.info("Iniciando calibración interactiva...")
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = np.array(sct.grab(monitor))
                screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            # 1. Calibrar el área de juego principal
            game_region = self._select_area("Calibrar Area de Juego", screenshot_bgr.copy())
            if not game_region:
                self.logger.warning("No se seleccionó un área de juego válida.")
                cv2.destroyAllWindows()
                return False

            self.config_manager.set('CAPTURE', 'game_left', str(game_region['left'] + monitor['left']))
            self.config_manager.set('CAPTURE', 'game_top', str(game_region['top'] + monitor['top']))
            self.config_manager.set('CAPTURE', 'game_width', str(game_region['width']))
            self.config_manager.set('CAPTURE', 'game_height', str(game_region['height']))

            # 2. Recortar el área de juego de la captura original (no se necesita nueva captura)
            x, y, w, h = game_region.values()
            game_area_img_bgr = screenshot_bgr[y:y+h, x:x+w]

            # 3. Seleccionar el área de la puntuación DENTRO del área de juego
            score_region_relative = self._select_area(
                "Calibrar Puntuacion (DENTRO del area de juego)",
                game_area_img_bgr
            )

            if score_region_relative:
                # Guardar las coordenadas relativas
                self.config_manager.set('SCORE', 'score_region_relative', str(score_region_relative))

            self.config_manager.save_config()
            self.logger.info("✅ Calibración completada y guardada.")

            # --- 4. Guardar imágenes de verificación ---
            self.logger.info("Guardando imágenes de verificación...")
            cv2.imwrite("calibrated_game_region.png", game_area_img_bgr)

            if score_region_relative:
                x, y, w, h = score_region_relative.values()
                score_crop = game_area_img_bgr[y:y+h, x:x+w]
                cv2.imwrite("calibrated_score_region.png", score_crop)

            self.logger.info("Imágenes de verificación guardadas.")
            return True
        except Exception as e:
            self.logger.error("Ocurrió un error durante la calibración: %s", e, exc_info=True)
            return False

    def _select_area(self, window_name: str, image: np.ndarray) -> Optional[Dict[str, int]]:
        global roi_points, drawing
        roi_points = []
        drawing = False

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        self.logger.info("Arrastra el ratón para seleccionar. Pulsa 'c' para confirmar, 'q' para cancelar.")
        
        clone = image.copy()
        while True:
            temp_img = clone.copy()
            if len(roi_points) > 0:
                cv2.rectangle(temp_img, roi_points[0], (roi_points[-1][0] if drawing else roi_points[-1][0], roi_points[-1][1] if drawing else roi_points[-1][1]), (0, 255, 0), 2)
            cv2.imshow(window_name, temp_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(roi_points) == 2:
                break
            if key == ord('q'):
                roi_points = []
                break
        
        cv2.destroyAllWindows()
        
        if len(roi_points) == 2:
            p1, p2 = roi_points
            left = min(p1[0], p2[0])
            top = min(p1[1], p2[1])
            width = abs(p1[0] - p2[0])
            height = abs(p1[1] - p2[1])
            if width > 0 and height > 0:
                return {'left': left, 'top': top, 'width': width, 'height': height}
        return None

    def get_capture_region(self) -> Optional[Dict[str, int]]:
        return self.config_manager.get_capture_region()

    def get_score_region(self) -> Optional[Dict[str, int]]:
        return self.config_manager.get_score_region()


def main():
    """Función principal"""
    calibrator = WindowCalibrator()

    if calibrator.calibrate():
        print("Calibración completada exitosamente")
    else:
        print("Calibración fallida")


if __name__ == "__main__":
    main()
