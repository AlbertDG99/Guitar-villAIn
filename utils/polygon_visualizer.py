#!/usr/bin/env python3
"""
🎯 VISUALIZADOR DE POLÍGONOS Y DETECCIÓN
=======================================
Visualiza polígonos configurados y detección de notas en tiempo real
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.screen_capture import ScreenCapture
from src.utils.config_manager import ConfigManager

class PolygonVisualizer:
    """Visualizador de polígonos y detección en tiempo real"""
    
    def __init__(self):
        # Configuración
        self.config_manager = ConfigManager()
        self.screen_capture = ScreenCapture(self.config_manager)
        
        # Cargar rangos HSV optimizados desde configuración global
        yellow_lower, yellow_upper = self.config_manager.get_yellow_hsv_range()
        green_lower, green_upper = self.config_manager.get_green_hsv_range()
        
        self.yellow_hsv = {
            'lower': np.array(yellow_lower),
            'upper': np.array(yellow_upper)
        }
        
        self.green_hsv = {
            'lower': np.array(green_lower),
            'upper': np.array(green_upper)
        }
        
        # Modo de visualización
        self.view_mode = 0  # 0=Normal, 1=Máscara Amarilla, 2=Máscara Verde
        self.view_modes = ["NORMAL", "MÁSCARA AMARILLA", "MÁSCARA VERDE"]
        
        # Cargar polígonos
        self.load_polygons()
        
        # Colores para visualización
        self.lane_colors = {
            'S': (0, 255, 0),    # Verde
            'D': (0, 255, 255),  # Cian
            'F': (255, 0, 0),    # Azul
            'J': (255, 0, 255),  # Magenta
            'K': (0, 128, 255),  # Naranja
            'L': (128, 255, 0)   # Verde-amarillo
        }
    
    def load_polygons(self):
        """Cargar polígonos configurados"""
        try:
            # Usar polígonos relativos directamente
            self.polygons = self.config_manager.get_note_lane_polygons_relative()
            
            if not self.polygons:
                # Fallback a polígonos absolutos con conversión
                original_polygons = self.config_manager.get_note_lane_polygons()
                
                # Calcular offset automático
                all_x = []
                all_y = []
                for points in original_polygons.values():
                    for point in points:
                        all_x.append(point[0])
                        all_y.append(point[1])
                
                if all_x and all_y:
                    offset_x = min(all_x)
                    offset_y = min(all_y)
                    
                    # Convertir usando offset
                    self.polygons = {}
                    for lane_name, points in original_polygons.items():
                        converted_points = []
                        for point in points:
                            relative_x = point[0] - offset_x
                            relative_y = point[1] - offset_y
                            converted_points.append((relative_x, relative_y))
                        self.polygons[lane_name] = converted_points
            
        except Exception as e:
            self.polygons = {}
    
    def detect_yellow_notes(self, hsv: np.ndarray) -> np.ndarray:
        """Detectar notas amarillas"""
        mask = cv2.inRange(hsv, self.yellow_hsv['lower'], self.yellow_hsv['upper'])
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def detect_green_notes(self, hsv: np.ndarray) -> np.ndarray:
        """Detectar notas verdes"""
        mask = cv2.inRange(hsv, self.green_hsv['lower'], self.green_hsv['upper'])
        kernel = np.ones((4, 4), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return mask
    
    def extract_lane_region(self, frame: np.ndarray, lane_name: str) -> np.ndarray:
        """Extraer región de un carril usando polígono"""
        if lane_name not in self.polygons:
            return np.zeros_like(frame)
            
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = self.polygons[lane_name]
        pts = np.array(points, np.int32)
        cv2.fillPoly(mask, [pts], (255,))
        return cv2.bitwise_and(frame, frame, mask=mask)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Procesar frame y retornar imagen con visualización"""
        detections = {
            'yellow': 0,
            'green': 0,
            'lanes': {}
        }
        
        # Convertir a HSV para máscaras de color
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Aplicar máscara según el modo de visualización
        if self.view_mode == 1:  # Máscara amarilla
            yellow_mask = cv2.inRange(hsv_frame, self.yellow_hsv['lower'], self.yellow_hsv['upper'])
            output_frame = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
        elif self.view_mode == 2:  # Máscara verde
            green_mask = cv2.inRange(hsv_frame, self.green_hsv['lower'], self.green_hsv['upper'])
            output_frame = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
        else:  # Modo normal
            output_frame = frame.copy()
        
        # Dibujar polígonos siempre (en todos los modos)
        for lane_name, points in self.polygons.items():
            pts = np.array(points, np.int32)
            color = self.lane_colors.get(lane_name, (255, 255, 255))
            cv2.polylines(output_frame, [pts], isClosed=True, color=color, thickness=2)
            
            # Etiqueta del carril
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(output_frame, lane_name, tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Procesar detecciones por carril (solo en modo normal)
        if self.view_mode == 0:
            for lane_name in self.polygons.keys():
                region = self.extract_lane_region(frame, lane_name)
                
                if region is None or region.size == 0:
                    continue
                
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                
                # Detectar amarillas
                yellow_mask = self.detect_yellow_notes(hsv)
                contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                yellow_count = 0
                for contour in contours_yellow:
                    area = cv2.contourArea(contour)
                    if area > 100:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        yellow_count += 1
                
                # Detectar verdes
                green_mask = self.detect_green_notes(hsv)
                contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                green_count = 0
                for contour in contours_green:
                    area = cv2.contourArea(contour)
                    if area > 80:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        green_count += 1
                
                detections['lanes'][lane_name] = {
                    'yellow': yellow_count,
                    'green': green_count
                }
                
                detections['yellow'] += yellow_count
                detections['green'] += green_count
        
        return output_frame, detections
    
    def run(self):
        """Ejecutar visualizador"""
        print("🎯 VISUALIZADOR INICIADO")
        print("Controles:")
        print("- 'q': Salir")
        print("- 's': Capturar frame")
        print("- '+': Cambiar vista (Normal → Amarilla → Verde)")
        print("- SPACE: Pausar/Reanudar")
        
        paused = False
        frame_count = 0
        
        try:
            while True:
                if not paused:
                    frame = self.screen_capture.capture_frame()
                    
                    if frame is not None:
                        output_frame, detections = self.process_frame(frame)
                        
                        # Información en pantalla
                        info_lines = [
                            f"📺 VISTA: {self.view_modes[self.view_mode]}",
                            f"🟡 {detections['yellow']} 🟢 {detections['green']}",
                            f"Frame: {frame_count}"
                        ]
                        
                        # Fondo para la información
                        overlay = output_frame.copy()
                        cv2.rectangle(overlay, (5, 5), (400, 90), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)
                        
                        for i, line in enumerate(info_lines):
                            y = 25 + (i * 25)
                            cv2.putText(output_frame, line, (10, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        frame_count += 1
                    else:
                        output_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(output_frame, "❌ Error de captura", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("Visualizador de Polígonos", output_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    cv2.imwrite(f"polygon_viz_{timestamp}.png", output_frame)
                    print(f"Frame guardado: polygon_viz_{timestamp}.png")
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️ Pausado' if paused else '▶️ Reanudado'}")
                elif key == ord('+') or key == ord('='):  # Tanto + como = (para teclados US)
                    self.view_mode = (self.view_mode + 1) % 3
                    print(f"📺 Vista cambiada a: {self.view_modes[self.view_mode]}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            cv2.destroyAllWindows()


def main():
    """Función principal"""
    try:
        visualizer = PolygonVisualizer()
        visualizer.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 