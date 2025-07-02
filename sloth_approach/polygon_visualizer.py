#!/usr/bin/env python3
"""
Visualizador de Poligonos y Deteccion
=======================================
Visualiza poligonos configurados y deteccion de notas en tiempo real
"""

import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, Tuple
import os

from .screen_capture import ScreenCapture
from .config_manager import ConfigManager

class PolygonVisualizer:
    """Visualizador de poligonos, deteccion y score en tiempo real."""
    
    def __init__(self):
        # --- Ruta de configuracion robusta ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.ini')
        
        # --- Inicializacion de Componentes ---
        self.config_manager = ConfigManager(config_path=config_path)
        self.screen_capture = ScreenCapture(self.config_manager)
        
        # --- Carga de Configuracion ---
        hsv_ranges = self.config_manager.get_hsv_ranges()
        self.morphology_params = self.config_manager.get_morphology_params()

        # Acceso directo a los rangos (falla si la config es incorrecta)
        green_range = hsv_ranges['green']
        self.green_hsv = {
            'lower': np.array([green_range['h_min'], green_range['s_min'], green_range['v_min']]),
            'upper': np.array([green_range['h_max'], green_range['s_max'], green_range['v_max']])
        }
        
        yellow_range = hsv_ranges['yellow']
        self.yellow_hsv = {
            'lower': np.array([yellow_range['h_min'], yellow_range['s_min'], yellow_range['v_min']]),
            'upper': np.array([yellow_range['h_max'], yellow_range['s_max'], yellow_range['v_max']])
        }
        
        self.polygons = self.config_manager.get_note_lane_polygons_relative()

        # --- Estado y Control ---
        self.total_green_count = 0
        self.total_yellow_count = 0
        
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        self.lane_colors = {
            'S': (0, 255, 0), 'D': (0, 255, 255), 'F': (255, 0, 0),
            'J': (255, 0, 255), 'K': (0, 128, 255), 'L': (128, 255, 0)
        }

    def process_lane_micro_image(self, lane_data):
        """Procesar un carril completo en una micro-imagen (paralelismo real)"""
        lane_name, points, frame = lane_data
        
        # 1. Calcular bounding box del polígono para recorte mínimo
        pts = np.array(points, np.int32)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        
        # Añadir un pequeño margen para evitar cortes
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(frame.shape[1], x_max + margin)
        y_max = min(frame.shape[0], y_max + margin)
        
        # 2. Recortar micro-imagen del carril
        micro_frame = frame[y_min:y_max, x_min:x_max]
        
        if micro_frame.size == 0:
            return {
                'lane_name': lane_name,
                'green_boxes': [],
                'yellow_boxes': [],
                'green_count': 0,
                'yellow_count': 0
            }
        
        # 3. Ajustar coordenadas del polígono al recorte
        local_polygon = pts - np.array([x_min, y_min])
        
        # 4. Conversión HSV en micro-imagen (muy rápido)
        hsv_micro = cv2.cvtColor(micro_frame, cv2.COLOR_BGR2HSV)
        
        # 5. Detección de notas verdes en micro-imagen
        green_mask = cv2.inRange(hsv_micro, self.green_hsv['lower'], self.green_hsv['upper'])
        
        # Operaciones morfológicas en micro-imagen
        close_size = self.morphology_params['close_size']
        dilate_size = self.morphology_params['dilate_size']
        
        close_kernel = np.ones((close_size, close_size), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, close_kernel)
        
        dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
        green_mask = cv2.dilate(green_mask, dilate_kernel, iterations=1)
        
        open_kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, open_kernel)
        
        # 6. Búsqueda de contornos verdes
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lane_results = {
            'lane_name': lane_name,
            'green_boxes': [],
            'yellow_boxes': [],
            'green_count': 0,
            'yellow_count': 0
        }
        
        # 7. Filtrar contornos verdes dentro del polígono
        for contour in green_contours:
            area = cv2.contourArea(contour)
            if self.morphology_params['min_area'] <= area <= self.morphology_params['max_area']:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Verificar si está dentro del polígono local
                    if cv2.pointPolygonTest(local_polygon, (cx, cy), False) >= 0:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Convertir coordenadas locales a globales
                        global_x = x + x_min
                        global_y = y + y_min
                        
                        lane_results['green_boxes'].append({
                            'x': global_x, 'y': global_y, 'w': w, 'h': h, 'area': area
                        })
                        lane_results['green_count'] += 1
        
        # 8. Si hay notas verdes, saltar amarillas (optimización de juego)
        if lane_results['green_count'] > 0:
            return lane_results
        
        # 9. Detección de notas amarillas (solo si no hay verdes)
        yellow_mask = cv2.inRange(hsv_micro, self.yellow_hsv['lower'], self.yellow_hsv['upper'])
        
        # Operaciones morfológicas menos agresivas para amarillo
        close_size_yellow = max(3, close_size // 2)
        dilate_size_yellow = max(2, dilate_size // 2)
        
        close_kernel_yellow = np.ones((close_size_yellow, close_size_yellow), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel_yellow)
        
        dilate_kernel_yellow = np.ones((dilate_size_yellow, dilate_size_yellow), np.uint8)
        yellow_mask = cv2.dilate(yellow_mask, dilate_kernel_yellow, iterations=1)
        
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, open_kernel)
        
        # 10. Búsqueda de contornos amarillos
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 11. Filtrar contornos amarillos
        for contour in yellow_contours:
            area = cv2.contourArea(contour)
            if self.morphology_params['min_area'] <= area <= self.morphology_params['max_area']:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if cv2.pointPolygonTest(local_polygon, (cx, cy), False) >= 0:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Convertir coordenadas locales a globales
                        global_x = x + x_min
                        global_y = y + y_min
                        
                        lane_results['yellow_boxes'].append({
                            'x': global_x, 'y': global_y, 'w': w, 'h': h, 'area': area
                        })
                        lane_results['yellow_count'] += 1
        
        return lane_results

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Procesar frame usando paralelismo real por carril"""
        detections = {
            'yellow': 0,
            'green': 0,
            'lanes': {}
        }
        
        # --- PREPARACIÓN ---
        output_frame = frame.copy()

        # --- PARALELISMO REAL: Cada hilo procesa un carril completo ---
        lane_tasks = []
        for lane_name, points in self.polygons.items():
            if points:
                lane_tasks.append((lane_name, points, frame))
        
        total_green_detections = 0
        total_yellow_detections = 0
        
        if lane_tasks:
            # Usar todos los núcleos disponibles (hasta 6 carriles)
            max_workers = min(len(lane_tasks), 6)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_results = [executor.submit(self.process_lane_micro_image, task) for task in lane_tasks]
                
                # Recopilar resultados
                for future in future_results:
                    result = future.result()
                    lane_name = result['lane_name']
                    
                    # Guardar estadísticas del carril
                    detections['lanes'][lane_name] = {
                        'yellow': result['yellow_count'],
                        'green': result['green_count']
                    }
                    
                    # Acumular totales
                    total_green_detections += result['green_count']
                    total_yellow_detections += result['yellow_count']
                    
                    # Dibujar cajas
                    for box in result['green_boxes']:
                        cv2.rectangle(output_frame, (box['x'], box['y']), 
                                    (box['x'] + box['w'], box['y'] + box['h']), (0, 255, 0), 2)
                    
                    for box in result['yellow_boxes']:
                        cv2.rectangle(output_frame, (box['x'], box['y']), 
                                    (box['x'] + box['w'], box['y'] + box['h']), (0, 255, 255), 2)
        
        # Dibujar polígonos siempre
        for lane_name, points in self.polygons.items():
            pts = np.array(points, np.int32)
            color = self.lane_colors.get(lane_name, (255, 255, 255))
            cv2.polylines(output_frame, [pts], isClosed=True, color=color, thickness=2)
            
            # Etiqueta del carril
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(output_frame, lane_name, tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Actualizar contadores globales
        detections['yellow'] = total_yellow_detections
        detections['green'] = total_green_detections
        self.total_green_count = total_green_detections
        self.total_yellow_count = total_yellow_detections
        
        # AÑADIR CONTADOR SUPERIOR
        self.add_top_counter(output_frame)
        
        # Calcular FPS
        self.fps_counter += 1
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time >= 1.0:  # Actualizar cada segundo
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        # Solo mostrar FPS (esquina inferior derecha)
        fps_text = f"FPS: {self.current_fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Posición esquina inferior derecha
        fps_x = output_frame.shape[1] - text_size[0] - 10
        fps_y = output_frame.shape[0] - 10
        
        # Fondo para FPS
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (fps_x - 5, fps_y - text_size[1] - 5), 
                    (fps_x + text_size[0] + 5, fps_y + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)
        
        # Texto FPS
        cv2.putText(output_frame, fps_text, (fps_x, fps_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return output_frame, detections
    
    def add_top_counter(self, frame: np.ndarray):
        """Añadir contador superior con detecciones en tiempo real"""
        # Fondo para el contador (parte superior)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Texto del contador (sin emojis - OpenCV no los soporta)
        counter_text = f"VERDES: {self.total_green_count}  |  AMARILLAS: {self.total_yellow_count}"
        
        # Calcular posición centrada
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        x = (frame.shape[1] - text_size[0]) // 2
        y = 45
        
        # Dibujar texto con sombra
        cv2.putText(frame, counter_text, (x + 2, y + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)  # Sombra
        cv2.putText(frame, counter_text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)  # Texto principal
        
        # Indicador de configuración activa con valores reales
        if hasattr(self, 'morphology_params'):
            # Mostrar valores EXACTOS del archivo
            green_close = self.morphology_params['close_size']  # 20
            green_dilate = self.morphology_params['dilate_size']  # 15
            config_text = f"HSV Plus | V_min: {self.green_hsv['lower'][2]} | Close: {green_close} | Dilate: {green_dilate}"
        else:
            config_text = "Configuración estándar"
        
        cv2.putText(frame, config_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    def draw_fps(self, frame: np.ndarray):
        """Dibuja los FPS actuales en el frame."""
        fps_text = f"FPS: {self.current_fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def run(self):
        """Ejecutar visualizador"""
        print("\n" + "="*60)
        print("🎯 POLYGON VISUALIZER PLUS")
        print("="*60)
        
        # Mostrar configuración cargada
        print("Configuracion HSV y Morfologia cargada desde config.ini:")
        print(f"  Verde V_min: {self.green_hsv['lower'][2]}")
        print(f"  Morfologia VERDE: Close={self.morphology_params['close_size']}, Dilate={self.morphology_params['dilate_size']}")
        print(f"  Morfologia AMARILLA: Close={self.morphology_params['close_size']//2}, Dilate={self.morphology_params['dilate_size']//2}")
        print(f"  Area de Deteccion: {self.morphology_params['min_area']}-{self.morphology_params['max_area']}")
        
        print("\nCONTROLES:")
        print("- 'q': Salir")
        print("- SPACE: Pausar/Reanudar")
        print("\nCARACTERÍSTICAS:")
        print("- Detección por carriles con THREADING optimizado")
        print("- Una sola conversión HSV + morfología global por frame")
        print("- Lógica de juego optimizada (si hay verde, no busca amarilla)")
        print("- Procesamiento paralelo de hasta 6 carriles")
        print("- Contador de FPS en tiempo real")
        print("="*60)
        
        paused = False
        frame_count = 0
        
        try:
            self.screen_capture.start() # Iniciar captura en segundo plano
            time.sleep(1) # Dar tiempo al hilo para que capture el primer frame
            
            while True:
                # si está en pausa, no capturamos nuevo frame
                if not paused:
                    frame = self.screen_capture.get_latest_frame()

                if frame is not None:
                    # Procesar frame principal para detección
                    output_frame, detections = self.process_frame(frame)
                    
                    # --- DIBUJAR SUPERPOSICIONES (OVERLAYS) ---
                    self.add_top_counter(output_frame)
                    self.draw_fps(output_frame)

                    # Mostrar frame procesado
                    cv2.imshow('Guitar Hero IA - Visualizador', output_frame)
                else:
                    print("Frame no capturado, esperando...")

                # Verificar teclas presionadas (importante: fuera del if frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):  # Aceptar tanto q como Q
                    print("🛑 Saliendo del programa...")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️ Pausado' if paused else '▶️ Reanudado'}")
                elif key == 27:  # ESC como alternativa
                    print("🛑 Saliendo del programa (ESC)...")
                    break
                    
        except KeyboardInterrupt:
            print("🛑 Interrupción por teclado (Ctrl+C)")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("🔄 Limpiando recursos...")
            self.screen_capture.stop() # Detener captura en segundo plano
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