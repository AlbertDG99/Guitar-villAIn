#!/usr/bin/env python3
"""
Visualizador de Poligonos y Deteccion con IA
============================================
Visualiza poligonos configurados, deteccion de notas en tiempo real
y ejecuta pulsaciones automaticas de teclas con logica anti-ban.
"""

import threading
from typing import Dict, Tuple
import os
import pydirectinput
import traceback
import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
import random
from mss import mss

from .screen_capture import ScreenCapture
from .config_manager import ConfigManager


class PolygonVisualizer:
    """Visualizador de poligonos, deteccion y IA para Guitar Hero."""
    
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
        self.running = True
        self.total_green_count = 0
        self.total_yellow_count = 0
        
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        self.lane_colors = {
            'S': (0, 255, 0), 'D': (0, 255, 255), 'F': (255, 0, 0),
            'J': (255, 0, 255), 'K': (0, 128, 255), 'L': (128, 255, 0)
        }

        # --- Sistema de Input y IA ---
        self.input_enabled = True # Siempre habilitado
        self.setup_input_system()

    def setup_input_system(self):
        """Configura el sistema de input, incluyendo el mapa de scancodes."""
        # Mapeo de carriles a teclas
        self.lane_to_key = {
            'S': 's', 'D': 'd', 'F': 'f',
            'J': 'j', 'K': 'k', 'L': 'l'
        }
        
        # Estado de las teclas
        self.keys_pressed = {lane: False for lane in self.lane_to_key.keys()}
        self.green_hold_active = {lane: False for lane in self.lane_to_key.keys()}
        self.key_cooldowns = {lane: 0.0 for lane in self.lane_to_key.keys()}
        
        # Configuración anti-ban
        self.action_delay_range = (0.01, 0.03)  # Reacción casi instantánea
        self.yellow_press_probability = 1  # 95% de probabilidad de pulsar amarillas
        self.green_press_probability = 1       # 100% de fiabilidad para notas verdes
        self.cooldown_duration = 0.15          # Cooldown general estable para amarillas
        self.green_note_cooldown = 0.3         # Cooldown de verdes más largo para evitar doble detección
        self.yellow_duration_range = (0.08, 0.15)
        self.fast_green_double_tap_duration = (0.05, 0.1) # Duración para el toque rápido de doble nota verde
        
        # Hilos de pulsaciones
        self.active_press_threads = {}
        self.press_thread_lock = threading.Lock()
        
        # Configurar pydirectinput
        pydirectinput.FAILSAFE = False
        pydirectinput.PAUSE = 0.001
        print("🎮 Sistema de IA configurado - Pulsaciones de bajo nivel habilitadas.")

    def press_key_with_duration(self, key: str, duration: float, lane: str):
        """Hilo para manejar pulsación de tecla con duración específica (amarillas)"""
        try:
            # Añadir retardo aleatorio para comportamiento más humano
            time.sleep(random.uniform(*self.action_delay_range))
            
            # Registrar hilo activo
            with self.press_thread_lock:
                self.active_press_threads[lane] = threading.current_thread()
            
            # Pulsación real de tecla
            pydirectinput.keyDown(key)
            self.keys_pressed[lane] = True
            
            # Mantener presionada por la duración especificada
            time.sleep(duration)
            
            # Liberación real de tecla
            pydirectinput.keyUp(key)
            self.keys_pressed[lane] = False
            
        except Exception as e:
            print(f"❌ Error en pulsación de {key}: {e}")
        finally:
            # Limpiar hilo activo
            with self.press_thread_lock:
                if lane in self.active_press_threads:
                    del self.active_press_threads[lane]

    def _execute_start_green_hold(self, key: str, lane: str):
        """(Hilo) Espera y luego inicia una pulsación mantenida."""
        try:
            time.sleep(random.uniform(*self.action_delay_range))
            
            pydirectinput.keyDown(key)
            self.keys_pressed[lane] = True
            print(f"🟢 HOLD START en carril {lane} (Tecla: {key})")

        except Exception as e:
            print(f"❌ Error al iniciar hold verde de {key}: {e}")
            # Si falla, revertimos los estados para evitar bloqueos
            self.green_hold_active[lane] = False
            self.keys_pressed[lane] = False

    def _execute_end_green_hold(self, key: str, lane: str):
        """(Hilo) Espera y luego finaliza una pulsación mantenida."""
        try:
            time.sleep(random.uniform(*self.action_delay_range))
            
            pydirectinput.keyUp(key)
            self.keys_pressed[lane] = False
            print(f"🔴 HOLD END en carril {lane} (Tecla: {key})")

        except Exception as e:
            print(f"❌ Error al finalizar hold verde de {key}: {e}")
            # Revertir solo la tecla presionada, el hold ya se considera terminado
            self.keys_pressed[lane] = False

    def panic_release_all_keys(self):
        """Función de pánico: Soltar todas las teclas inmediatamente"""
        print("🚨 PÁNICO: Soltando todas las teclas...")
        
        try:
            # Soltar todas las teclas en el juego
            for lane, key in self.lane_to_key.items():
                pydirectinput.keyUp(key)
                self.keys_pressed[lane] = False
                self.green_hold_active[lane] = False
            
            # Limpiar hilos activos de pulsaciones
            with self.press_thread_lock:
                self.active_press_threads.clear()
            
            print("✅ Todas las teclas liberadas correctamente")
            
        except Exception as e:
            print(f"❌ Error en función de pánico: {e}")

    def handle_yellow_note(self, lane: str):
        """Maneja la detección de una nota amarilla con lógica anti-ban"""
        current_time = time.time()
        
        # Verificar cooldown
        if current_time - self.key_cooldowns[lane] < self.cooldown_duration:
            return  # En cooldown, ignorar
        

        # Verificar si ya hay una tecla presionada en este carril
        if self.keys_pressed[lane]:
            return  # Ya está presionada, ignorar
        
        # Lógica anti-ban: probabilidad aleatoria
        if random.random() > self.yellow_press_probability:
            return  # No pulsar esta vez (80% probabilidad)
        
        # Duración aleatoria para la pulsación
        duration = random.uniform(*self.yellow_duration_range)
        
        # Actualizar cooldown
        self.key_cooldowns[lane] = current_time
        
        # Lanzar hilo de pulsación
        key = self.lane_to_key[lane]
        press_thread = threading.Thread(
            target=self.press_key_with_duration,
            args=(key, duration, lane),
            daemon=True
        )
        press_thread.start()

    def handle_green_note(self, lane: str, count: int):
        """Maneja la detección de una nota verde, con lógica especial para dobles."""
        current_time = time.time()

        # --- Caso especial: Doble nota verde detectada simultáneamente ---
        if count >= 2:
            # Usar el cooldown general rápido para esta acción especial
            if current_time - self.key_cooldowns[lane] < self.cooldown_duration:
                return
            
            # No actuar si ya hay una tecla presionada para evitar conflictos
            if self.keys_pressed[lane]:
                return

            print(f"⚡️ Doble verde en {lane}. Ejecutando toque rápido.")
            
            key = self.lane_to_key[lane]
            duration = random.uniform(*self.fast_green_double_tap_duration)
            
            self.key_cooldowns[lane] = current_time
            
            # Reutilizar el hilo de pulsación con duración, ideal para esta tarea
            press_thread = threading.Thread(
                target=self.press_key_with_duration,
                args=(key, duration, lane),
                daemon=True
            )
            press_thread.start()
            return  # La acción para este carril está decidida

        # --- Lógica normal de notas verdes (inicio/fin de hold) ---
        
        # Verificar cooldown específico para notas verdes para evitar re-triggering
        if current_time - self.key_cooldowns[lane] < self.green_note_cooldown:
            return
        
        # Lógica anti-ban: probabilidad aleatoria
        if random.random() > self.green_press_probability:
            return
        
        key = self.lane_to_key[lane]
        
        # Si no hay hold activo, es la primera nota verde (iniciar hold)
        if not self.green_hold_active[lane]:
            # Actualizamos estado síncronamente para evitar race conditions
            self.green_hold_active[lane] = True
            self.key_cooldowns[lane] = current_time
            
            # Lanzamos la acción de pulsación en un hilo para no bloquear
            threading.Thread(
                target=self._execute_start_green_hold,
                args=(key, lane),
                daemon=True
            ).start()
            
        # Si ya hay hold activo, es la segunda nota verde (finalizar hold)
        else:
            # Actualizamos estado síncronamente
            self.green_hold_active[lane] = False
            self.key_cooldowns[lane] = current_time

            # Lanzamos la acción de liberación en un hilo
            threading.Thread(
                target=self._execute_end_green_hold,
                args=(key, lane),
                daemon=True
            ).start()

    def process_ai_actions(self, detections: Dict):
        """Procesa las detecciones y ejecuta acciones de IA"""
        # Procesar cada carril
        for lane_name, lane_data in detections['lanes'].items():
            green_count = lane_data.get('green', 0)
            yellow_count = lane_data.get('yellow', 0)
            
            # Ejecutar acción según el tipo de nota detectada
            if green_count > 0:
                self.handle_green_note(lane_name, green_count)
            if yellow_count > 0:
                self.handle_yellow_note(lane_name)

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
        
        # 8. Optimización: Si hay hold verde activo, saltar búsqueda de amarillas
        if self.green_hold_active[lane_name]:
            return lane_results
        
        # 9. Si hay notas verdes detectadas, saltar amarillas (optimización de juego)
        if lane_results['green_count'] > 0:
            return lane_results

        # 10. Detección de notas amarillas (solo si no hay verdes ni hold activo)
        yellow_mask = cv2.inRange(hsv_micro, self.yellow_hsv['lower'], self.yellow_hsv['upper'])
        
        # Operaciones morfológicas menos agresivas para amarillo
        close_size_yellow = max(3, close_size // 2)
        dilate_size_yellow = max(2, dilate_size // 2)
        
        close_kernel_yellow = np.ones((close_size_yellow, close_size_yellow), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel_yellow)
        
        dilate_kernel_yellow = np.ones((dilate_size_yellow, dilate_size_yellow), np.uint8)
        yellow_mask = cv2.dilate(yellow_mask, dilate_kernel_yellow, iterations=1)
        
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, open_kernel)
        
        # 11. Búsqueda de contornos amarillos
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 12. Filtrar contornos amarillos
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
                    
                    # Dibujar cajas de detección
                    for box in result['green_boxes']:
                        cv2.rectangle(output_frame, (box['x'], box['y']), 
                                    (box['x'] + box['w'], box['y'] + box['h']), (0, 255, 0), 2)
                    
                    for box in result['yellow_boxes']:
                        cv2.rectangle(output_frame, (box['x'], box['y']), 
                                    (box['x'] + box['w'], box['y'] + box['h']), (0, 255, 255), 2)
        
        # --- PROCESAMIENTO DE IA ---
        if self.input_enabled:
            self.process_ai_actions(detections)
        
        # --- DIBUJO DE POLÍGONOS Y EFECTOS ---
        overlay = output_frame.copy()
        
        # Primero, dibujar los rellenos de las teclas presionadas en el overlay
        for lane_name, points in self.polygons.items():
            if self.keys_pressed.get(lane_name, False):
                pts = np.array(points, np.int32)
                color = self.lane_colors.get(lane_name, (255, 255, 255))
                if color:
                    cv2.fillPoly(overlay, [pts], color)

        # Mezclar el overlay con el frame de salida
        alpha = 0.5  # 50% de transparencia
        cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)

        # Ahora, dibujar los contornos y etiquetas sobre el frame ya mezclado
        for lane_name, points in self.polygons.items():
            pts = np.array(points, np.int32)
            color = self.lane_colors.get(lane_name, (255, 255, 255))
            
            # Dibujar contorno
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
        overlay_fps = output_frame.copy()
        cv2.rectangle(overlay_fps, (fps_x - 5, fps_y - text_size[1] - 5), 
                    (fps_x + text_size[0] + 5, fps_y + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay_fps, 0.7, output_frame, 0.3, 0, output_frame)
        
        # Texto FPS
        cv2.putText(output_frame, fps_text, (fps_x, fps_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return output_frame, detections
    
    def add_top_counter(self, frame: np.ndarray):
        """Añadir contador superior con detecciones en tiempo real"""
        # Fondo para el contador (parte superior)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Texto del contador principal
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
        
        # Estado del sistema de IA
        if self.input_enabled:
            ai_text = "🤖 IA ACTIVA | Anti-Ban: ON"
            ai_color = (0, 255, 0)  # Verde
        else:
            ai_text = "👁️ SOLO VISUALIZACIÓN"
            ai_color = (0, 255, 255)  # Amarillo
        
        cv2.putText(frame, ai_text, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ai_color, 2)
        
        # Indicador de teclas presionadas
        pressed_keys = [lane for lane, pressed in self.keys_pressed.items() if pressed]
        if pressed_keys:
            keys_text = f"TECLAS: {', '.join(pressed_keys)}"
            cv2.putText(frame, keys_text, (frame.shape[1] - 200, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Bucle principal de ejecución del bot."""
        print("\n" + "="*50)
        print("🚀 Bot de Guitar Hero IA iniciado.")
        print("🔥 ¡Pulsaciones de teclas REALES activadas!")
        print("Presiona 'Q' en la ventana de captura para salir.")
        print("Presiona 'ESPACIO' en la ventana de captura para la función de pánico (soltar todas las teclas).")
        print("="*50 + "\n")

        window_name = 'Guitar Hero IA - Sloth Approach'
        try:
            # Crear una ventana con nombre para poder manipularla
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            with mss() as sct:
                monitors = sct.monitors
                target_monitor = None
                # El monitor [0] es el virtual que abarca todos. [1] suele ser el primario.
                # Buscamos un monitor que no esté en la posición (0,0) (secundario)
                if len(monitors) > 2:
                    for monitor in monitors[1:]:
                        if monitor['left'] != 0 or monitor['top'] != 0:
                            target_monitor = monitor
                            break
                
                # Si no se encontró un monitor secundario, usamos el primario
                if not target_monitor and len(monitors) > 1:
                    target_monitor = monitors[1]
                
                if target_monitor:
                    print(f"🖥️ Moviendo y maximizando ventana en monitor ({target_monitor['left']}, {target_monitor['top']})")
                    cv2.moveWindow(window_name, target_monitor['left'], target_monitor['top'])
                    cv2.resizeWindow(window_name, target_monitor['width'], target_monitor['height'])
                else:
                    print("⚠️ No se detectó ningún monitor. Usando tamaño por defecto.")

        except Exception as e:
            print(f"❌ Error al configurar la ventana en el monitor secundario: {e}")
            print("   Se continuará con la ventana por defecto.")


        self.screen_capture.start()
        time.sleep(1) # Dar tiempo a que la captura inicie

        try:
            while self.running:
                frame = self.screen_capture.get_latest_frame()

                if frame is not None:
                    output_frame, detections = self.process_frame(frame)
                    
                    cv2.imshow(window_name, output_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord(' '):
                    self.panic_release_all_keys()

        except KeyboardInterrupt:
            print("\n🛑 Interrupción de usuario detectada.")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Limpia todos los recursos antes de salir."""
        print("🔄 Limpiando recursos...")
        self.running = False
        self.screen_capture.stop()
        self.panic_release_all_keys() # Asegura que todas las teclas se suelten
        cv2.destroyAllWindows()
        print("✅ Limpieza completa. ¡Adiós!")


def main():
    """Función principal"""
    try:
        visualizer = PolygonVisualizer()
        visualizer.run()
        
    except Exception as e:
        print(f"\n❌ Ha ocurrido un error fatal: {e}")
        traceback.print_exc()
        input("\nPresiona ENTER para salir.")


if __name__ == "__main__":
    main() 