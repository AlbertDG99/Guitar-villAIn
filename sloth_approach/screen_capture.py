from __future__ import annotations
"""
Sistema de captura de pantalla optimizado para Guitar Hero
========================================================

Maneja la captura eficiente de frames específicos de la ventana de juego
usando múltiples métodos (MSS, PyAutoGUI) con optimizaciones de rendimiento.

Características:
- Captura Multi-método (MSS preferido, PyAutoGUI fallback)
- Calibración automática de región
- Optimizaciones de memoria y rendimiento
- Soporte para multi-monitor

Autor: Sistema Guitar Hero IA
Fecha: 2025
"""

import time
import threading
import cv2
import numpy as np
import pyautogui
import configparser
import logging

# Configuración de logging básica para ver errores
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s:%(lineno)d | %(levelname)s | %(message)s')


class ScreenCapture:
    """
    Maneja la captura de pantalla en un hilo dedicado para un rendimiento máximo.
    Funciona como un productor, capturando frames constantemente en segundo plano.
    """
    def __init__(self, config_manager):
        self.logger = logging.getLogger('ScreenCapture')
        self.config_manager = config_manager
        
        capture_config = self.config_manager.get_capture_area_config()
        if not capture_config:
            self.logger.error("❌ La configuración de captura está vacía. No se puede iniciar.")
            raise ValueError("La configuración de captura no puede ser None.")
            
        self.capture_left = int(capture_config['left'])
        self.capture_top = int(capture_config['top'])
        self.capture_width = int(capture_config['width'])
        self.capture_height = int(capture_config['height'])

        self.latest_frame = np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)
        self.frame_lock = threading.Lock()
        
        self.is_running = False
        self.capture_thread = None
        
        self.use_mss = self._check_mss_availability()

    def _check_mss_availability(self) -> bool:
        """Verifica si MSS está disponible para la captura."""
        try:
            import mss
            self.logger.info("✅ MSS está disponible y se usará para la captura.")
            return True
        except ImportError:
            self.logger.warning("⚠️ MSS no disponible, usando PyAutoGUI (más lento).")
            return False

    def start(self):
        """Inicia el hilo de captura en segundo plano."""
        if self.is_running:
            self.logger.warning("El hilo de captura ya está en ejecución.")
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.logger.info("▶️ Hilo de captura de pantalla iniciado.")

    def stop(self):
        """Detiene el hilo de captura."""
        self.is_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        self.logger.info("⏹️ Hilo de captura de pantalla detenido.")

    def _capture_loop(self):
        """El bucle principal que se ejecuta en el hilo."""
        region = {
            'left': self.capture_left,
            'top': self.capture_top,
            'width': self.capture_width,
            'height': self.capture_height
        }
        
        # --- Lógica de captura con MSS (preferido) ---
        if self.use_mss:
            try:
                import mss
                with mss.mss() as sct:
                    self.logger.info("🚀 Captura iniciada con MSS en el hilo.")
                    while self.is_running:
                        screenshot = sct.grab(region)
                        frame = np.array(screenshot)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        
                        if frame.size == 0:
                            self.logger.warning("⚠️ Frame capturado con MSS está vacío. Saltando.")
                            continue
                            
                        with self.frame_lock:
                            self.latest_frame = frame
                        
                        time.sleep(0.001)
                return # Termina la ejecución del hilo si el bucle acaba
            except Exception as e:
                self.logger.error(f"❌ Error fatal durante la captura con MSS: {e}.")
                self.is_running = False
                # No retornamos, para que pueda intentar con PyAutoGUI si sigue corriendo
        
        # --- Lógica de captura con PyAutoGUI (fallback) ---
        if not self.is_running: return # Salir si el hilo fue detenido por un error de MSS

        self.logger.info("🚀 Usando PyAutoGUI para la captura (fallback o método principal).")
        while self.is_running:
            try:
                screenshot = pyautogui.screenshot(region=(self.capture_left, self.capture_top, self.capture_width, self.capture_height))
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                if frame.size == 0:
                    self.logger.warning("⚠️ Frame capturado con PyAutoGUI está vacío. Saltando.")
                    continue
                    
                with self.frame_lock:
                    self.latest_frame = frame

            except Exception as e:
                self.logger.error(f"❌ Error en el bucle de captura con PyAutoGUI: {e}")
                self.is_running = False # Detener en caso de error grave
                break 
            time.sleep(0.001)

    def get_latest_frame(self) -> np.ndarray:
        """
        Obtiene el último frame capturado de forma segura.
        Esta es la función que llamará el entorno. Es muy rápida.
        """
        with self.frame_lock:
            return self.latest_frame.copy()

    def calibrate_region(self, display_duration=3.0):
        """Calibrar región de captura mostrando preview."""
        self.logger.info("Iniciando calibración de la región de captura...")
        self.start()
        time.sleep(1) # Dar tiempo al hilo para que capture el primer frame

        start_time = time.time()
        while time.time() - start_time < display_duration:
            frame = self.get_latest_frame()
            if frame.size == 0:
                self.logger.error("❌ No se pudo obtener frame para calibración")
                break

            # Mostrar la región
            preview = frame.copy()
            cv2.rectangle(preview, (0, 0), (self.capture_width - 1, self.capture_height - 1), (0, 255, 0), 2)
            cv2.putText(preview, "Región de Captura", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibración de Captura", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.stop()
        self.logger.info("Calibración finalizada.")

    def get_capture_info(self):
        """Obtener información actual de captura"""
        return {
            'left': self.capture_left,
            'top': self.capture_top,
            'width': self.capture_width,
            'height': self.capture_height,
            'method': 'MSS' if self.use_mss else 'PyAutoGUI',
            'active': self.is_running
        }

    def __del__(self):
        """Cleanup al destruir la instancia"""
        self.stop()

    def get_monitor_region(self):
        """Obtener región específica del monitor objetivo"""
        try:
            # Las coordenadas de calibración ya son absolutas,
            # no necesitamos sumar offset del monitor
            return {
                'left': self.capture_left,
                'top': self.capture_top,
                'width': self.capture_width,
                'height': self.capture_height
            }

        except (IndexError, AttributeError, ValueError) as error:
            self.logger.warning("⚠️ Error obteniendo región de monitor: %s", error)
            return None

    def update_region(self, left, top, width, height):
        """Actualizar región de captura"""
        self.capture_left = left
        self.capture_top = top
        self.capture_width = width
        self.capture_height = height

        # Guardar en configuración
        self.config_manager.config.set('CAPTURE', 'game_left', str(left))
        self.config_manager.config.set('CAPTURE', 'game_top', str(top))
        self.config_manager.config.set('CAPTURE', 'game_width', str(width))
        self.config_manager.config.set('CAPTURE', 'game_height', str(height))

        self.logger.info("📐 Región actualizada: %dx%d en (%d, %d)",
                        width, height, left, top)

    def get_fps(self):
        """Calcular FPS actual"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

        self.frame_count += 1
        return self.fps
