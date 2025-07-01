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
from threading import Lock

import cv2
import numpy as np
import pyautogui

from src.utils.logger import setup_logger


class ScreenCapture:  # pylint: disable=too-many-instance-attributes
    """Maneja la captura de pantalla optimizada para Guitar Hero"""

    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = setup_logger('ScreenCapture')

        # Configuración de captura
        self.capture_left = 0
        self.capture_top = 0
        self.capture_width = 1920
        self.capture_height = 1080
        self.target_fps = 60

        # Configuración de monitor
        self.target_monitor = 1
        self.monitor_width = 1920
        self.monitor_height = 1080

        # Variables de control
        self.capture_active = False
        self.use_mss = True
        self.mss_instance = None
        self.capture_lock = Lock()

        # Métricas de rendimiento
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        # Cargar configuración
        self.load_configuration()

    def load_configuration(self):
        """Cargar configuración desde config manager"""
        try:
            # Configuración de captura
            self.capture_left = int(self.config.get('CAPTURE', 'game_left', '0'))
            self.capture_top = int(self.config.get('CAPTURE', 'game_top', '0'))
            self.capture_width = int(self.config.get('CAPTURE', 'game_width', '1920'))
            self.capture_height = int(self.config.get('CAPTURE', 'game_height', '1080'))

            # Configuración de monitor
            self.target_monitor = int(self.config.get('CAPTURE', 'target_monitor', '1'))
            self.monitor_width = int(self.config.get('CAPTURE', 'monitor_width', '1920'))
            self.monitor_height = int(self.config.get('CAPTURE', 'monitor_height', '1080'))

            # Configuración de rendimiento
            self.target_fps = int(self.config.get('PERFORMANCE', 'target_fps', '60'))

            self.logger.info("✅ Configuración de captura cargada")

        except (ValueError, KeyError) as error:
            self.logger.error("❌ Error cargando configuración: %s", error)
            self.logger.info("🔧 Usando valores por defecto")

    def initialize_mss(self):
        """Inicializar MSS para captura rápida"""
        try:
            import mss as mss_lib  # pylint: disable=import-outside-toplevel
            self.mss_instance = mss_lib.mss()
            self.use_mss = True
            self.logger.info("✅ MSS inicializado correctamente")
            return True

        except ImportError:
            self.logger.warning("⚠️ MSS no disponible, usando PyAutoGUI")
            self.use_mss = False
            return False

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

    def capture_frame(self):
        """Capturar frame actual usando método preferido"""
        with self.capture_lock:
            if self.use_mss:
                return self._capture_with_mss()
            return self._capture_with_pyautogui()

    def _capture_with_mss(self):
        """Captura usando MSS (método más rápido)"""
        try:
            if not self.mss_instance:
                if not self.initialize_mss():
                    return None

            region = self.get_monitor_region()
            if not region or not self.mss_instance:
                return None

            # Capturar usando MSS
            screenshot = self.mss_instance.grab(region)

            # Convertir a array numpy
            frame = np.array(screenshot)

            # Convertir BGRA a BGR
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            return frame

        except (ImportError, AttributeError, ValueError) as error:
            self.logger.warning("⚠️ Error captura MSS: %s, cambiando a PyAutoGUI", error)
            self.use_mss = False
            return self._capture_with_pyautogui()

    def _capture_with_pyautogui(self):
        """Captura usando PyAutoGUI (método fallback)"""
        try:
            # Capturar región específica
            screenshot = pyautogui.screenshot(region=(
                self.capture_left,
                self.capture_top,
                self.capture_width,
                self.capture_height
            ))

            # Convertir a array numpy y BGR
            frame = np.array(screenshot)
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return frame

        except (ImportError, AttributeError, ValueError) as error:
            self.logger.error("❌ Error captura PyAutoGUI: %s", error)
            return None

    def calibrate_region(self, display_duration=3.0):
        """Calibrar región de captura mostrando preview"""
        try:
            self.logger.info("🎯 Iniciando calibración de región...")

            # Capturar frame de prueba
            frame = self.capture_frame()
            if frame is None:
                self.logger.error("❌ No se pudo capturar frame para calibración")
                return False

            # Mostrar región calibrada
            cv2.imshow('Region Calibrada - Presiona cualquier tecla', frame)
            cv2.waitKey(int(display_duration * 1000))
            cv2.destroyAllWindows()

            self.logger.info("✅ Calibración completada")
            return True

        except (ImportError, AttributeError, ValueError) as error:
            self.logger.error("❌ Error en calibración: %s", error)
            return False

    def update_region(self, left, top, width, height):
        """Actualizar región de captura"""
        self.capture_left = left
        self.capture_top = top
        self.capture_width = width
        self.capture_height = height

        # Guardar en configuración
        self.config.set('CAPTURE', 'game_left', str(left))
        self.config.set('CAPTURE', 'game_top', str(top))
        self.config.set('CAPTURE', 'game_width', str(width))
        self.config.set('CAPTURE', 'game_height', str(height))

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

    def start_capture(self):
        """Iniciar sistema de captura"""
        if not self.capture_active:
            self.capture_active = True
            self.initialize_mss()
            self.logger.info("▶️ Sistema de captura iniciado")

    def stop_capture(self):
        """Detener sistema de captura"""
        if self.capture_active:
            self.capture_active = False
            if self.mss_instance:
                self.mss_instance.close()
                self.mss_instance = None
            self.logger.info("⏹️ Sistema de captura detenido")

    def get_capture_info(self):
        """Obtener información actual de captura"""
        return {
            'region': {
                'left': self.capture_left,
                'top': self.capture_top,
                'width': self.capture_width,
                'height': self.capture_height
            },
            'monitor': self.target_monitor,
            'method': 'MSS' if self.use_mss else 'PyAutoGUI',
            'fps': self.get_fps(),
            'active': self.capture_active
        }

    def __del__(self):
        """Cleanup al destruir la instancia"""
        try:
            self.stop_capture()
        except:  # pylint: disable=bare-except
            pass
