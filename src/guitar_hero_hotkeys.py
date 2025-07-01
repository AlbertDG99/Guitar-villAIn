"""
Guitar Hero IA - Control por Hotkeys
===================================

Sistema principal con control por hotkeys para doble monitor.
Evita la pausa del juego al cambiar de ventana.
"""

import time
import threading
from pathlib import Path
import sys

# Añadir path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from src.hotkey_controller import HotkeyController, create_overlay_display
from src.window_calibrator import WindowCalibrator
from src.core.screen_capture import ScreenCapture
from src.core.note_detector import NoteDetector
from src.core.timing_system import TimingSystem
from src.core.input_controller import InputController
from src.ai.dqn_agent import DQNAgent
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger


class GuitarHeroHotkeySystem:
    """Sistema principal que integra todos los componentes de Guitar Hero IA."""

    def __init__(self):
        self.logger = setup_logger('GuitarHeroSystem')
        self.config = ConfigManager()
        self.hotkey_controller = HotkeyController()

        self.running = False
        self.ai_enabled = False

        self._initialize_components()
        self._setup_hotkeys()
        self.logger.info("🎮 Guitar Hero Hotkey System inicializado")

    def _initialize_components(self):
        """Inicializa todos los componentes del sistema."""
        self.screen_capture = ScreenCapture(self.config)
        self.note_detector = NoteDetector(self.config)
        self.timing_system = TimingSystem(self.config)
        self.input_controller = InputController(self.config)
        self.ai_agent = DQNAgent(self.config)

    def _setup_hotkeys(self):
        """Configurar callbacks de hotkeys."""
        hotkeys = {
            'ctrl+alt+s': ('start_stop', self.toggle_run_state),
            'f10': ('calibrate', self.run_calibration),
            'ctrl+alt+q': ('emergency_stop', self.emergency_stop),
            'ctrl+alt+d': ('toggle_display', self.toggle_display),
        }

        for key, (action, callback) in hotkeys.items():
            self.hotkey_controller.register_hotkey(key, action, callback)

        self.hotkey_controller.start_monitoring()

    def toggle_run_state(self):
        """Inicia o detiene el bucle principal del sistema."""
        self.running = not self.running
        if self.running:
            self.logger.info("🚀 Sistema INICIADO")
            create_overlay_display("Sistema INICIADO", duration=2)
            threading.Thread(target=self._main_loop, daemon=True).start()
        else:
            self.logger.info("🛑 Sistema DETENIDO")
            create_overlay_display("Sistema DETENIDO", duration=2)

    def run_calibration(self):
        """Ejecuta el proceso de calibración de ventana."""
        self.logger.info("🎯 Iniciando calibración de ventana...")
        create_overlay_display("Iniciando calibración...", duration=2)
        calibrator = WindowCalibrator()
        calibrator.calibrate()
        self.logger.info("✅ Calibración finalizada.")
        create_overlay_display("Calibración finalizada", duration=2)

    def emergency_stop(self):
        """Detiene todo inmediatamente y libera todas las teclas presionadas."""
        self.running = False
        if self.input_controller:
            self.input_controller.emergency_release_all()
        self.logger.critical("🚨 PARADA DE EMERGENCIA 🚨")
        create_overlay_display("PARADA DE EMERGENCIA", duration=3)

    def toggle_display(self):
        """Activa o desactiva un overlay de información (función placeholder)."""
        self.logger.info("Función de mostrar/ocultar información no implementada.")
        create_overlay_display("Display (No implementado)", duration=2)

    def cleanup(self):
        """Limpia los recursos al cerrar el sistema."""
        if self.input_controller:
            self.input_controller.emergency_release_all()
        self.hotkey_controller.stop_monitoring()
        self.logger.info("✅ Limpieza completada.")

    def _main_loop(self):
        """Bucle principal de procesamiento del juego."""
        while self.running:
            start_time = time.time()

            frame = self.screen_capture.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            detected_notes = self.note_detector.detect_notes(frame)
            self.timing_system.update_notes(detected_notes)
            ready_events = self.timing_system.get_ready_events(time.time())
            self.input_controller.execute_timing_events(ready_events)

            if self.ai_enabled and self.ai_agent:
                # Aquí iría la lógica de la IA para decidir acciones
                pass

            processing_time = time.time() - start_time
            sleep_time = max(0, 0.016 - processing_time)  # Apuntar a ~60 FPS
            time.sleep(sleep_time)

    def start(self):
        """Inicia el sistema y espera comandos por hotkeys."""
        print("Sistema de Hotkeys para Guitar Hero IA activado.")
        print("Presiona las hotkeys para controlar el sistema.")
        print("Ctrl+C en la consola para salir.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nSaliendo del sistema...")
            self.cleanup()


def main():
    """Función principal para ejecutar el sistema de hotkeys."""
    system = GuitarHeroHotkeySystem()
    system.start()


if __name__ == '__main__':
    main()
