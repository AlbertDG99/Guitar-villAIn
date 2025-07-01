"""
Hotkey Controller - Controlador de Teclas Rápidas
================================================

Sistema para controlar Guitar Hero IA sin cambiar de ventana del juego.
"""

import sys
import time
import threading
from typing import Callable, Dict, Optional

import keyboard

from src.utils.logger import setup_logger
from src.utils.overlay import create_overlay_display


class HotkeyController:
    """Controlador de hotkeys para acciones del sistema."""

    def __init__(self):
        self.logger = setup_logger('HotkeyController')
        self.active = False
        self.hotkeys: Dict[str, str] = {}
        self.callbacks: Dict[str, Callable] = {}
        self.monitor_thread: Optional[threading.Thread] = None

        # Hotkeys por defecto
        self.default_hotkeys = {
            'ctrl+alt+s': 'start',
            'ctrl+alt+p': 'pause',
            'ctrl+alt+q': 'stop',
            'ctrl+alt+i': 'info'
        }
        self.register_default_hotkeys()

    def register_hotkey(self, hotkey: str, action: str, callback: Callable):
        """Registrar un hotkey para una acción con un callback."""
        self.hotkeys[hotkey] = action
        self.callbacks[action] = callback
        try:
            keyboard.add_hotkey(hotkey, self._on_hotkey, args=(action,))
            self.logger.info("Hotkey '%s' registrado para la acción '%s'", hotkey, action)
        except ImportError:
            self.logger.error("La librería 'keyboard' no está instalada. "
                              "No se pueden registrar hotkeys.")

    def register_default_hotkeys(self):
        """Registrar hotkeys por defecto que emiten eventos."""
        for hotkey, action in self.default_hotkeys.items():
            self.register_hotkey(hotkey, action, self._default_callback)

    def _default_callback(self, action: str):
        """Callback por defecto que loguea la acción."""
        self.logger.info("Acción por defecto ejecutada: %s", action)
        # Aquí se podría emitir un evento a un sistema principal
        if action == 'start' and 'start' in self.callbacks:
            self.callbacks['start']()
        elif action == 'stop' and 'stop' in self.callbacks:
            self.callbacks['stop']()

    def _on_hotkey(self, action: str):
        """Función que se ejecuta al presionar un hotkey."""
        if action in self.callbacks:
            self.logger.info("Hotkey presionado para acción: '%s'", action)
            self.callbacks[action]()
        else:
            self.logger.warning("No hay callback para la acción '%s'", action)

    def start_monitoring(self):
        """Iniciar el monitoreo de hotkeys en un hilo separado."""
        if not self.active:
            self.active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Monitoreo de hotkeys iniciado.")

    def stop_monitoring(self):
        """Detener el monitoreo de hotkeys."""
        if self.active:
            self.active = False
            self.logger.info("Monitoreo de hotkeys detenido.")

    def _monitor_loop(self):
        """
        Bucle de monitoreo que se ejecuta en un hilo.
        Mantiene el hilo vivo para que los hotkeys funcionen en segundo plano.
        """
        while self.active:
            time.sleep(0.1)

    def is_active(self) -> bool:
        """Verificar si el monitoreo está activo."""
        return self.active


def main():
    """Función principal para probar el controlador de hotkeys."""
    controller = HotkeyController()

    def start_action():
        """Acción de ejemplo para iniciar."""
        print("Iniciando sistema...")
        create_overlay_display("Sistema Iniciado")

    def stop_action():
        """Acción de ejemplo para detener."""
        print("Deteniendo sistema...")
        create_overlay_display("Sistema Detenido")
        controller.stop_monitoring()
        sys.exit(0)

    controller.register_hotkey('ctrl+alt+1', 'start', start_action)
    controller.register_hotkey('ctrl+alt+0', 'stop', stop_action)

    print("Hotkey controller en modo de prueba.")
    print("  Ctrl+Alt+1 para iniciar.")
    print("  Ctrl+Alt+0 para detener.")

    controller.start_monitoring()

    try:
        while controller.is_active():
            time.sleep(1)
    except KeyboardInterrupt:
        controller.stop_monitoring()
        print("\nTest terminado.")

if __name__ == "__main__":
    main()
