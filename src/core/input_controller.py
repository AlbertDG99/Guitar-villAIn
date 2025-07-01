"""
Input Controller - Controlador de Input
=======================================

Controlador para enviar comandos de teclado al juego.
"""

import time
import threading
from typing import List, Set, Dict, Optional
import queue


try:
    import pyautogui
except ImportError:
    pyautogui = None
    print("Warning: pyautogui not available. Input simulation disabled.")

from src.utils.logger import setup_logger, performance_logger
from .timing_system import TimingEvent



class InputController:  # pylint: disable=too-many-instance-attributes
    """Controlador de input de teclado"""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logger('InputController')

        # Configuración de teclas
        self.key_bindings = config.get_key_bindings()
        self.key_press_duration = config.getint('INPUT', 'key_press_duration_ms', 50) / 1000.0
        self.max_simultaneous_keys = config.getint('INPUT', 'simultaneous_keys_max', 6)

        # Estado del controlador
        self.pressed_keys: Set[str] = set()
        self.key_timers: Dict[str, threading.Timer] = {}
        self.input_queue: queue.PriorityQueue[TimingEvent] = queue.PriorityQueue()

        # Estadísticas
        self.keys_pressed = 0
        self.input_latency_times: List[float] = []

        # Verificar pyautogui
        if pyautogui:
            pyautogui.FAILSAFE = True  # Permitir failsafe en esquina
            pyautogui.PAUSE = 0  # Sin pausa entre comandos
            self.logger.info("PyAutoGUI initialized successfully")
        else:
            self.logger.warning("PyAutoGUI not available - input simulation disabled")

        self.logger.info("InputController initialized - Keys: %s", self.key_bindings)
        self.logger.debug("Press duration: %.3fs", self.key_press_duration)

    def execute_action(self, action: int):
        """Ejecuta una acción de la IA (presionar una tecla)."""
        if 0 <= action < 5:
            self.press_key_manual(action)
        elif action == 5:
            pass  # No hacer nada
        else:
            self.logger.warning("Acción no válida: %d", action)

    def execute_timing_events(self, timing_events: List[TimingEvent]):
        """
        Ejecutar eventos de timing programados

        Args:
            timing_events: Lista de eventos de timing a ejecutar
        """
        current_time = time.time()

        for event in timing_events:
            self._schedule_key_event(event, current_time)

    def _schedule_key_event(self, event: TimingEvent, current_time: float):
        """
        Programar ejecución de un evento de tecla

        Args:
            event: Evento de timing
            current_time: Tiempo actual
        """
        # Calcular delay hasta el momento de presión
        delay = max(0, event.press_time - current_time)

        # Obtener tecla correspondiente al carril
        if event.lane < len(self.key_bindings):
            key = self.key_bindings[event.lane]

            # Programar presión de tecla
            press_timer = threading.Timer(delay, self._press_key, args=[key, event])
            press_timer.start()

            # Programar liberación de tecla
            if event.release_time:
                # Para notas sostenidas
                release_delay = max(0, event.release_time - current_time)
                release_timer = threading.Timer(release_delay, self._release_key, args=[key])
                release_timer.start()
            else:
                # Para notas normales
                release_delay = delay + self.key_press_duration
                release_timer = threading.Timer(release_delay, self._release_key, args=[key])
                release_timer.start()

            self.logger.debug(
                "Evento de tecla '%s' programado - Presionar en %.3fs, Carril %d",
                key, delay, event.lane
            )
        else:
            self.logger.error("Carril inválido %d para bindings de teclas", event.lane)

    def _press_key(self, key: str, event: TimingEvent):
        """
        Presionar una tecla específica

        Args:
            key: Tecla a presionar
            event: Evento de timing asociado
        """
        if not pyautogui:
            self.logger.debug("[SIMULADO] Presionar tecla: %s", key)
            return

        try:
            if key in self.pressed_keys:
                self.logger.warning("La tecla %s ya está presionada", key)
                return

            pyautogui.keyDown(key)
            self.pressed_keys.add(key)
            self.keys_pressed += 1

            self.logger.debug("Tecla presionada: %s (Carril %d)", key, event.lane)

        except (pyautogui.PyAutoGUIException, TypeError, ValueError) as e:
            self.logger.error("Error presionando la tecla %s: %s", key, e)

    def _release_key(self, key: str):
        """
        Liberar una tecla específica

        Args:
            key: Tecla a liberar
        """
        if not pyautogui:
            self.logger.debug("[SIMULADO] Liberar tecla: %s", key)
            return

        try:
            if key in self.pressed_keys:
                pyautogui.keyUp(key)
                self.pressed_keys.remove(key)
                self.logger.debug("Tecla liberada: %s", key)
            else:
                self.logger.warning(
                    "Se intentó liberar la tecla %s que no estaba presionada", key
                )

        except (pyautogui.PyAutoGUIException, TypeError, ValueError) as e:
            self.logger.error("Error liberando la tecla %s: %s", key, e)

    def press_key_manual(self, lane: int, duration: Optional[float] = None):
        """
        Presionar tecla manualmente (para testing)

        Args:
            lane: Carril (0-5)
            duration: Duración de presión (opcional)
        """
        if not 0 <= lane < len(self.key_bindings):
            self.logger.error("Carril inválido: %d", lane)
            return

        key = self.key_bindings[lane]
        press_duration = duration if duration is not None else self.key_press_duration

        # Presionar inmediatamente
        self._press_key(key, TimingEvent(lane=lane, press_time=time.time()))

        # Programar liberación
        release_timer = threading.Timer(press_duration, self._release_key, args=[key])
        release_timer.start()

        self.logger.info("Pulsación manual de tecla: %s por %.3fs", key, press_duration)

    def press_chord(self, lanes: List[int], duration: Optional[float] = None):
        """
        Presionar múltiples teclas simultáneamente (acordes)

        Args:
            lanes: Lista de carriles a presionar
            duration: Duración de presión
        """
        if len(lanes) > self.max_simultaneous_keys:
            self.logger.warning("Demasiadas teclas simultáneas: %d", len(lanes))
            lanes = lanes[:self.max_simultaneous_keys]

        press_duration = duration if duration is not None else self.key_press_duration
        current_time = time.time()

        # Presionar todas las teclas
        for lane in lanes:
            if lane < len(self.key_bindings):
                event = TimingEvent(lane=lane, press_time=current_time)
                key = self.key_bindings[lane]
                self._press_key(key, event)

        # Programar liberación simultánea
        def release_all():
            for lane in lanes:
                if lane < len(self.key_bindings):
                    key = self.key_bindings[lane]
                    self._release_key(key)

        release_timer = threading.Timer(press_duration, release_all)
        release_timer.start()

        self.logger.info("Acorde presionado: carriles %s", lanes)

    def emergency_release_all(self):
        """Liberar todas las teclas en caso de emergencia"""
        if not pyautogui:
            self.pressed_keys.clear()
            self.logger.info("Liberación de emergencia: Todas las teclas [SIMULADO]")
            return

        try:
            for key in list(self.pressed_keys):
                pyautogui.keyUp(key)
            self.pressed_keys.clear()
            self.logger.info("Liberación de emergencia: Todas las teclas liberadas.")

        except (pyautogui.PyAutoGUIException, TypeError, ValueError) as e:
            self.logger.error("Error en liberación de emergencia: %s", e)

    def is_key_pressed(self, lane: int) -> bool:
        """
        Verificar si una tecla está presionada

        Args:
            lane: Carril a verificar

        Returns:
            True si la tecla está presionada
        """
        if lane < len(self.key_bindings):
            return self.key_bindings[lane] in self.pressed_keys
        return False

    def get_pressed_keys(self) -> List[str]:
        """Devuelve la lista de teclas presionadas"""
        return list(self.pressed_keys)

    def configure_failsafe(self, enabled: bool = True):
        """Activar o desactivar el failsafe de PyAutoGUI"""
        if pyautogui:
            pyautogui.FAILSAFE = enabled
            self.logger.info("Failsafe de PyAutoGUI %s", "habilitado" if enabled else "deshabilitado")

    def get_input_stats(self) -> Dict:
        """
        Obtener estadísticas del controlador de input

        Returns:
            Diccionario con estadísticas
        """
        avg_latency = 0.0
        if self.input_latency_times:
            avg_latency = sum(self.input_latency_times) / len(self.input_latency_times)

        return {
            "keys_pressed": self.keys_pressed,
            "average_latency_ms": avg_latency
        }

    def reset_stats(self):
        """Reiniciar estadísticas"""
        self.keys_pressed = 0
        self.input_latency_times.clear()
        self.logger.info("Estadísticas de input reiniciadas.")
