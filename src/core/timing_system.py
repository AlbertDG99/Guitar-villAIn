"""
Timing System - Sistema de Timing
=================================

Sistema para calcular el timing preciso de cuándo presionar las teclas.
"""

import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import math


from .note_detector import Note

from utils.logger import setup_logger, performance_logger



@dataclass
class TimingEvent:  # pylint: disable=too-many-instance-attributes
    """Evento de timing para presionar una tecla"""
    lane: int
    press_time: float
    release_time: Optional[float] = None  # Para notas sostenidas
    note_id: str = ""
    confidence: float = 1.0


class TimingSystem:  # pylint: disable=too-many-instance-attributes
    """Sistema de cálculo de timing preciso"""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logger('TimingSystem')

        # Configuración de timing
        self.note_speed = config.getfloat('TIMING', 'note_speed_pixels_per_second', 400.0)
        self.perfect_window = config.getint('TIMING', 'perfect_timing_window_ms', 50) / 1000.0
        self.good_window = config.getint('TIMING', 'good_timing_window_ms', 100) / 1000.0
        self.ok_window = config.getint('TIMING', 'ok_timing_window_ms', 150) / 1000.0

        # Compensación de latencia
        self.input_latency = config.getint('TIMING', 'input_latency_compensation_ms', 10) / 1000.0
        self.audio_latency = config.getint('TIMING', 'audio_latency_compensation_ms', 20) / 1000.0

        # Zona de target (donde se deben presionar las notas)
        self.target_y = 450  # Posición Y del target

        # Estado del sistema
        self.active_notes: Dict[str, Note] = {}
        self.timing_events: List[TimingEvent] = []
        self.last_calibration_time = time.time()

        # Estadísticas
        self.timing_accuracy_history = []
        self.processed_notes = 0

        self.logger.info(f"TimingSystem initialized - Speed: {self.note_speed} px/s")
        self.logger.debug(f"Timing windows - Perfect: {self.perfect_window}s, Good: {self.good_window}s, OK: {self.ok_window}s")

    def update_notes(self, detected_notes: List[Note]) -> List[TimingEvent]:
        """
        Actualizar notas detectadas y calcular eventos de timing

        Args:
            detected_notes: Lista de notas detectadas en el frame actual

        Returns:
            Lista de eventos de timing para ejecutar
        """
        current_time = time.time()
        timing_events = []

        # Procesar nuevas notas
        for note in detected_notes:
            note_id = self._generate_note_id(note)

            if note_id not in self.active_notes:
                # Nueva nota detectada
                self.active_notes[note_id] = note

                # Calcular timing de ejecución
                timing_event = self._calculate_timing_event(note, current_time)
                if timing_event:
                    timing_events.append(timing_event)
                    self.timing_events.append(timing_event)

        # Limpiar notas obsoletas
        self._cleanup_old_notes(current_time)

        # Actualizar estadísticas
        self.processed_notes += len(detected_notes)

        return timing_events

    def _calculate_timing_event(self, note: Note, current_time: float) -> Optional[TimingEvent]:
        """
        Calcular cuándo presionar una nota específica

        Args:
            note: Nota detectada
            current_time: Tiempo actual

        Returns:
            Evento de timing o None si no se puede calcular
        """
        try:
            # Calcular distancia al target
            distance_to_target = self.target_y - note.y

            if distance_to_target <= 0:
                # La nota ya pasó el target
                return None

            # Calcular tiempo hasta el target
            time_to_target = distance_to_target / self.note_speed

            # Aplicar compensaciones de latencia
            adjusted_time = time_to_target - self.input_latency - self.audio_latency

            # Tiempo absoluto de presión
            press_time = current_time + adjusted_time

            # Para notas sostenidas, calcular tiempo de release
            release_time = None
            if note.note_type == 'sustain':
                # Estimar duración basada en altura de la nota
                sustain_duration = note.height / self.note_speed
                release_time = press_time + sustain_duration

            timing_event = TimingEvent(
                lane=note.lane,
                press_time=press_time,
                release_time=release_time,
                note_id=self._generate_note_id(note),
                confidence=note.confidence
            )

            self.logger.debug(f"Calculated timing - Lane {note.lane}: {adjusted_time:.3f}s from now")

            return timing_event

        except Exception as e:
            self.logger.error(f"Error calculating timing: {e}")
            return None

    def _generate_note_id(self, note: Note) -> str:
        """Generar ID único para una nota"""
        return f"{note.lane}_{note.x}_{note.y}_{note.timestamp:.3f}"

    def _cleanup_old_notes(self, current_time: float, max_age: float = 2.0):
        """Limpiar notas que son muy antiguas"""
        old_notes = []

        for note_id, note in self.active_notes.items():
            if current_time - note.timestamp > max_age:
                old_notes.append(note_id)

        for note_id in old_notes:
            del self.active_notes[note_id]

    def get_ready_events(self, current_time: float, lookahead: float = 0.1) -> List[TimingEvent]:
        """
        Obtener eventos que están listos para ejecutar

        Args:
            current_time: Tiempo actual
            lookahead: Tiempo de anticipación en segundos

        Returns:
            Lista de eventos listos para ejecutar
        """
        ready_events = []

        for event in self.timing_events:
            time_until_press = event.press_time - current_time

            # Evento listo si está dentro del lookahead
            if 0 <= time_until_press <= lookahead:
                ready_events.append(event)

        # Remover eventos procesados
        self.timing_events = [e for e in self.timing_events if e not in ready_events]

        return ready_events

    def calibrate_timing(self, actual_hit_time: float, expected_hit_time: float) -> float:
        """
        Calibrar timing basado en resultados reales

        Args:
            actual_hit_time: Tiempo real del hit
            expected_hit_time: Tiempo esperado del hit

        Returns:
            Error de timing en segundos
        """
        timing_error = actual_hit_time - expected_hit_time
        self.timing_accuracy_history.append(timing_error)

        # Mantener solo los últimos 100 valores
        if len(self.timing_accuracy_history) > 100:
            self.timing_accuracy_history.pop(0)

        # Calcular ajuste automático
        if len(self.timing_accuracy_history) >= 10:
            avg_error = sum(self.timing_accuracy_history) / len(self.timing_accuracy_history)

            # Ajustar compensación de latencia si hay error consistente
            if abs(avg_error) > 0.01:  # 10ms de error
                self.input_latency += avg_error * 0.1  # Ajuste gradual
                self.logger.info(f"Auto-calibrated latency: {self.input_latency:.3f}s")

        performance_logger.log_timing('timing_error', abs(timing_error) * 1000)

        return timing_error

    def get_timing_score(self, timing_error: float) -> Tuple[str, int]:
        """
        Calcular puntuación basada en error de timing

        Args:
            timing_error: Error de timing en segundos

        Returns:
            Tupla con (calificación, puntos)
        """
        abs_error = abs(timing_error)

        if abs_error <= self.perfect_window:
            return "PERFECT", 100
        elif abs_error <= self.good_window:
            return "GOOD", 75
        elif abs_error <= self.ok_window:
            return "OK", 50
        else:
            return "MISS", 0

    def predict_note_trajectory(self, note: Note, time_ahead: float = 1.0) -> Tuple[int, int]:
        """
        Predecir posición futura de una nota

        Args:
            note: Nota a predecir
            time_ahead: Tiempo de predicción en segundos

        Returns:
            Tupla con (x, y) de posición futura
        """
        # Calcular desplazamiento vertical
        vertical_displacement = self.note_speed * time_ahead

        # Nueva posición Y
        future_y = note.y + vertical_displacement

        # X se mantiene constante (movimiento vertical)
        future_x = note.x

        return (int(future_x), int(future_y))

    def get_timing_stats(self) -> Dict:
        """Obtener estadísticas de timing"""
        avg_error = 0.0
        if self.timing_accuracy_history:
            avg_error = sum(abs(e) for e in self.timing_accuracy_history) / len(self.timing_accuracy_history)

        return {
            'processed_notes': self.processed_notes,
            'active_notes_count': len(self.active_notes),
            'pending_events_count': len(self.timing_events),
            'average_timing_error_ms': avg_error * 1000,
            'input_latency_ms': self.input_latency * 1000,
            'audio_latency_ms': self.audio_latency * 1000,
            'note_speed_px_per_sec': self.note_speed
        }
