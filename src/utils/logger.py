"""
Logger - Sistema de Logging
============================

Sistema de logging centralizado para Guitar Hero IA.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional
from colorama import Fore, Back, Style, init


# Inicializar colorama para Windows
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Formatter con colores para la consola"""

    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Back.WHITE
    }

    def format(self, record: logging.LogRecord) -> str:
        """Aplica formato de color al registro de log."""
        color = self.COLORS.get(record.levelno, '')
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"

        formatter = logging.Formatter(
            '%(asctime)s | %(name)s:%(lineno)d | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        return formatter.format(record)


def setup_logger(
    name: str, log_level: str = 'INFO', log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configurar un logger con formato personalizado.

    Args:
        name (str): Nombre del logger.
        log_level (str): Nivel de logging (DEBUG, INFO, etc.).
        log_file (str): Ruta al archivo de log (opcional).

    Returns:
        logging.Logger: Instancia del logger configurado.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)

        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class PerformanceLogger:
    """Logger especializado para métricas de rendimiento."""

    def __init__(self, name: str = "Performance"):
        self.logger = setup_logger(name, log_file="logs/performance.log")
        self.metrics: Dict[str, List[float]] = {}

    def log_metric(self, metric_name: str, value: float, unit: str = ""):
        """Registrar una métrica de rendimiento."""
        self.logger.info("METRIC | %s: %.4f %s", metric_name, value, unit)
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def log_timing(self, operation: str, duration_ms: float):
        """Registrar el tiempo de una operación en milisegundos."""
        self.log_metric(f"timing_{operation}", duration_ms, "ms")

    def log_accuracy(self, accuracy: float):
        """Registrar la precisión (accuracy) como porcentaje."""
        self.log_metric("accuracy", accuracy * 100, "%")

    def log_fps(self, fps: float):
        """Registrar los frames por segundo (FPS)."""
        self.log_metric("fps", fps, "fps")

    def get_average(self, metric_name: str) -> float:
        """Obtener el promedio de una métrica registrada."""
        values = self.metrics.get(metric_name)
        return sum(values) / len(values) if values else 0.0

    def clear_metrics(self):
        """Limpiar todas las métricas almacenadas."""
        self.metrics.clear()


class GameLogger:
    """Logger especializado para eventos del juego."""

    def __init__(self, name: str = "GameEvents"):
        self.logger = setup_logger(name, log_file="logs/game_events.log")

    def log_note_detected(self, lane: int, note_type: str, confidence: float):
        """Registrar la detección de una nota."""
        self.logger.debug(
            "DETECT | Lane %d | %s | Confidence: %.2f",
            lane, note_type, confidence
        )

    def log_note_played(self, lane: int, timing_score: str, points: int):
        """Registrar una nota tocada por el jugador."""
        self.logger.info(
            "PLAY | Lane %d | %s | Points: %d",
            lane, timing_score, points
        )

    def log_combo(self, combo_count: int):
        """Registrar el contador de combo."""
        if combo_count > 0 and combo_count % 10 == 0:
            self.logger.info("COMBO | %d notas consecutivas!", combo_count)

    def log_score(self, current_score: int, max_score: int):
        """Registrar la puntuación actual."""
        percentage = (current_score / max_score * 100) if max_score > 0 else 0.0
        self.logger.info(
            "SCORE | %d/%d (%.1f%%)",
            current_score, max_score, percentage
        )

    def log_song_complete(self, final_score: int, accuracy: float, max_combo: int):
        """Registrar la finalización de una canción."""
        self.logger.info(
            "COMPLETE | Score: %d | Accuracy: %.1f%% | Max Combo: %d",
            final_score, accuracy, max_combo
        )


# Instancias globales para fácil acceso
performance_logger = PerformanceLogger()
game_logger = GameLogger()
