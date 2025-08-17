"""Centralized logging system for Guitar Hero AI."""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional
from colorama import Fore, Back, Style, init


init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Color formatter for console"""

    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Back.WHITE
    }

    def format(self, record: logging.LogRecord) -> str:
        """Applies color formatting to the log record."""
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
    Configure a logger with custom formatting.

    Args:
        name (str): Logger name.
        log_level (str): Logging level (DEBUG, INFO, etc.).
        log_file (str): Path to log file (optional).

    Returns:
        logging.Logger: Configured logger instance.
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
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)

        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class PerformanceLogger:
    """Specialized logger for performance metrics."""

    def __init__(self, name: str = "Performance"):
        self.logger = setup_logger(name, log_file="logs/performance.log")
        self.metrics: Dict[str, List[float]] = {}
        self.last_report_s: float = 0.0

    def log_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log a performance metric."""
        self.logger.info("METRIC | %s: %.4f %s", metric_name, value, unit)
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def log_timing(self, operation: str, duration_ms: float):
        """Log the time of an operation in milliseconds."""
        self.log_metric(f"timing_{operation}", duration_ms, "ms")

    def log_accuracy(self, accuracy: float):
        """Log accuracy as percentage."""
        self.log_metric("accuracy", accuracy * 100, "%")

    def log_fps(self, fps: float):
        """Log frames per second (FPS)."""
        self.log_metric("fps", fps, "fps")

    def maybe_console_report(self, min_interval_s: float = 1.0):
        """Occasional concise console report of averages."""
        import time
        now = time.time()
        if now - self.last_report_s < min_interval_s:
            return
        self.last_report_s = now
        fps_avg = self.get_average("fps")
        lat_avg = self.get_average("latency_ms")
        if fps_avg or lat_avg:
            print(f"[Perf] fps={fps_avg:.1f} avg_latency_ms={lat_avg:.2f}")

    def get_average(self, metric_name: str) -> float:
        """Get the average of a logged metric."""
        values = self.metrics.get(metric_name)
        return sum(values) / len(values) if values else 0.0

    def clear_metrics(self):
        """Clear all stored metrics."""
        self.metrics.clear()


class GameLogger:
    """Specialized logger for game events."""

    def __init__(self, name: str = "GameEvents"):
        self.logger = setup_logger(name, log_file="logs/game_events.log")

    def log_note_detected(self, lane: int, note_type: str, confidence: float):
        """Log the detection of a note."""
        self.logger.debug(
            "DETECT | Lane %d | %s | Confidence: %.2f",
            lane, note_type, confidence
        )

    def log_note_played(self, lane: int, timing_score: str, points: int):
        """Log a note played by the player."""
        self.logger.info(
            "PLAY | Lane %d | %s | Points: %d",
            lane, timing_score, points
        )

    def log_combo(self, combo_count: int):
        """Log the combo counter."""
        if combo_count > 0 and combo_count % 10 == 0:
            self.logger.info("COMBO | %d consecutive notes!", combo_count)

    def log_score(self, current_score: int, max_score: int):
        """Log the current score."""
        percentage = (
            current_score /
            max_score *
            100) if max_score > 0 else 0.0
        self.logger.info(
            "SCORE | %d/%d (%.1f%%)",
            current_score, max_score, percentage
        )

    def log_song_complete(
            self,
            final_score: int,
            accuracy: float,
            max_combo: int):
        """Log the completion of a song."""
        self.logger.info(
            "COMPLETE | Score: %d | Accuracy: %.1f%% | Max Combo: %d",
            final_score, accuracy, max_combo
        )


performance_logger = PerformanceLogger()
game_logger = GameLogger()
