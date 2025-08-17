"""Simple performance tracker for FPS and latency, local to this approach."""

import time
from typing import Optional


class PerformanceTracker:
    """Aggregates per-frame latency and computes FPS periodically."""

    def __init__(self, report_interval_seconds: float = 1.0):
        self.report_interval_seconds = report_interval_seconds
        self._last_report_time = time.time()
        self._frame_count = 0
        self._latency_sum_ms = 0.0

    def on_frame_processed(self, latency_ms: float) -> Optional[dict]:
        """Record one frame latency. Returns metrics dict when a report is due."""
        self._frame_count += 1
        self._latency_sum_ms += latency_ms

        now = time.time()
        elapsed = now - self._last_report_time
        if elapsed >= self.report_interval_seconds:
            fps = self._frame_count / elapsed if elapsed > 0 else 0.0
            avg_latency_ms = (self._latency_sum_ms / self._frame_count) if self._frame_count else 0.0

            metrics = {
                'fps': fps,
                'avg_latency_ms': avg_latency_ms,
                'frames': self._frame_count,
                'window_s': elapsed,
            }

            # Reset window
            self._frame_count = 0
            self._latency_sum_ms = 0.0
            self._last_report_time = now

            # Lightweight console report
            print(f"[Perf] fps={fps:.1f} avg_latency_ms={avg_latency_ms:.2f}")
            return metrics

        return None



