#!/usr/bin/env python3
"""
Benchmark Rápido de FPS
=======================
Script simple para medir rendimiento del detector principal
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import statistics
from collections import deque
from src.utils.config_manager import ConfigManager
from src.core.screen_capture import ScreenCapture
from src.core.note_detector import NoteDetector

def quick_fps_test(duration_seconds=10):
    """Test rápido de FPS sin ventana"""
    print("⚡ BENCHMARK RÁPIDO DE FPS")
    print("=" * 30)
    
    # Inicializar componentes
    config = ConfigManager()
    screen_capture = ScreenCapture(config)
    note_detector = NoteDetector(config)
    
    try:
        screen_capture.initialize_mss()
        print("✅ Componentes inicializados")
    except Exception as e:
        print(f"❌ Error inicializando: {e}")
        return
    
    # Métricas
    frame_times = deque(maxlen=1000)
    detection_times = deque(maxlen=1000)
    total_detections = 0
    
    print(f"\n🚀 Ejecutando test por {duration_seconds} segundos...")
    print("📊 Estadísticas cada 2 segundos:")
    
    start_time = time.time()
    last_report = start_time
    frame_count = 0
    
    while time.time() - start_time < duration_seconds:
        # Medir captura
        capture_start = time.time()
        frame = screen_capture.capture_frame()
        capture_time = (time.time() - capture_start) * 1000
        
        if frame is None:
            continue
            
        # Medir detección
        detection_start = time.time()
        detections = note_detector.detect_notes(frame)
        detection_time = (time.time() - detection_start) * 1000
        
        # Guardar métricas
        total_time = capture_time + detection_time
        frame_times.append(total_time)
        detection_times.append(detection_time)
        total_detections += len(detections) if detections else 0
        frame_count += 1
        
        # Reportar cada 2 segundos
        current_time = time.time()
        if current_time - last_report >= 2.0:
            elapsed = current_time - start_time
            recent_times = list(frame_times)[-100:]  # Últimos 100 frames
            avg_time = statistics.mean(recent_times) if recent_times else 0
            fps = 1000 / avg_time if avg_time > 0 else 0
            
            print(f"  [{elapsed:.1f}s] FPS: {fps:.1f} | Frame: {avg_time:.1f}ms | Detecciones: {total_detections}")
            last_report = current_time
    
    # Estadísticas finales
    total_time = time.time() - start_time
    
    if not frame_times:
        print("❌ No se procesaron frames")
        return
    
    print(f"\n🏁 RESULTADOS FINALES:")
    print(f"  ⏱️ Duración: {total_time:.1f}s")
    print(f"  📈 Frames procesados: {frame_count}")
    print(f"  📊 FPS promedio: {frame_count/total_time:.1f}")
    
    # Estadísticas de tiempo
    avg_frame_time = statistics.mean(frame_times)
    min_frame_time = min(frame_times)
    max_frame_time = max(frame_times)
    median_frame_time = statistics.median(frame_times)
    
    print(f"\n⚡ TIEMPOS DE FRAME:")
    print(f"  • Promedio: {avg_frame_time:.1f}ms ({1000/avg_frame_time:.1f} FPS)")
    print(f"  • Mediana: {median_frame_time:.1f}ms ({1000/median_frame_time:.1f} FPS)")
    print(f"  • Mejor: {min_frame_time:.1f}ms ({1000/min_frame_time:.1f} FPS)")
    print(f"  • Peor: {max_frame_time:.1f}ms ({1000/max_frame_time:.1f} FPS)")
    
    # Estadísticas de detección
    if detection_times:
        avg_detection_time = statistics.mean(detection_times)
        print(f"\n🎯 DETECCIÓN:")
        print(f"  • Tiempo promedio: {avg_detection_time:.1f}ms")
        print(f"  • Total detectadas: {total_detections}")
        print(f"  • Rate: {total_detections/total_time:.1f} detecciones/s")
    
    # Clasificación de rendimiento
    fps_effective = 1000 / avg_frame_time
    print(f"\n🏆 EVALUACIÓN DE RENDIMIENTO:")
    
    if fps_effective >= 30:
        print("  ⭐⭐⭐ EXCELENTE - Juego fluido")
    elif fps_effective >= 20:
        print("  ⭐⭐ BUENO - Rendimiento aceptable")
    elif fps_effective >= 10:
        print("  ⭐ REGULAR - Rendimiento limitado")
    else:
        print("  ❌ POBRE - Necesita optimización")
    
    print(f"  📈 FPS objetivo: 30+ (actual: {fps_effective:.1f})")

def compare_with_baseline():
    """Comparar con rendimiento baseline esperado"""
    print(f"\n📋 COMPARACIÓN CON BASELINE:")
    print(f"  🎯 FPS objetivo: 30 FPS (33.3ms/frame)")
    print(f"  ⚡ FPS aceptable: 20 FPS (50ms/frame)")
    print(f"  🐌 FPS mínimo: 10 FPS (100ms/frame)")

def main():
    print("🎮 Benchmark Rápido del Sistema Guitar Hero IA")
    print("=" * 50)
    
    print("📝 Este test mide el rendimiento sin interfaz gráfica")
    print("💡 Asegúrate de que Guitar Hero esté abierto para mejores resultados")
    
    # Opciones de duración
    print(f"\n⏱️ Duración del test:")
    print(f"  1. Rápido (5 segundos)")
    print(f"  2. Normal (10 segundos)")  
    print(f"  3. Completo (30 segundos)")
    
    choice = input(f"\nElige opción (1-3, default=2): ").strip() or "2"
    
    duration_map = {"1": 5, "2": 10, "3": 30}
    duration = duration_map.get(choice, 10)
    
    print(f"\n🚀 Iniciando benchmark de {duration} segundos...")
    
    try:
        quick_fps_test(duration)
        compare_with_baseline()
        
    except KeyboardInterrupt:
        print(f"\n👋 Benchmark interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 