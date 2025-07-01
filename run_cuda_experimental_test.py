#!/usr/bin/env python3
"""
🚀 CUDA Experimental Test - Prueba de Rendimiento GPU vs CPU
===========================================================

Test del detector experimental con CUDA basado en:
https://opencv.org/platforms/cuda/

Compara rendimiento entre:
- Operaciones CPU tradicionales
- Operaciones CUDA cuando están disponibles
- Métricas detalladas de rendimiento
"""

import sys
import os
import cv2
import time
import numpy as np
from pathlib import Path

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from cuda_experimental_detector import CudaExperimentalDetector
    from utils.config_manager import ConfigManager
    from utils.logger import setup_logger
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("💡 Asegúrate de que el proyecto esté configurado correctamente")
    sys.exit(1)

def display_banner():
    """Muestra banner informativo"""
    print("🚀 CUDA EXPERIMENTAL TEST")
    print("=" * 60)
    print()
    print("📋 INFORMACIÓN DEL TEST:")
    print("   • Basado en documentación oficial OpenCV CUDA")
    print("   • Implementa cv::cuda::GpuMat y operaciones nativas")  
    print("   • Fallback automático a CPU")
    print("   • Métricas comparativas GPU vs CPU")
    print()
    
    # Verificar estado CUDA
    print("🔍 VERIFICACIÓN CUDA:")
    print(f"   • OpenCV Version: {cv2.__version__}")
    print(f"   • Módulo CUDA: {'✅' if hasattr(cv2, 'cuda') else '❌'}")
    
    if hasattr(cv2, 'cuda'):
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"   • CUDA Devices: {device_count} {'✅' if device_count > 0 else '❌'}")
        
        if device_count > 0:
            print("   • GPU Info:")
            try:
                device_info = cv2.cuda.DeviceInfo(0)
                print(f"     - Nombre: {device_info.name()}")
                print(f"     - Memoria: {device_info.totalGlobalMem() / (1024**3):.1f} GB")
                print(f"     - Compute: {device_info.majorVersion()}.{device_info.minorVersion()}")
            except Exception as e:
                print(f"     - Error obteniendo info: {e}")
    
    # Verificar PyTorch CUDA para comparación
    try:
        import torch
        print(f"   • PyTorch CUDA: {'✅' if torch.cuda.is_available() else '❌'}")
        if torch.cuda.is_available():
            print(f"     - Devices: {torch.cuda.device_count()}")
            print(f"     - GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("   • PyTorch: No instalado")
    
    print()

def run_performance_test(detector: CudaExperimentalDetector, test_duration: int = 30):
    """
    Ejecuta test de rendimiento por tiempo determinado
    
    Args:
        detector: Instancia del detector CUDA
        test_duration: Duración del test en segundos
    """
    print(f"🚀 INICIANDO TEST DE RENDIMIENTO ({test_duration}s)")
    print("=" * 60)
    print()
    print("📊 Presiona 'q' para salir antes de tiempo")
    print("📊 El test mostrará métricas cada 5 segundos")
    print()
    
    start_time = time.time()
    frame_count = 0
    last_stats_time = start_time
    
    # Variables para métricas locales
    local_times = []
    
    try:
        while time.time() - start_time < test_duration:
            current_time = time.time()
            
            # Capturar frame
            frame_start = time.time()
            frame = detector.screen_capture.capture_frame()
            if frame is None:
                continue
                
            # Detectar usando CUDA
            detections = detector.detect_frame_cuda(frame)
            frame_time = (time.time() - frame_start) * 1000
            
            local_times.append(frame_time)
            frame_count += 1
            
            # Mostrar estadísticas cada 5 segundos
            if current_time - last_stats_time >= 5.0:
                show_interim_stats(detector, frame_count, current_time - start_time, local_times)
                last_stats_time = current_time
                local_times = []  # Reset para próximo intervalo
            
            # Visualización opcional (comentado para máximo rendimiento)
            # display_frame = visualize_detections(frame, detections)
            # cv2.imshow('CUDA Test', display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️ Test interrumpido por usuario")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Test interrumpido (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error durante test: {e}")
    
    finally:
        total_time = time.time() - start_time
        show_final_stats(detector, frame_count, total_time)

def show_interim_stats(detector: CudaExperimentalDetector, frame_count: int, 
                      elapsed_time: float, recent_times: list):
    """Muestra estadísticas intermedias"""
    stats = detector.get_performance_stats()
    if not stats:
        return
        
    avg_recent = np.mean(recent_times) if recent_times else 0
    fps = frame_count / elapsed_time
    
    print(f"📊 ESTADÍSTICAS ({elapsed_time:.1f}s):")
    print(f"   • Frames procesados: {frame_count}")
    print(f"   • FPS promedio: {fps:.1f}")
    print(f"   • Tiempo reciente: {avg_recent:.1f}ms")
    print(f"   • CUDA disponible: {'✅' if stats['cuda_available'] else '❌'}")
    print(f"   • Operaciones GPU: {stats['gpu_operations']}")
    print(f"   • Operaciones CPU: {stats['cpu_operations']}")
    
    if stats['gpu_operations'] > 0:
        print(f"   • Tiempo GPU: {stats['avg_gpu_time_ms']:.1f}ms")
        print(f"   • Speedup GPU: {stats['gpu_speedup']:.2f}x")
    
    print()

def show_final_stats(detector: CudaExperimentalDetector, frame_count: int, total_time: float):
    """Muestra estadísticas finales completas"""
    stats = detector.get_performance_stats()
    
    print("\n🏁 RESULTADOS FINALES")
    print("=" * 60)
    print()
    
    # Estadísticas generales
    print("📊 RENDIMIENTO GENERAL:")
    print(f"   • Duración total: {total_time:.1f}s")
    print(f"   • Frames procesados: {frame_count}")
    print(f"   • FPS promedio: {frame_count / total_time:.1f}")
    print(f"   • Detecciones totales: {stats.get('total_detections', 0)}")
    print()
    
    # Estadísticas CUDA vs CPU
    print("🖥️ ANÁLISIS CUDA vs CPU:")
    print(f"   • CUDA disponible: {'✅' if stats.get('cuda_available', False) else '❌'}")
    print(f"   • Operaciones GPU: {stats.get('gpu_operations', 0)}")
    print(f"   • Operaciones CPU: {stats.get('cpu_operations', 0)}")
    
    if stats.get('gpu_operations', 0) > 0:
        print(f"   • Tiempo promedio GPU: {stats.get('avg_gpu_time_ms', 0):.1f}ms")
        print(f"   • Tiempo promedio CPU: {stats.get('avg_cpu_time_ms', 0):.1f}ms")
        print(f"   • Speedup GPU: {stats.get('gpu_speedup', 1.0):.2f}x")
        
        # Análisis de beneficios
        speedup = stats.get('gpu_speedup', 1.0)
        if speedup > 1.5:
            print("   ✅ GPU proporciona aceleración significativa")
        elif speedup > 1.1:
            print("   ⚡ GPU proporciona aceleración moderada")
        else:
            print("   ⚠️ GPU no proporciona beneficios significativos")
    else:
        print("   ❌ No se ejecutaron operaciones GPU")
        print("   💡 Posibles causas:")
        print("     - OpenCV sin soporte CUDA")
        print("     - No se detectaron dispositivos CUDA")
        print("     - Error en inicialización GPU")
    
    print()
    
    # Recomendaciones
    print("💡 RECOMENDACIONES:")
    if stats.get('cuda_available', False):
        if stats.get('gpu_speedup', 1.0) > 1.5:
            print("   ✅ Continuar usando CUDA - excelente aceleración")
        else:
            print("   ⚡ CUDA funcional pero beneficios limitados")
    else:
        print("   🔧 Considerar instalar OpenCV con soporte CUDA")
        print("   📖 Ver sección 'Configuración GPU/CUDA' del README")
    print()

def visualize_detections(frame: np.ndarray, detections) -> np.ndarray:
    """Visualiza las detecciones en el frame (opcional)"""
    vis_frame = frame.copy()
    
    for detection in detections:
        x, y, w, h = detection.bbox
        color = (0, 255, 255) if detection.color_type == 'yellow' else (0, 255, 0)
        
        # Rectángulo
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
        
        # Información
        text = f"{detection.color_type[:1].upper()}{detection.lane.upper()}"
        text += f" {detection.confidence:.2f}"
        text += f" {'G' if detection.used_gpu else 'C'}"
        
        cv2.putText(vis_frame, text, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return vis_frame

def main():
    """Función principal del test"""
    display_banner()
    
    # Verificar configuración
    config_manager = ConfigManager()
    if not config_manager.get('CAPTURE', 'game_left'):
        print("❌ No se encontró configuración de captura")
        print("💡 Ejecuta primero: python src/guitar_hero_main.py")
        print("   Y realiza la calibración de ventana (Opción 1)")
        return
    
    # Crear detector experimental
    try:
        print("🔧 INICIALIZANDO DETECTOR CUDA...")
        detector = CudaExperimentalDetector(
            confidence_threshold=0.95,
            max_workers=6
        )
        print("✅ Detector inicializado correctamente")
        print()
        
    except Exception as e:
        print(f"❌ Error inicializando detector: {e}")
        return
    
    # Preguntar duración del test
    try:
        duration = input("⏱️ Duración del test en segundos (30): ").strip()
        duration = int(duration) if duration else 30
        duration = max(10, min(duration, 300))  # Entre 10s y 5min
    except ValueError:
        duration = 30
    
    print(f"🚀 Test configurado para {duration} segundos")
    print("Presiona Enter para continuar...")
    input()
    
    # Ejecutar test
    try:
        run_performance_test(detector, duration)
    finally:
        detector.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 