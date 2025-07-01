#!/usr/bin/env python3
"""
Verificador de Estado del Sistema
================================
Script para verificar configuración, componentes y estado del sistema
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_manager import ConfigManager
from src.core.screen_capture import ScreenCapture
from src.core.note_detector import NoteDetector

def check_configuration():
    """Verificar configuración del sistema"""
    print("🔧 VERIFICANDO CONFIGURACIÓN")
    print("-" * 30)
    
    config = ConfigManager()
    
    # Región de captura
    print("📱 Región de captura:")
    try:
        game_left = config.getint('CAPTURE', 'game_left', 0)
        game_top = config.getint('CAPTURE', 'game_top', 0)
        game_width = config.getint('CAPTURE', 'game_width', 0)
        game_height = config.getint('CAPTURE', 'game_height', 0)
        print(f"  Position: ({game_left}, {game_top})")
        print(f"  Size: {game_width} x {game_height}")
        
        if game_width == 0 or game_height == 0:
            print("  ⚠️ Región no configurada - ejecuta calibración")
        else:
            print("  ✅ Región configurada")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Polígonos
    print("\n🔺 Polígonos de detección:")
    try:
        polygons = config.get_note_lane_polygons_relative()
        print(f"  Carriles configurados: {len(polygons)}")
        
        total_area = 0
        for lane, points in polygons.items():
            import cv2
            import numpy as np
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                area = cv2.contourArea(pts)
                total_area += area
                print(f"    {lane}: {len(points)} puntos, {area:.0f}px²")
        
        print(f"  Área total: {total_area:.0f}px²")
        
        if len(polygons) == 6:
            print("  ✅ Todos los carriles configurados")
        else:
            print("  ⚠️ Faltan carriles - ejecuta polygon_calibrator.py")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Templates
    print("\n🖼️ Plantillas de notas:")
    try:
        detector = NoteDetector(config)
        templates = detector.templates if hasattr(detector, 'templates') else {}
        print(f"  Plantillas cargadas: {len(templates)}")
        
        for name, template in templates.items():
            if hasattr(template, 'width') and hasattr(template, 'height'):
                print(f"    {name}: {template.width}x{template.height}px")
        
        if len(templates) >= 3:
            print("  ✅ Plantillas suficientes")
        else:
            print("  ⚠️ Pocas plantillas - verifica data/templates/")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # HSV ranges
    print("\n🌈 Rangos HSV:")
    try:
        hsv_ranges = config.get_hsv_ranges()
        print(f"  Colores configurados: {len(hsv_ranges)}")
        
        for color, ranges in hsv_ranges.items():
            h_range = f"{ranges.get('h_min', 0)}-{ranges.get('h_max', 179)}"
            s_range = f"{ranges.get('s_min', 0)}-{ranges.get('s_max', 255)}"
            v_range = f"{ranges.get('v_min', 0)}-{ranges.get('v_max', 255)}"
            print(f"    {color}: H:{h_range}, S:{s_range}, V:{v_range}")
        
        if len(hsv_ranges) >= 2:
            print("  ✅ Rangos configurados")
        else:
            print("  ⚠️ Pocos rangos HSV configurados")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

def check_components():
    """Verificar componentes del sistema"""
    print("\n🧩 VERIFICANDO COMPONENTES")
    print("-" * 30)
    
    config = ConfigManager()
    
    # Screen Capture
    print("📺 Screen Capture:")
    try:
        screen_capture = ScreenCapture(config)
        screen_capture.initialize_mss()
        print("  ✅ Inicializado correctamente")
        
        # Test capture
        frame = screen_capture.capture_frame()
        if frame is not None:
            print(f"  ✅ Captura exitosa: {frame.shape}")
        else:
            print("  ❌ Error en captura")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Note Detector
    print("\n🎯 Note Detector:")
    try:
        note_detector = NoteDetector(config)
        print("  ✅ Inicializado correctamente")
        print(f"  Threshold: {note_detector.detection_threshold}")
        print(f"  Templates: {len(note_detector.templates) if hasattr(note_detector, 'templates') else 0}")
        print(f"  HSV ranges: {len(note_detector.hsv_color_ranges) if hasattr(note_detector, 'hsv_color_ranges') else 0}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

def check_files():
    """Verificar archivos importantes"""
    print("\n📁 VERIFICANDO ARCHIVOS")
    print("-" * 25)
    
    files_to_check = [
        ('config/config.ini', 'Configuración principal'),
        ('data/templates/', 'Directorio de plantillas'),
        ('data/templates/yellow_star.png', 'Plantilla amarilla'),
        ('data/templates/green_star_start.png', 'Plantilla verde inicio'),
        ('data/templates/green_star_end.png', 'Plantilla verde fin'),
        ('src/core/note_detector.py', 'Detector de notas'),
        ('src/core/screen_capture.py', 'Captura de pantalla'),
        ('src/utils/config_manager.py', 'Gestor de configuración'),
    ]
    
    for file_path, description in files_to_check:
        full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), file_path)
        
        if os.path.exists(full_path):
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                print(f"  ✅ {description}: {size} bytes")
            else:
                # Es directorio
                count = len(os.listdir(full_path)) if os.path.isdir(full_path) else 0
                print(f"  ✅ {description}: {count} archivos")
        else:
            print(f"  ❌ {description}: No encontrado")

def performance_benchmark():
    """Benchmark rápido de rendimiento"""
    print("\n⚡ BENCHMARK DE RENDIMIENTO")
    print("-" * 30)
    
    try:
        config = ConfigManager()
        screen_capture = ScreenCapture(config)
        screen_capture.initialize_mss()
        
        import time
        
        # Test de captura
        print("📺 Test de captura de pantalla:")
        capture_times = []
        for i in range(10):
            start = time.time()
            frame = screen_capture.capture_frame()
            capture_time = (time.time() - start) * 1000
            capture_times.append(capture_time)
        
        avg_capture = sum(capture_times) / len(capture_times)
        print(f"  Promedio: {avg_capture:.2f}ms")
        print(f"  Rango: {min(capture_times):.2f}ms - {max(capture_times):.2f}ms")
        
        if avg_capture < 10:
            print("  ✅ Rendimiento excelente")
        elif avg_capture < 20:
            print("  ✅ Rendimiento bueno")
        else:
            print("  ⚠️ Rendimiento lento")
        
        # Test de detección (si hay frame válido)
        if frame is not None:
            print("\n🎯 Test de detección de notas:")
            note_detector = NoteDetector(config)
            
            detection_times = []
            for i in range(5):
                start = time.time()
                detections = note_detector.detect_notes(frame)
                detection_time = (time.time() - start) * 1000
                detection_times.append(detection_time)
            
            avg_detection = sum(detection_times) / len(detection_times)
            print(f"  Promedio: {avg_detection:.2f}ms")
            print(f"  Detecciones: {len(detections) if detections else 0}")
            
            if avg_detection < 50:
                print("  ✅ Detección rápida")
            elif avg_detection < 100:
                print("  ✅ Detección aceptable")
            else:
                print("  ⚠️ Detección lenta")
        
    except Exception as e:
        print(f"  ❌ Error en benchmark: {e}")

def main():
    print("🔍 VERIFICADOR DE ESTADO DEL SISTEMA")
    print("=" * 50)
    
    check_configuration()
    check_components()
    check_files()
    performance_benchmark()
    
    print("\n✅ VERIFICACIÓN COMPLETA")
    print("Si hay errores, revisa la configuración y ejecuta las herramientas necesarias.")

if __name__ == "__main__":
    main() 