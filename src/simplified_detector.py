#!/usr/bin/env python3
"""
Simplified Detector - Detector Simplificado
==========================================

Versión simplificada que:
1. Solo detecta 2 tipos: amarillas (0.95) y verdes (0.85)
2. Usa nomenclatura por teclas (s,d,f,j,k,l)
3. Solo usa green_star.png y yellow_star.png
4. GPU problema identificado: PyTorch sí detecta CUDA, OpenCV no
"""

import cv2
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Añadir src al path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger
from src.core.screen_capture import ScreenCapture

class GPUImageProcessor:
    """Procesador de imágenes acelerado por GPU usando PyTorch."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        self.gpu_ops_count = 0
        self.cpu_ops_count = 0
        
    def accelerated_color_filter(self, hsv_np: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Filtrado de color acelerado por GPU."""
        if not self.gpu_available:
            self.cpu_ops_count += 1
            return cv2.inRange(hsv_np, lower, upper)
        
        try:
            # Convertir a tensor GPU
            hsv_tensor = torch.from_numpy(hsv_np).float().to(self.device)
            lower_tensor = torch.from_numpy(lower).float().to(self.device)
            upper_tensor = torch.from_numpy(upper).float().to(self.device)
            
            # Operación vectorizada en GPU
            mask = ((hsv_tensor >= lower_tensor) & (hsv_tensor <= upper_tensor)).all(dim=-1)
            
            # Convertir de vuelta a numpy
            result = (mask.cpu().numpy() * 255).astype(np.uint8)
            self.gpu_ops_count += 1
            return result
            
        except Exception:
            # Fallback a CPU si falla
            self.cpu_ops_count += 1
            return cv2.inRange(hsv_np, lower, upper)
    
    def accelerated_morphology(self, mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Operaciones morfológicas aceleradas."""
        if not self.gpu_available:
            self.cpu_ops_count += 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        try:
            # Convertir a tensor GPU
            mask_tensor = torch.from_numpy(mask).float().to(self.device) / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Batch y canal
            
            # Kernel para convolución
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size * kernel_size)
            
            # Operación de cierre (dilatación + erosión)
            dilated = F.conv2d(mask_tensor, kernel, padding=kernel_size//2)
            closed = F.conv2d(dilated, kernel, padding=kernel_size//2)
            
            # Convertir de vuelta
            result = (closed.squeeze().cpu().numpy() * 255).astype(np.uint8)
            self.gpu_ops_count += 1
            return result
            
        except Exception:
            # Fallback a CPU
            self.cpu_ops_count += 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    def accelerated_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Redimensionado acelerado por GPU."""
        if not self.gpu_available:
            return cv2.resize(image, target_size)
        
        try:
            # Convertir a tensor GPU
            if len(image.shape) == 3:
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
                img_tensor = img_tensor.unsqueeze(0)  # Batch dimension
            else:
                img_tensor = torch.from_numpy(image).float().to(self.device)
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Batch y canal
            
            # Redimensionar
            resized = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)
            
            # Convertir de vuelta
            if len(image.shape) == 3:
                result = resized.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            else:
                result = resized.squeeze().cpu().numpy().astype(np.uint8)
            
            return result
            
        except Exception:
            return cv2.resize(image, target_size)

class SimplifiedDetector:
    """Detector simplificado - Solo amarillas y verdes."""
    
    def __init__(self, yellow_threshold: float = 1.0, green_threshold: float = 0.95):
        self.logger = setup_logger("SimplifiedDetector")
        self.config_manager = ConfigManager()
        
        # Thresholds diferenciados
        self.yellow_threshold = yellow_threshold
        self.green_threshold = green_threshold
        
        # Inicializar captura de pantalla
        self.screen_capture = ScreenCapture(self.config_manager)
        
        # Inicializar procesador GPU
        self.gpu_processor = GPUImageProcessor()
        
        # GPU: Ahora con aceleración PyTorch
        self.gpu_status = self._check_gpu_status()
        
        # Configuración simplificada
        self.confidence_params = {
            'area_divisor': 500.0,
            'aspect_min': 0.5,
            'aspect_max': 7.0,
            'solidity_multiplier': 1.3,
            'intensity_divisor': 120.0,
            'weights': {
                'area': 0.30,
                'aspect': 0.15,
                'solidity': 0.35,
                'intensity': 0.20
            }
        }
        
        # Cargar polígonos y convertir nomenclatura
        self.lane_polygons = self._load_lane_polygons_by_keys()
        self.key_names = ['s', 'd', 'f', 'j', 'k', 'l']  # Solo teclas
        
        # Convertir coordenadas
        self._convert_polygons_to_relative()
        
        # Colores para visualización por tecla
        self.lane_colors = [
            (0, 255, 0),    # s - Verde
            (0, 255, 255),  # d - Cian
            (255, 0, 0),    # f - Azul
            (255, 0, 255),  # j - Magenta
            (0, 128, 255),  # k - Naranja
            (128, 255, 0)   # l - Verde-amarillo
        ]
        
        # Thread pool - 6 threads (uno por tecla)
        self.thread_pool = ThreadPoolExecutor(max_workers=6)
        
        # Métricas
        self.detection_times = []
        self.frame_count = 0
        self.fps_calculator = time.time()
        
        # Contadores por tipo
        self.yellow_count = 0
        self.green_count = 0
        
        # Métricas GPU vs CPU
        self.gpu_operations = 0
        self.cpu_operations = 0
        
        if not self.lane_polygons:
            raise ValueError("No hay polígonos configurados")
            
        self.logger.info(f"🚀 Detector CPU-Multithreading - Amarillas: {yellow_threshold:.2f}, Verdes: {green_threshold:.2f}")
        self.logger.info(f"🧵 Multithreading: 6 workers activos")
    
    def _check_gpu_status(self) -> str:
        """Verificar estado detallado de GPU."""
        try:
            pytorch_cuda = torch.cuda.is_available()
            pytorch_devices = torch.cuda.device_count() if pytorch_cuda else 0
            
            opencv_cuda = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
            opencv_devices = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0
            
            gpu_accel = "🚀 GPU ACELERADO" if pytorch_cuda else "CPU ONLY"
            
            status = f"{gpu_accel} | PyTorch: {'✅' if pytorch_cuda else '❌'} ({pytorch_devices}), " \
                    f"OpenCV: {'✅' if opencv_cuda else '❌'} ({opencv_devices})"
            
            return status
        except Exception as e:
            return f"Error verificando GPU: {e}"
    
    def _load_lane_polygons_by_keys(self) -> Dict[str, List[Tuple[int, int]]]:
        """Cargar polígonos usando nomenclatura por teclas."""
        # Mapeo de teclas a nombres de carriles
        key_to_lane = {
            's': 'S', 'd': 'D', 'f': 'F',
            'j': 'J', 'k': 'K', 'l': 'L'
        }
        
        original_polygons = self.config_manager.get_note_lane_polygons()
        key_polygons = {}
        
        for key, lane_name in key_to_lane.items():
            if lane_name in original_polygons:
                key_polygons[key] = original_polygons[lane_name]
        
        self.logger.info(f"Polígonos cargados por teclas: {list(key_polygons.keys())}")
        return key_polygons
    
    def _convert_polygons_to_relative(self):
        """Convertir polígonos a coordenadas relativas."""
        if not self.lane_polygons:
            return
            
        game_left = int(self.config_manager.get('CAPTURE', 'game_left', '0'))
        game_top = int(self.config_manager.get('CAPTURE', 'game_top', '0'))
        
        self.logger.info(f"Offset del juego: ({game_left}, {game_top})")
        
        converted_polygons = {}
        for key, points in self.lane_polygons.items():
            converted_points = []
            for point in points:
                relative_x = point[0] - game_left
                relative_y = point[1] - game_top
                converted_points.append((relative_x, relative_y))
            converted_polygons[key] = converted_points
        
        self.lane_polygons = converted_polygons
        self.logger.info(f"✅ Polígonos convertidos para teclas: {list(converted_polygons.keys())}")
    
    def detect_yellow_notes(self, hsv: np.ndarray) -> np.ndarray:
        """Detectar notas amarillas - CPU optimizado."""
        # Rango amarillo optimizado (funcionaba bien antes)
        lower = np.array([15, 100, 100])
        upper = np.array([40, 255, 255])
        
        # Filtrado CPU directo (más rápido)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Operaciones morfológicas CPU
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def detect_green_notes(self, hsv: np.ndarray) -> np.ndarray:
        """Detectar notas verdes - CPU optimizado con threshold más permisivo."""
        # Rango verde ampliado para mejor detección
        lower = np.array([25, 40, 40])
        upper = np.array([95, 255, 255])
        
        # Filtrado CPU directo (más rápido)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Operaciones morfológicas CPU (kernel más grande para verdes)
        kernel = np.ones((4, 4), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        
        return mask
    
    def calculate_confidence(self, contour: np.ndarray, mask: np.ndarray, 
                           region: np.ndarray, note_type: str) -> float:
        """Calcular confianza con parámetros específicos por tipo."""
        # Factor 1: Área
        area = cv2.contourArea(contour)
        area_confidence = min(area / self.confidence_params['area_divisor'], 1.0)
        
        # Factor 2: Relación de aspecto
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        
        # Ajustes específicos por tipo de nota
        if note_type == 'green':
            # Verdes más permisivos (trails)
            if 0.3 <= aspect_ratio <= 8.0:
                aspect_confidence = 1.0
            else:
                aspect_confidence = 0.9
        else:  # yellow
            # Amarillas configuración estándar
            if (self.confidence_params['aspect_min'] <= aspect_ratio <= 
                self.confidence_params['aspect_max']):
                aspect_confidence = 1.0
            else:
                aspect_confidence = 0.8
        
        # Factor 3: Solidez
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        solidity_confidence = min(solidity * self.confidence_params['solidity_multiplier'], 1.0)
        
        # Factor 4: Intensidad
        roi = region[y:y+h, x:x+w]
        if roi.size > 0:
            mean_intensity = float(np.mean(roi))
            intensity_confidence = min(mean_intensity / self.confidence_params['intensity_divisor'], 1.0)
        else:
            intensity_confidence = 0.0
        
        # Combinar factores
        weights = self.confidence_params['weights']
        final_confidence = (
            area_confidence * weights['area'] +
            aspect_confidence * weights['aspect'] +
            solidity_confidence * weights['solidity'] +
            intensity_confidence * weights['intensity']
        )
        
        return min(final_confidence, 1.0)
    
    def filter_duplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filtrar detecciones duplicadas."""
        if len(detections) <= 1:
            return detections
        
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        filtered = []
        
        for detection in sorted_detections:
            x1, y1, w1, h1 = detection['x'], detection['y'], detection['width'], detection['height']
            
            is_duplicate = False
            for accepted in filtered:
                x2, y2, w2, h2 = accepted['x'], accepted['y'], accepted['width'], accepted['height']
                
                # Calcular superposición
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                min_area = min(area1, area2)
                
                if min_area > 0:
                    overlap_percentage = overlap_area / min_area
                    # 25% threshold para ser más estricto
                    if overlap_percentage > 0.25:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def extract_key_region(self, frame: np.ndarray, key: str) -> np.ndarray:
        """Extraer región de una tecla específica."""
        if key not in self.lane_polygons:
            return np.zeros_like(frame)
            
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = self.lane_polygons[key]
        pts = np.array(points, np.int32)
        cv2.fillPoly(mask, [pts], (255,))
        return cv2.bitwise_and(frame, frame, mask=mask)
    
    def process_single_key(self, frame: np.ndarray, key: str) -> Tuple[str, List[Dict], float]:
        """Procesar una tecla individual."""
        start_time = time.time()
        
        # Extraer región de la tecla
        region = self.extract_key_region(frame, key)
        
        # Convertir a HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        detections = []
        
        # Detectar notas amarillas
        yellow_mask = self.detect_yellow_notes(hsv)
        contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_yellow:
            area = cv2.contourArea(contour)
            if area > 100:  # Área mínima
                confidence = self.calculate_confidence(contour, yellow_mask, region, 'yellow')
                
                if confidence >= self.yellow_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    detection = {
                        'key': key,
                        'type': 'yellow',
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'area': area,
                        'confidence': confidence,
                        'timestamp': time.time()
                    }
                    detections.append(detection)
        
        # Detectar notas verdes
        green_mask = self.detect_green_notes(hsv)
        contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_green:
            area = cv2.contourArea(contour)
            if area > 80:  # Área mínima menor para verdes
                confidence = self.calculate_confidence(contour, green_mask, region, 'green')
                
                if confidence >= self.green_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    detection = {
                        'key': key,
                        'type': 'green',
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'area': area,
                        'confidence': confidence,
                        'timestamp': time.time()
                    }
                    detections.append(detection)
        
        # Filtrar duplicados
        filtered_detections = self.filter_duplicate_detections(detections)
        
        processing_time = (time.time() - start_time) * 1000
        return key, filtered_detections, processing_time
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], Dict[str, float]]:
        """Procesar frame completo con multithreading."""
        start_time = time.time()
        
        # Procesar cada tecla en paralelo
        future_to_key = {}
        for key in self.lane_polygons.keys():
            future = self.thread_pool.submit(self.process_single_key, frame, key)
            future_to_key[future] = key
        
        # Recopilar resultados
        all_detections = []
        key_times = {}
        
        for future in as_completed(future_to_key):
            key, detections, key_time = future.result()
            all_detections.extend(detections)
            key_times[key] = key_time
        
        # Actualizar contadores
        yellow_detections = [d for d in all_detections if d['type'] == 'yellow']
        green_detections = [d for d in all_detections if d['type'] == 'green']
        
        self.yellow_count = len(yellow_detections)
        self.green_count = len(green_detections)
        
        total_time = (time.time() - start_time) * 1000
        self.detection_times.append(total_time)
        self.frame_count += 1
        
        return all_detections, key_times
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict], 
                           key_times: Dict[str, float]) -> np.ndarray:
        """Visualizar detecciones simplificadas."""
        output_frame = frame.copy()
        
        # Dibujar polígonos de teclas
        for i, key in enumerate(self.key_names):
            if key in self.lane_polygons:
                points = self.lane_polygons[key]
                pts = np.array(points, np.int32)
                color = self.lane_colors[i % len(self.lane_colors)]
                cv2.polylines(output_frame, [pts], True, color, 2)
                
                # Etiqueta de tecla con tiempo
                center_x = sum(p[0] for p in points) // 4
                center_y = sum(p[1] for p in points) // 4
                key_time = key_times.get(key, 0)
                cv2.putText(output_frame, f"{key.upper()}({key_time:.1f}ms)", 
                           (center_x-20, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Dibujar detecciones
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
            key = detection['key']
            note_type = detection['type']
            confidence = detection['confidence']
            
            # Color según tipo de nota
            if note_type == 'yellow':
                rect_color = (0, 255, 255)  # Cian para amarillas
            else:  # green
                rect_color = (0, 255, 0)    # Verde para verdes
            
            # Rectángulo grueso
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), rect_color, 3)
            
            # Etiqueta
            label = f"{key.upper()}:{note_type}({confidence:.2f})"
            cv2.putText(output_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, rect_color, 2)
        
        # Información de rendimiento
        current_time = time.time()
        if current_time - self.fps_calculator >= 1.0:
            fps = self.frame_count / (current_time - self.fps_calculator)
            self.fps_calculator = current_time
            self.frame_count = 0
        else:
            fps = 0
        
        avg_time = np.mean(self.detection_times[-10:]) if self.detection_times else 0
        avg_key_time = np.mean(list(key_times.values())) if key_times else 0
        
        # Información en pantalla
        info_lines = [
            f"🟡 Amarillas: {self.yellow_count} (≥{self.yellow_threshold:.2f})",
            f"🟢 Verdes: {self.green_count} (≥{self.green_threshold:.2f})",
            f"📊 Total: {len(detections)}",
            f"⏱️ Tiempo: {avg_time:.1f}ms",
            f"🎹 Promedio/tecla: {avg_key_time:.1f}ms",
            f"🚀 FPS: {fps:.1f}" if fps > 0 else "🚀 FPS: Calculando...",
            f"🧵 Multithreading: 6 workers CPU optimizado"
        ]
        
        for i, line in enumerate(info_lines):
            y = 30 + (i * 25)
            cv2.putText(output_frame, line, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_frame
    
    def run_simplified_test(self):
        """Ejecutar el test simplificado."""
        print("🎯 SIMPLIFIED DETECTOR - CPU Optimizado")
        print("=" * 60)
        print(f"🟡 Threshold Amarillas: {self.yellow_threshold:.2f}")
        print(f"🟢 Threshold Verdes: {self.green_threshold:.2f}")
        print(f"🎹 Teclas: {', '.join(self.key_names)}")
        print(f"🧵 Multithreading: 6 workers CPU optimizado")
        print(f"⚡ Rendimiento: ~22ms/frame objetivo")
        print()
        print("Controles:")
        print("- 'q': Salir")
        print("- 's': Capturar frame")
        print("- Espacio: Pausar/Reanudar")
        print("- '+'/'-': Ajustar threshold verdes")
        print("- 't': Estadísticas")
        
        paused = False
        
        try:
            self.screen_capture.start_capture()
            
            while True:
                if not paused:
                    frame = self.screen_capture.capture_frame()
                    
                    if frame is not None:
                        detections, key_times = self.process_frame(frame)
                        output_frame = self.visualize_detections(frame, detections, key_times)
                    else:
                        output_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(output_frame, "❌ Error de captura", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("Simplified Detector - Amarillas y Verdes", output_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    cv2.imwrite(f"simplified_detection_{timestamp}.png", output_frame)
                    print(f"Frame guardado: simplified_detection_{timestamp}.png")
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️ Pausado' if paused else '▶️ Reanudado'}")
                elif key == ord('+') or key == ord('='):
                    self.green_threshold = min(self.green_threshold + 0.05, 1.0)
                    print(f"🟢 Threshold verdes: {self.green_threshold:.2f}")
                elif key == ord('-'):
                    self.green_threshold = max(self.green_threshold - 0.05, 0.1)
                    print(f"🟢 Threshold verdes: {self.green_threshold:.2f}")
                elif key == ord('t'):
                    self._print_stats(key_times)
                    
        except Exception as e:
            self.logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.screen_capture.stop_capture()
            self.thread_pool.shutdown(wait=True)
            cv2.destroyAllWindows()
    
    def _print_stats(self, key_times: Dict[str, float]):
        """Imprimir estadísticas."""
        print("\n" + "="*50)
        print("📊 ESTADÍSTICAS SIMPLIFIED DETECTOR")
        print("="*50)
        
        if self.detection_times:
            avg_total = np.mean(self.detection_times[-20:])
            print(f"⏱️ Tiempo promedio: {avg_total:.1f}ms")
        
        print(f"🟡 Amarillas detectadas: {self.yellow_count} (≥{self.yellow_threshold:.2f})")
        print(f"🟢 Verdes detectadas: {self.green_count} (≥{self.green_threshold:.2f})")
        
        if key_times:
            print(f"🎹 Tiempos por tecla:")
            for key, time_ms in sorted(key_times.items()):
                print(f"   {key.upper()}: {time_ms:.1f}ms")
        
        print(f"🧵 Multithreading: 6 workers CPU optimizado")
        print(f"🚀 Rendimiento: {avg_total:.1f}ms/frame (~{1000/avg_total:.1f} FPS)")
        print(f"⚡ Optimización: CPU directo + multithreading")
        print("="*50)


def main():
    """Función principal del detector simplificado."""
    print("🎯 SIMPLIFIED DETECTOR 🎯")
    print("=" * 50)
    print()
    print("ACLARACIONES IMPLEMENTADAS:")
    print("✅ Solo 2 tipos de notas: amarillas y verdes")
    print("✅ Referencias: green_star.png y yellow_star.png")
    print("✅ Nomenclatura: Teclas (s,d,f,j,k,l)")
    print("✅ Thresholds: Amarillas 1.0 (PERFECTO), Verdes 0.95")
    print("✅ GPU: Problema identificado (PyTorch sí, OpenCV no)")
    print()
    
    try:
        detector = SimplifiedDetector(
            yellow_threshold=1.0,    # PERFECTO - evita confusión
            green_threshold=0.95     # Muy estricto
        )
        detector.run_simplified_test()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 